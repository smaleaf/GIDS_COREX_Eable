#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  SYNC_TO_GITHUB.sh
#  内网 Bitbucket → GitHub 双向同步脚本
#
#  用法：
#    bash /root/GIDS_cufile_github/SYNC_TO_GITHUB.sh "提交说明"
#
#  原理：
#    用 git format-patch 提取内部仓库新增 commits，
#    再用 git am 应用到 GitHub 镜像仓库，避免因
#    filter-repo 造成的历史分叉问题。
#
#  内部仓库：/root/GIDS_cufile       → Bitbucket (origin/master)
#  GitHub 镜像：/root/GIDS_cufile_github → GitHub   (origin/main)
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

INTERNAL_REPO="/root/GIDS_cufile"
GITHUB_REPO="/root/GIDS_cufile_github"
SYNC_TRACKER="$GITHUB_REPO/.last_sync_internal_commit"
PATCH_DIR="/tmp/gids_sync_patches"

# ── 颜色输出 ─────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()      { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
die()     { echo -e "${RED}[FAIL]${NC}  $*" >&2; exit 1; }
section() { echo -e "\n${BOLD}${YELLOW}══ $* ══${NC}"; }

# ── 参数处理 ─────────────────────────────────────────────────
COMMIT_MSG="${1:-}"
if [ -z "$COMMIT_MSG" ]; then
    COMMIT_MSG="sync: $(date '+%Y-%m-%d %H:%M') 自动同步"
    warn "未提供提交说明，使用默认：\"$COMMIT_MSG\""
fi

# ── Step 1: 内部仓库提交推送 ─────────────────────────────────
section "Step 1 / 3  内部仓库 (Bitbucket)"
cd "$INTERNAL_REPO" || die "找不到内部仓库：$INTERNAL_REPO"

info "当前目录：$(pwd)"
info "当前分支：$(git branch --show-current)"

if git diff --quiet && git diff --cached --quiet && \
   [ -z "$(git ls-files --others --exclude-standard)" ]; then
    ok "工作区干净，无需提交"
else
    info "暂存所有改动..."
    git add -A
    info "提交：$COMMIT_MSG"
    git commit -m "$COMMIT_MSG"
fi

info "推送到 Bitbucket..."
git push origin master
ok "Bitbucket 推送完成"

NEW_HEAD=$(git rev-parse HEAD)

# ── Step 2: 生成增量补丁 ─────────────────────────────────────
section "Step 2 / 3  生成增量补丁"

if [ ! -f "$SYNC_TRACKER" ]; then
    die "未找到同步基线文件 $SYNC_TRACKER，请手动初始化：\n  cd $INTERNAL_REPO && git rev-parse HEAD > $SYNC_TRACKER"
fi

LAST_SYNC=$(cat "$SYNC_TRACKER" | tr -d '[:space:]')
info "上次同步基线：$LAST_SYNC"
info "本次最新 HEAD：$NEW_HEAD"

if [ "$LAST_SYNC" = "$NEW_HEAD" ]; then
    ok "内部仓库无新提交，无需同步 GitHub"
    exit 0
fi

# 检查基线 commit 是否存在于内部仓库
cd "$INTERNAL_REPO"
if ! git cat-file -e "${LAST_SYNC}^{commit}" 2>/dev/null; then
    die "基线 commit $LAST_SYNC 在内部仓库中不存在，请重置同步基线：\n  git -C $INTERNAL_REPO rev-parse HEAD > $SYNC_TRACKER"
fi

# 生成补丁文件
rm -rf "$PATCH_DIR" && mkdir -p "$PATCH_DIR"
PATCH_COUNT=$(git format-patch "${LAST_SYNC}..HEAD" \
    --output-directory "$PATCH_DIR" \
    -- . ':!GIDS/data/' ':!*.npy' ':!*.bin' ':!*.whl' \
    | wc -l)

if [ "$PATCH_COUNT" -eq 0 ]; then
    ok "无实质性文件变更（可能仅有数据文件变更），跳过 GitHub 同步"
    # 更新基线
    echo "$NEW_HEAD" > "$SYNC_TRACKER"
    exit 0
fi

info "生成了 $PATCH_COUNT 个补丁文件："
ls "$PATCH_DIR"/*.patch 2>/dev/null | while read f; do echo "    $f"; done

# ── Step 3: 应用补丁并推送 GitHub ───────────────────────────
section "Step 3 / 3  应用补丁并推送 GitHub"
cd "$GITHUB_REPO" || die "找不到 GitHub 仓库：$GITHUB_REPO"

info "应用补丁到 GitHub 镜像..."
if ! git am --ignore-whitespace --3way "$PATCH_DIR"/*.patch; then
    warn "git am 应用补丁失败，可能有冲突，正在回退..."
    git am --abort 2>/dev/null || true
    die "补丁应用失败，请手动处理冲突后再运行脚本"
fi

# 更新 .last_sync_internal_commit（记录新基线）
echo "$NEW_HEAD" > "$SYNC_TRACKER"
git add .last_sync_internal_commit
git commit --amend --no-edit 2>/dev/null || true

info "推送到 GitHub..."
git push origin main
ok "GitHub 推送完成"

# 清理补丁文件
rm -rf "$PATCH_DIR"

# ── 完成汇总 ─────────────────────────────────────────────────
section "同步完成"
echo -e "  ${GREEN}Bitbucket${NC}  →  $(cd "$INTERNAL_REPO" && git remote get-url origin)"
echo -e "  ${GREEN}GitHub${NC}     →  $(cd "$GITHUB_REPO"   && git remote get-url origin)"
echo ""
echo -e "  同步 commit 数：${BOLD}${PATCH_COUNT}${NC}"
echo -e "  最新 commit：${BOLD}$(cd "$GITHUB_REPO" && git log -1 --oneline)${NC}"
echo ""
ok "两份仓库已同步 ✓"
