#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  SYNC_TO_GITHUB.sh
#  内网 Bitbucket → GitHub 双向同步脚本
#
#  用法：
#    bash /root/GIDS_cufile_github/SYNC_TO_GITHUB.sh "提交说明"
#
#  效果：
#    1. 在内部仓库提交所有改动并推送到 Bitbucket
#    2. 将最新 master 同步到 GitHub 的 main 分支
#
#  内部仓库：/root/GIDS_cufile       → Bitbucket (origin/master)
#  GitHub 镜像：/root/GIDS_cufile_github → GitHub   (origin/main)
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

INTERNAL_REPO="/root/GIDS_cufile"
GITHUB_REPO="/root/GIDS_cufile_github"

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
section "Step 1 / 2  内部仓库 (Bitbucket)"
cd "$INTERNAL_REPO" || die "找不到内部仓库：$INTERNAL_REPO"

info "当前目录：$(pwd)"
info "当前分支：$(git branch --show-current)"

# 有改动才提交
if git diff --quiet && git diff --cached --quiet && [ -z "$(git ls-files --others --exclude-standard)" ]; then
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

# ── Step 2: 同步到 GitHub ────────────────────────────────────
section "Step 2 / 2  GitHub 镜像同步"
cd "$GITHUB_REPO" || die "找不到 GitHub 仓库：$GITHUB_REPO"

info "当前目录：$(pwd)"
info "拉取内部仓库最新 master → 本地 main..."
git fetch "$INTERNAL_REPO" master:main

info "推送到 GitHub..."
git push origin main
ok "GitHub 推送完成"

# ── 完成汇总 ─────────────────────────────────────────────────
section "同步完成"
echo -e "  ${GREEN}Bitbucket${NC}  →  $(cd "$INTERNAL_REPO" && git remote get-url origin)"
echo -e "  ${GREEN}GitHub${NC}     →  $(cd "$GITHUB_REPO"   && git remote get-url origin)"
echo ""
echo -e "  最新 commit：${BOLD}$(cd "$GITHUB_REPO" && git log -1 --oneline)${NC}"
echo ""
ok "两份仓库已同步 ✓"
