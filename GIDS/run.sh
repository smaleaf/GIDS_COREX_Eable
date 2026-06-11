#!/bin/bash
# ==============================================================
# GIDS-IX 自动化脚本 (Iluvatar GPU / cuFile)
# ==============================================================
# 用法:  bash run.sh <command> [args...]
#
# 命令:
#   setup          构建 IXFeatureStore C++ 模块
#   prepare-data   下载数据集并写入 SSD 特征文件
#   train          运行 GNN 训练
#   verify         验证所有组件是否就绪
#   all            一键执行: setup -> prepare-data -> train
#   clean          清理构建产物
# ==============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"

# ============================================================
# 环境配置
# ============================================================
COREX_ENABLE="/home/corex/sw_home_1/sw_home/enable"
COREX_LIB64="/home/corex/sw_home_1/sw_home/local/corex/lib64"
USR_LOCAL_LIB64="/usr/local/lib64"

# ============================================================
# 默认参数
# ============================================================
GPU_IDX="${GPU_IDX:-0}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
DATASET="${DATASET:-ogbn-products}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/data}"
FEAT_FILE="${FEAT_FILE:-/mnt/nvme0/node_feat.bin}"
NUM_SSD="${NUM_SSD:-1}"
PAGE_SIZE="${PAGE_SIZE:-4096}"
CACHE_SIZE="${CACHE_SIZE:-10}"
FANOUT="${FANOUT:-10}"
N_LAYERS="${N_LAYERS:-3}"
N_HIDDEN="${N_HIDDEN:-256}"
LR="${LR:-0.003}"
DROPOUT="${DROPOUT:-0.5}"
NUM_WORKERS="${NUM_WORKERS:-0}"
CPU_BUFFER_SIZE="${CPU_BUFFER_SIZE:-0}"
ACCUMULATOR="${ACCUMULATOR:-True}"
WB_SIZE="${WB_SIZE:-40}"
PAD_FACTOR="${PAD_FACTOR:-1}"

# ============================================================
# 颜色输出
# ============================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${BLUE}[GIDS-IX]${NC} $1"; }
log_ok()    { echo -e "${GREEN}[GIDS-IX]${NC} ✅ $1"; }
log_warn()  { echo -e "${YELLOW}[GIDS-IX]${NC} ⚠️  $1"; }
log_error() { echo -e "${RED}[GIDS-IX]${NC} ❌ $1"; }

# ============================================================
# 环境激活
# ============================================================
activate_env() {
    if [ -f "${COREX_ENABLE}" ]; then
        source "${COREX_ENABLE}"
    else
        log_warn "SDK enable script not found: ${COREX_ENABLE}"
    fi
    export LD_LIBRARY_PATH="${COREX_LIB64}:${USR_LOCAL_LIB64}:${LD_LIBRARY_PATH}"
}

# ============================================================
# setup - 构建 IXFeatureStore 模块
# ============================================================
cmd_setup() {
    log_info "========== 构建 IXFeatureStore 模块 =========="

    activate_env

    cd "${PROJECT_DIR}/gids_module_ix"
    bash build_ix.sh

    log_ok "构建完成"
    log_info "模块位置: ${PROJECT_DIR}/GIDS_Setup/GIDS/IXFeatureStore*.so"
}

# ============================================================
# verify - 验证所有组件
# ============================================================
cmd_verify() {
    log_info "========== 环境验证 =========="
    activate_env

    # 1. ixsmi
    if ixsmi &>/dev/null; then
        log_ok "ixsmi 可用"
        ixsmi 2>&1 | grep -E "GPU|IX-ML|Driver|Memory" || true
    else
        log_error "ixsmi 不可用 - 请检查 Corex 驱动安装"
        return 1
    fi

    # 2. Python
    local PYTHON="$(which python3)"
    log_info "Python: ${PYTHON} ($(python3 --version 2>&1))"

    # 3. PyTorch + GPU
    local torch_ok=$(python3 -c "
import torch
t = torch.zeros(3,3, device='cuda:${GPU_IDX}')
print('OK')
" 2>&1)
    if echo "$torch_ok" | grep -q "OK"; then
        log_ok "PyTorch + cuda:${GPU_IDX} 正常"
    else
        log_error "PyTorch GPU 测试失败:"
        echo "$torch_ok"
        return 1
    fi

    # 4. IXFeatureStore
    local ixfs_ok=$(python3 -c "
import sys
sys.path.insert(0, '${PROJECT_DIR}/GIDS_Setup/GIDS')
import IXFeatureStore
fs = IXFeatureStore.IXFeatureStore_float()
print('OK')
" 2>&1)
    if echo "$ixfs_ok" | grep -q "OK"; then
        log_ok "IXFeatureStore 模块可加载"
    else
        log_error "IXFeatureStore 模块加载失败:"
        echo "$ixfs_ok"
        return 1
    fi

    # 5. cuFile
    python3 -c "
import sys
sys.path.insert(0, '${PROJECT_DIR}/GIDS_Setup/GIDS')
import IXFeatureStore
fs = IXFeatureStore.IXFeatureStore_float()
fs.init_controllers(1, ['/tmp/__gids_verify__.bin'], 4096, 0, 10000, 0)
fs.print_stats()
" 2>&1 | grep -E "cuFile mode|WARNING"
    rm -f /tmp/__gids_verify__.bin

    # 6. 依赖库
    log_info "检查动态库依赖..."
    for lib in libcudart.so libcufile.so libcupti.so libcuinfer.so.7; do
        if ldconfig -p 2>/dev/null | grep -q "$lib" || find "${COREX_LIB64}" "${USR_LOCAL_LIB64}" -name "$lib" 2>/dev/null | grep -q .; then
            log_ok "$lib"
        else
            log_warn "$lib 未找到"
        fi
    done

    # 7. DGL
    python3 -c "import dgl; print('DGL', dgl.__version__)" 2>&1 && log_ok "DGL 可用" || log_warn "DGL 未安装"

    # 8. 特征文件
    if [ -f "${FEAT_FILE}" ]; then
        local feat_size=$(stat -c%s "${FEAT_FILE}" 2>/dev/null || echo "?")
        log_ok "特征文件: ${FEAT_FILE} (${feat_size} bytes)"
    else
        log_warn "特征文件不存在: ${FEAT_FILE} (请运行 prepare-data)"
    fi

    echo ""
    log_ok "========== 验证完成 =========="
}

# ============================================================
# prepare-data - 下载数据集并写入 SSD
# ============================================================
cmd_prepare_data() {
    log_info "========== 准备数据 =========="
    activate_env

    mkdir -p "${DATA_DIR}"

    # 下载 OGB 数据集
    log_info "下载数据集: ${DATASET}"
    python3 -c "
import os
os.makedirs('${DATA_DIR}', exist_ok=True)

if '${DATASET}' == 'ogbn-products':
    from ogb.nodeproppred import DglNodePropPredDataset
    d = DglNodePropPredDataset(name='${DATASET}', root='${DATA_DIR}')
    graph, labels = d[0]
    feat = graph.ndata['feat']
    print(f'Nodes: {graph.number_of_nodes()}')
    print(f'Features: {feat.shape}')
    print(f'Labels: {labels.shape}')
    feat = feat.numpy().ravel().astype('float32')
    feat.tofile('${FEAT_FILE}')
    print(f'Features written to ${FEAT_FILE} ({feat.nbytes / 1024**2:.1f} MB)')
elif '${DATASET}' == 'ogbn-papers100M':
    from ogb.nodeproppred import DglNodePropPredDataset
    d = DglNodePropPredDataset(name='${DATASET}', root='${DATA_DIR}')
    graph, labels = d[0]
    feat = graph.ndata['feat']
    print(f'Nodes: {graph.number_of_nodes()}')
    print(f'Features: {feat.shape}')
    feat = feat.numpy().ravel().astype('float32')
    feat.tofile('${FEAT_FILE}')
    print(f'Features written to ${FEAT_FILE} ({feat.nbytes / 1024**3:.2f} GB)')
else:
    raise ValueError(f'Unknown dataset: ${DATASET}')
" 2>&1

    log_ok "数据准备完成: ${FEAT_FILE}"
}

# ============================================================
# train - 运行训练
# ============================================================
cmd_train() {
    log_info "========== 开始训练 =========="
    activate_env

    # 检查特征文件
    if [ ! -f "${FEAT_FILE}" ]; then
        log_error "特征文件不存在: ${FEAT_FILE}"
        log_info "请先运行: bash run.sh prepare-data"
        return 1
    fi

    cd "${PROJECT_DIR}"

    export PYTHONPATH="${PROJECT_DIR}/GIDS_Setup/GIDS:${PYTHONPATH}"

    local cmd=(
        python3 evaluation/homogenous_train_ix.py
        --dataset "${DATASET}"
        --data_dir "${DATA_DIR}"
        --gpu "${GPU_IDX}"
        --epochs "${EPOCHS}"
        --batch_size "${BATCH_SIZE}"
        --num_ssd "${NUM_SSD}"
        --page_size "${PAGE_SIZE}"
        --cache_size "${CACHE_SIZE}"
        --fanout "${FANOUT}"
        --n_layers "${N_LAYERS}"
        --n_hidden "${N_HIDDEN}"
        --lr "${LR}"
        --dropout "${DROPOUT}"
        --num_workers "${NUM_WORKERS}"
        --cpu_buffer_size "${CPU_BUFFER_SIZE}"
        --accumulator "${ACCUMULATOR}"
        --wb_size "${WB_SIZE}"
        --pad_factor "${PAD_FACTOR}"
        --file_paths "${FEAT_FILE}"
    )

    log_info "执行: ${cmd[*]}"
    echo ""

    "${cmd[@]}"

    log_ok "训练完成"
}

# ============================================================
# all - 一键执行全流程
# ============================================================
cmd_all() {
    log_info "=============================================="
    log_info "  GIDS-IX 一键部署训练"
    log_info "  数据集: ${DATASET}"
    log_info "  GPU: cuda:${GPU_IDX}"
    log_info "  特征文件: ${FEAT_FILE}"
    log_info "  Epochs: ${EPOCHS}"
    log_info "=============================================="
    echo ""

    cmd_setup
    echo ""
    cmd_verify
    echo ""

    if [ ! -f "${FEAT_FILE}" ]; then
        cmd_prepare_data
        echo ""
    else
        log_info "特征文件已存在，跳过数据准备"
        echo ""
    fi

    cmd_train
}

# ============================================================
# clean - 清理
# ============================================================
cmd_clean() {
    log_info "清理构建产物..."
    rm -rf "${PROJECT_DIR}/gids_module_ix/build_ix"
    rm -f "${PROJECT_DIR}/gids_module_ix/ix_feature_store.cpp"
    rm -f "${PROJECT_DIR}/GIDS_Setup/GIDS/IXFeatureStore"*.so
    rm -f "${PROJECT_DIR}/GIDS_Setup/GIDS/libixstubs.so"
    log_ok "清理完成"
}

# ============================================================
# 帮助
# ============================================================
cmd_help() {
    echo "GIDS-IX 自动化脚本 (Iluvatar GPU / cuFile)"
    echo ""
    echo "用法:  bash run.sh <command> [args...]"
    echo ""
    echo "命令:"
    echo "  setup          构建 IXFeatureStore C++ 模块"
    echo "  verify         验证所有组件是否就绪"
    echo "  prepare-data   下载数据集并写入 SSD 特征文件"
    echo "  train          运行 GNN 训练"
    echo "  all            一键执行: setup -> verify -> prepare-data -> train"
    echo "  clean          清理构建产物"
    echo "  help           显示此帮助信息"
    echo ""
    echo "环境变量 (可覆盖默认值):"
    echo "  GPU_IDX         GPU 索引 (默认: 0)"
    echo "  EPOCHS          训练轮数 (默认: 10)"
    echo "  BATCH_SIZE      批次大小 (默认: 1024)"
    echo "  DATASET         数据集名称 (默认: ogbn-products)"
    echo "  FEAT_FILE       特征文件路径 (默认: /mnt/nvme0/node_feat.bin)"
    echo "  NUM_SSD         SSD 数量 (默认: 1)"
    echo "  CACHE_SIZE      缓存大小 (默认: 10)"
    echo ""
    echo "示例:"
    echo "  bash run.sh all"
    echo "  EPOCHS=3 DATASET=ogbn-papers100M bash run.sh train"
    echo "  GPU_IDX=1 FEAT_FILE=/mnt/nvme1/node_feat.bin bash run.sh all"
}

# ============================================================
# 主入口
# ============================================================
case "${1:-help}" in
    setup)        cmd_setup ;;
    verify)       cmd_verify ;;
    prepare-data) cmd_prepare_data ;;
    train)        cmd_train ;;
    all)          cmd_all ;;
    clean)        cmd_clean ;;
    help|--help|-h) cmd_help ;;
    *)
        log_error "未知命令: $1"
        cmd_help
        exit 1
        ;;
esac