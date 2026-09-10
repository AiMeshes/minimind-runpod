#!/bin/bash
# 在 Pod 内部运行，核验训练链路是否完整走通。
#
# 用途：RunPod 的 REST API 没有日志端点，出问题只能 SSH 进来查。
# 这个脚本把需要检查的五项一次跑完，输出结论而不是让人对着原始输出猜。
#
# 用法（从本地）:
#   scp -P <port> -i ~/.ssh/id_rsa scripts/onpod_check.sh root@<ip>:/tmp/
#   ssh -p <port> -i ~/.ssh/id_rsa root@<ip> "bash /tmp/onpod_check.sh"
set -uo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
MINIMIND="$WORKSPACE/minimind"
PASS=0
FAIL=0

ok()   { echo "  ✓ $*"; PASS=$((PASS+1)); }
bad()  { echo "  ✗ $*"; FAIL=$((FAIL+1)); }
info() { echo "  · $*"; }

echo "════════════════════════════════════════════════════════════"
echo "  training chain check @ $(hostname)  $(date -u +%H:%M:%SZ)"
echo "════════════════════════════════════════════════════════════"

echo
echo "[1/6] GPU 可见性"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
               --format=csv,noheader | sed 's/^/  · /'
    ok "nvidia-smi 可用"
else
    bad "nvidia-smi 不存在 —— 容器可能没拿到 GPU"
fi

echo
echo "[2/6] 依赖（官方镜像需补装）"
if python -c "import torch, transformers, datasets" 2>/dev/null; then
    python -c "
import torch, transformers, datasets
print(f'  · torch {torch.__version__}  cuda={torch.cuda.is_available()}  '
      f'device_count={torch.cuda.device_count()}')
print(f'  · transformers {transformers.__version__}')
print(f'  · datasets {datasets.__version__}')
"
    ok "依赖可 import"
    python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
        && ok "torch.cuda.is_available() == True" \
        || bad "torch 装了但看不到 CUDA"
else
    bad "依赖缺失 —— prepare_deps 没跑完或失败"
fi

echo
echo "[3/6] 代码已克隆"
if [ -f "$MINIMIND/trainer/train_pretrain.py" ]; then
    info "$(cd "$MINIMIND" && git log -1 --format='%h %s' 2>/dev/null || echo '无 git 信息')"
    ok "minimind 代码就位"
else
    bad "$MINIMIND/trainer/train_pretrain.py 不存在"
fi

echo
echo "[4/6] 数据已下载"
DS="$MINIMIND/dataset/pretrain_t2t_mini.jsonl"
if [ -f "$DS" ]; then
    sz=$(du -h "$DS" | cut -f1)
    lines=$(wc -l < "$DS" 2>/dev/null || echo '?')
    info "$DS  ($sz, $lines 行)"
    # 1.24GB 的完整文件才算下完；明显偏小说明中断了
    if [ "$(stat -f%z "$DS" 2>/dev/null || stat -c%s "$DS")" -gt 1000000000 ]; then
        ok "数据完整（>1GB）"
    else
        bad "数据文件偏小，可能下载中断"
    fi
else
    bad "$DS 不存在"
fi

echo
echo "[5/6] 训练进程 / GPU 占用"
if ps aux | grep -E "train_pretrain\.py" | grep -v grep >/dev/null; then
    ps aux | grep -E "train_pretrain\.py" | grep -v grep | awk '{print "  · pid="$2" cpu="$3"% mem="$4"% "$11" "$12" "$13}' | head -3
    ok "训练进程在运行"
    UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    if [ -n "$UTIL" ] && [ "$UTIL" -gt 5 ] 2>/dev/null; then
        ok "GPU 利用率 ${UTIL}% —— 真的在算"
    else
        info "GPU 利用率 ${UTIL:-?}%（数据加载阶段可能偏低）"
    fi
else
    if [ -f "$WORKSPACE/train.log" ]; then
        bad "训练进程不在运行，日志尾部："
        tail -15 "$WORKSPACE/train.log" | sed 's/^/      /'
    else
        bad "训练进程不在运行，且无 $WORKSPACE/train.log"
    fi
fi

echo
echo "[6/6] checkpoint 落盘"
CKPT="$MINIMIND/checkpoints"
if [ -d "$CKPT" ] && [ -n "$(ls -A "$CKPT" 2>/dev/null)" ]; then
    ls -la "$CKPT" | tail -n +2 | awk '{print "  · "$5" bytes  "$9}'
    if ls "$CKPT"/*_resume.pth >/dev/null 2>&1; then
        ok "续训文件存在（抗抢占的关键）"
        # 立刻验证可读性 —— 损坏的 checkpoint 比没有更危险
        if python -c "
import torch, glob, sys
f = sorted(glob.glob('$CKPT/*_resume.pth'))[-1]
d = torch.load(f, map_location='cpu', weights_only=False)
missing = [k for k in ('model','optimizer','epoch','step') if k not in d]
sys.exit(1 if missing else 0)
" 2>/dev/null; then
            ok "续训文件可读且状态完整"
        else
            bad "续训文件损坏或缺少关键字段"
        fi
    else
        bad "无 *_resume.pth —— 被抢占后无法续训"
    fi
else
    info "checkpoint 目录为空 —— save_interval 还没到（正常，若刚启动）"
fi

echo
echo "[附] 训练日志尾部"
if [ -f "$WORKSPACE/train.log" ]; then
    tail -8 "$WORKSPACE/train.log" | sed 's/^/  │ /'
else
    info "无 $WORKSPACE/train.log（旧版 entrypoint 或训练未启动）"
fi

echo
echo "════════════════════════════════════════════════════════════"
echo "  通过 $PASS 项 / 失败 $FAIL 项"
echo "════════════════════════════════════════════════════════════"
exit "$FAIL"
