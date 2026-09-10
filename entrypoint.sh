#!/bin/bash
# minimind RunPod 外壳：数据准备 + 抗抢占 supervisor
#
# 设计要点：
#   - 不从代码层面改 minimind，靠目录约定让 ../checkpoints 落在 network volume 上
#   - 训练进程崩溃或被抢占时，外层循环自动用 --from_resume 1 从最新 ckp 接上
#   - 数据准备幂等 + 原子落盘，抢占后重跑不会留下半份数据
set -uo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
MINIMIND_DIR="${MINIMIND_DIR:-$WORKSPACE/minimind}"
MINIMIND_REPO="${MINIMIND_REPO:-https://github.com/AiMeshes/minimind.git}"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')}"
NUM_GPUS="${NUM_GPUS:-1}"
SUPERVISOR_STATE="$WORKSPACE/.supervisor"
RESTART_DELAY="${RESTART_DELAY:-15}"
HEARTBEAT_INTERVAL="${HEARTBEAT_INTERVAL:-30}"

# 快速失败熔断：连续 N 次「运行不足 M 秒就退出」则停止重试并退出。
# 用来区分「被抢占」（跑了几小时才挂）和「配置写错了」（秒挂）——
# 后者无限重试只是在烧钱。
FAST_FAIL_SECONDS="${FAST_FAIL_SECONDS:-60}"
FAST_FAIL_LIMIT="${FAST_FAIL_LIMIT:-5}"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

mkdir -p "$WORKSPACE" "$SUPERVISOR_STATE"

# ---------------------------------------------------------------------------
# 1. 拉取 minimind（幂等）
# ---------------------------------------------------------------------------
prepare_code() {
    if [ -d "$MINIMIND_DIR/.git" ]; then
        log "minimind 已存在，尝试更新..."
        git -C "$MINIMIND_DIR" fetch --depth 1 origin master 2>/dev/null \
            && git -C "$MINIMIND_DIR" reset --hard origin/master 2>/dev/null \
            && log "已更新到最新 master" \
            || log "更新失败，沿用本地版本（离线可继续）"
    else
        log "克隆 minimind: $MINIMIND_REPO"
        local tmp="${MINIMIND_DIR}.tmp.$$"
        rm -rf "$tmp"
        git clone --depth 1 "$MINIMIND_REPO" "$tmp" || { log "克隆失败"; return 1; }
        rm -rf "$MINIMIND_DIR"
        mv "$tmp" "$MINIMIND_DIR"          # 原子替换
        log "克隆完成"
    fi
}

# ---------------------------------------------------------------------------
# 2. 准备数据（幂等 + 原子）
# ---------------------------------------------------------------------------
prepare_data() {
    local ds_dir="$MINIMIND_DIR/dataset"
    mkdir -p "$ds_dir"

    # 无 DATA_URL 则跳过（数据可能已预置在 network volume 上）
    if [ -z "${DATA_URL:-}" ]; then
        log "DATA_URL 未设置，跳过下载"
        return 0
    fi

    local fname
    fname="$(basename "${DATA_URL%%\?*}")"
    local target="$ds_dir/$fname"

    if [ -f "${target}.ready" ]; then
        log "数据已就绪: $fname"
        return 0
    fi

    log "下载数据: $DATA_URL"
    local tmp="${target}.tmp.$$"
    rm -f "$tmp"

    # 支持 http(s) 与 oss/hf 直链；断点续传
    if ! curl -fL --retry 5 --retry-delay 10 -C - -o "$tmp" "$DATA_URL"; then
        log "下载失败，保留 tmp 以便下次续传"
        return 1
    fi

    mv "$tmp" "$target"                     # 原子
    touch "${target}.ready"
    log "数据就绪: $target ($(du -h "$target" | cut -f1))"
}

# ---------------------------------------------------------------------------
# 2.5 依赖准备（用官方镜像时必须；自建镜像里已装好则秒过）
#
#     官方 runpod/pytorch 镜像只有 torch，没有 transformers/datasets。
#     这里按需补装 —— 已装好时只做一次 import 检查，几乎不耗时。
#
#     取舍：自建镜像启动快（预装好），官方镜像零构建成本。
#     官方镜像每次启动多花几分钟 ≈ $0.0x，先用它跑通流程更划算。
# ---------------------------------------------------------------------------
prepare_deps() {
    if [ "${INSTALL_DEPS:-1}" != "1" ]; then
        log "INSTALL_DEPS!=1，跳过依赖检查"
        return 0
    fi

    local pkgs="${PIP_PACKAGES:-transformers==4.57.6 datasets==3.6.0}"
    local probe="${PIP_PROBE:-transformers datasets}"

    # 快路径：能 import 就说明装好了，直接跳过（自建镜像走这条）
    if python -c "
import importlib, sys
for m in '${probe}'.split():
    importlib.import_module(m)
" 2>/dev/null; then
        log "依赖已就绪，跳过安装"
        return 0
    fi

    log "补装依赖: ${pkgs}"
    log "（官方镜像不含这些包，首次启动需要几分钟。自建镜像可省去此步）"

    # shellcheck disable=SC2086  # 有意按空格拆分
    if ! python -m pip install --no-cache-dir --quiet ${pkgs}; then
        log "依赖安装失败 —— 检查网络或包名"
        return 1
    fi

    if ! python -c "
import importlib
for m in '${probe}'.split():
    importlib.import_module(m)
" 2>/dev/null; then
        log "安装后仍无法 import ${probe}，请检查包名"
        return 1
    fi

    log "依赖安装完成"
}

# ---------------------------------------------------------------------------
# 3. 心跳：供外部监控判断存活（watch.py 用它决定是否重建 Pod）
# ---------------------------------------------------------------------------
start_heartbeat() {
    # 刻意把子进程的 stdio 重定向到 /dev/null：
    # 若它继承 supervisor 的 stdout，被 kill 时正卡在 sleep 里的孤儿进程
    # 会继续持有该管道，导致调用方（docker logs / 父进程）迟迟等不到 EOF。
    (
        while true; do
            date -u +%s > "$SUPERVISOR_STATE/heartbeat"
            sleep "$HEARTBEAT_INTERVAL"
        done
    ) >/dev/null 2>&1 &
    HEARTBEAT_PID=$!
    echo "$HEARTBEAT_PID" > "$SUPERVISOR_STATE/heartbeat.pid"
}

# ---------------------------------------------------------------------------
# 4. 可选：启动 sshd（覆盖基座 ENTRYPOINT 会丢掉 RunPod 的 /start.sh）
#
#    长训期间能 SSH 进去看日志/翻 checkpoint 很有用。RunPod 通过 PUBLIC_KEY
#    环境变量注入公钥，这里复刻 /start.sh 的行为。
# ---------------------------------------------------------------------------
start_sshd_if_requested() {
    [ "${START_SSHD:-0}" = "1" ] || return 0

    mkdir -p /root/.ssh
    if [ -n "${PUBLIC_KEY:-}" ]; then
        echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
        chmod 700 /root/.ssh
        chmod 600 /root/.ssh/authorized_keys
        log "已写入 SSH 公钥"
    else
        log "START_SSHD=1 但未提供 PUBLIC_KEY，SSH 可能无法登录"
    fi

    mkdir -p /var/run/sshd
    /usr/sbin/sshd 2>/dev/null && log "sshd 已启动（端口 22）" || log "sshd 启动失败"
}

# ---------------------------------------------------------------------------
# 5. Supervisor 主循环
# ---------------------------------------------------------------------------
main() {
    log "=== minimind RunPod supervisor 启动 ==="
    log "GPU 数: $NUM_GPUS | 工作目录: $WORKSPACE"

    start_sshd_if_requested
    prepare_deps || { log "依赖准备失败，退出"; exit 1; }
    prepare_code || { log "代码准备失败，退出"; exit 1; }
    prepare_data || log "数据准备失败，仍尝试启动训练"

    cd "$MINIMIND_DIR/trainer" || { log "进入 trainer 目录失败"; exit 1; }

    # 训练超参：优先用命令行参数，否则读 TRAIN_ARGS 环境变量（按空格拆分）
    local -a train_args=()
    if [ "$#" -gt 0 ]; then
        train_args=("$@")
    elif [ -n "${TRAIN_ARGS:-}" ]; then
        # shellcheck disable=SC2206  # 有意按空格拆分：TRAIN_ARGS 是我们自己的配置
        train_args=(${TRAIN_ARGS})
    fi
    if [ ${#train_args[@]} -gt 0 ]; then
        log "训练参数: ${train_args[*]}"
    else
        log "训练参数: （使用 minimind 默认值）"
    fi

    start_heartbeat
    trap 'log "收到终止信号，清理心跳"; kill "$HEARTBEAT_PID" 2>/dev/null; exit 0' TERM INT

    local attempt=0
    local consecutive_fast_fails=0

    while true; do
        attempt=$((attempt + 1))
        log "--- 第 $attempt 次尝试（--from_resume 1，自动从最新 checkpoint 续训）---"

        local start_ts
        start_ts=$(date +%s)

        # --from_resume 1 是 minimind 自带能力：自动找 ../checkpoints 下的 *_resume.pth
        # GPU 数变化时它还会自动换算 step（trainer_utils.py:110-114）
        # 注意：set -u 下展开空数组在 bash 3.2 会报 unbound variable，
        # 所以这里用条件分支而不是直接展开（兼容性，且行为更显式）
        if [ ${#train_args[@]} -gt 0 ]; then
            torchrun \
                --nproc_per_node="$NUM_GPUS" \
                --master_port="${MASTER_PORT:-29500}" \
                train_pretrain.py \
                --from_resume 1 \
                "${train_args[@]}"
        else
            torchrun \
                --nproc_per_node="$NUM_GPUS" \
                --master_port="${MASTER_PORT:-29500}" \
                train_pretrain.py \
                --from_resume 1
        fi
        local code=$?
        local elapsed=$(( $(date +%s) - start_ts ))

        if [ $code -eq 0 ]; then
            log "训练正常结束（耗时 ${elapsed}s）"
            break
        fi

        # 注意 ${code} 的花括号是必需的：紧跟全角括号时，bash 3.2（macOS 自带）
        # 不做 UTF-8 感知，会把括号的字节当成变量名的一部分
        log "训练异常退出 code=${code}（耗时 ${elapsed}s）"

        # 快速连续失败说明是配置/环境问题，不是抢占 —— 退避避免烧钱
        if [ "$elapsed" -lt "$FAST_FAIL_SECONDS" ]; then
            consecutive_fast_fails=$((consecutive_fast_fails + 1))
        else
            consecutive_fast_fails=0
        fi

        if [ "$consecutive_fast_fails" -ge "$FAST_FAIL_LIMIT" ]; then
            log "连续 $consecutive_fast_fails 次快速失败（每次 < ${FAST_FAIL_SECONDS}s），"
            log "疑似配置错误而非抢占。保留现场并退出，避免继续烧钱。请检查日志。"
            exit 1
        fi

        log "${RESTART_DELAY}s 后重启..."
        sleep "$RESTART_DELAY"
    done

    kill "$HEARTBEAT_PID" 2>/dev/null || true
    log "=== supervisor 退出 ==="
}

main "$@"
