#!/bin/bash
# 本地验证全套 —— 不需要 GPU，不花一分钱。
#
# 为什么值得跑：整个抗抢占设计都建立在两个假设上，而这两个假设
# 都可以在本地用 CPU 证伪（如果它们错了，上云跑几天才发现就太贵了）：
#   1. minimind 的 --from_resume 1 真的能从 checkpoint 接续（含 scheduler 状态）
#   2. 外壳的 supervisor 能正确区分「被抢占」和「配置错误」
#
# 用法:
#   bash tests/run_all.sh
#   bash tests/run_all.sh --workdir /tmp/mytest --keep
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

PY="${PY:-$ROOT/.venv/bin/python}"
if [ ! -x "$PY" ]; then
    echo "找不到 $PY"
    echo
    echo "先建环境："
    echo "  uv venv --python 3.12 .venv"
    echo "  uv pip install --python .venv/bin/python torch transformers==4.57.6 datasets==3.6.0"
    exit 1
fi

pass=0
fail=0

run() {
    local name="$1"; shift
    echo
    echo "════════════════════════════════════════════════════════════"
    echo "  验证: $name"
    echo "════════════════════════════════════════════════════════════"
    if "$PY" "$@"; then
        pass=$((pass + 1))
    else
        fail=$((fail + 1))
        echo "  ↑ 该项失败"
    fi
}

run "断点续训机制（--from_resume 1 是否真的接续）" \
    tests/verify_resume.py "$@"
run "supervisor 行为（抢占 vs 配置错误的区分）" \
    tests/verify_supervisor.py
run "watch.py 恢复逻辑（该恢复时恢复、不该动时不动）" \
    tests/verify_watch.py

echo
echo "════════════════════════════════════════════════════════════"
if [ "$fail" -eq 0 ]; then
    echo "  全部通过（$pass/$((pass + fail))）"
else
    echo "  $fail 项失败（$pass 通过 / $((pass + fail)) 总计）"
fi
echo "════════════════════════════════════════════════════════════"
exit "$fail"
