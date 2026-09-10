#!/bin/bash
# 构建镜像前的依赖检查 —— 省下一次 10GB 构建才发现版本冲突的时间。
#
# 检查两件事：
#   1. Dockerfile 里的包在 linux/amd64 + py3.12 上可解析（有对应 wheel）
#   2. 在 torch==2.6.0（基础镜像预装版本）约束下仍可解析
#
# 第 2 条尤其重要：若 trl / transformers 要求更高的 torch，pip 会在构建时
# 升级 torch，破坏基础镜像的 CUDA 12.8 配套。依赖解析能提前暴露这个问题。
#
# 用法:
#   bash tests/check_docker_deps.sh
set -uo pipefail

cd "$(dirname "$0")/.."
DOCKERFILE="${1:-Dockerfile}"

if ! command -v uv >/dev/null 2>&1; then
    echo "需要 uv 才能运行（brew install uv 或 curl -LsSf https://astral.sh/uv/install.sh | sh）"
    exit 1
fi

if [ ! -f "$DOCKERFILE" ]; then
    echo "找不到 $DOCKERFILE"
    exit 1
fi

# 先把反斜杠续行合并成单行 —— 否则多行 RUN 里只有第一行带 "pip install"，
# 后续行的包全部漏掉。
JOINED=$(perl -0777 -pe 's/\\\r?\n\s*/ /g' "$DOCKERFILE")

# 再从 pip install 行里抽包名。包名必须以字母开头，
# 否则会误抓 shell 条件里的 "$INSTALL_SERVING" = "1" 这类字面量。
PKGS=$(printf '%s\n' "$JOINED" \
       | grep -E 'pip install' \
       | grep -oE '"[a-zA-Z][a-zA-Z0-9_.-]*(==[0-9.]+)?"' \
       | tr -d '"' \
       | grep -vE '^(torch|torchvision)$' \
       | sort -u)

if [ -z "$PKGS" ]; then
    echo "未能从 $DOCKERFILE 提取到依赖包"
    exit 1
fi

echo "从 $DOCKERFILE 提取到 $(echo "$PKGS" | wc -l | tr -d ' ') 个包："
echo "$PKGS" | sed 's/^/  /'
echo

TMP=$(mktemp)
trap 'rm -f "$TMP" /tmp/_resolved_deps.txt' EXIT
echo "$PKGS" > "$TMP"

PINNED_TORCH=$(grep -oE 'torch260|torch[0-9]{3,}' "$DOCKERFILE" | head -1 || true)
CONSTRAINT=""
if [ -n "$PINNED_TORCH" ]; then
    # 镜像标签形如 ...-torch260-...，还原成 2.6.0
    ver=$(echo "$PINNED_TORCH" | grep -oE '[0-9]{3,}')
    CONSTRAINT="$(
        printf '%s' "$ver" | sed 's/^\(.\)\(.\)\(.\)$/\1.\2.\3/'
    )"
fi

echo "── 检查 1: linux/amd64 + py3.12 可解析 ──"
if uv pip compile --python-version 3.12 --python-platform x86_64-unknown-linux-gnu \
        --quiet --no-header "$TMP" -o /tmp/_resolved_deps.txt 2>&1 | tail -20; then
    echo "  ✓ 解析成功（$(grep -c '==' /tmp/_resolved_deps.txt) 个包）"
else
    echo "  ✗ 解析失败 —— 该依赖组合在 linux/amd64 上装不上，先修 Dockerfile"
    exit 1
fi

if [ -z "$CONSTRAINT" ]; then
    echo
    echo "（未从镜像标签解析出 torch 版本，跳过检查 2）"
    exit 0
fi

echo
echo "── 检查 2: 在 torch==$CONSTRAINT 约束下可解析 ──"
if uv pip compile --python-version 3.12 --python-platform x86_64-unknown-linux-gnu \
        --quiet --no-header "$TMP" \
        --constraint <(echo "torch==$CONSTRAINT") -o /tmp/_resolved_deps.txt 2>&1 | tail -20; then
    echo "  ✓ 解析成功 —— pip 构建时不会升级 torch，基础镜像的 CUDA 配套不受影响"
else
    echo "  ✗ 与 torch==$CONSTRAINT 冲突 ——"
    echo "    pip 会在构建时升级 torch，破坏基础镜像的 CUDA 配套。"
    echo "    需要降低相关包版本，或换用 torch 更匹配的基础镜像。"
    exit 1
fi

echo
echo "依赖检查通过"
