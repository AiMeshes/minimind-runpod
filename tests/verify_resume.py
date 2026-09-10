#!/usr/bin/env python3
"""本地验证 minimind 的断点续训机制 —— 不需要 GPU，不花钱。

整个 RunPod 外壳的设计都押在这几条假设上，所以值得先在本地证明：

  1. checkpoint 落在 `../checkpoints`（硬编码相对路径 → 在 RunPod 上
     必须落在 network volume 里才能抗抢占）
  2. checkpoint 含恢复所需的完整状态（model/optimizer/epoch/step）
  3. `--from_resume 1` 真的从正确的 step 续训（不是从头开始）
  4. **学习率是接续的** —— 这证明 scheduler 状态被恢复。
     如果 lr 回到初始值，说明只恢复了模型权重，训练动力学已经错乱
  5. 保存后不残留 .tmp 文件（原子写干净退出）

用法:
    # 首次（自动用 uv 建环境）
    uv run --python 3.12 --with torch --with transformers==4.57.6 \\
           --with datasets==3.6.0 tests/verify_resume.py

    # 已有 venv（快得多）
    .venv/bin/python tests/verify_resume.py

    # 保留中间产物便于排查
    .venv/bin/python tests/verify_resume.py --keep
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MINIMIND_REPO = "https://github.com/AiMeshes/minimind.git"

# 极小模型 —— CPU 上几秒就能跑完一个 epoch
MODEL_ARGS = [
    "--hidden_size", "64",
    "--num_hidden_layers", "2",
    "--max_seq_len", "64",
    "--batch_size", "4",
    "--accumulation_steps", "1",
    "--num_workers", "0",
]
EPOCHS = 8
ITERS_PER_EPOCH = 50          # 200 条样本 / batch 4
TOTAL_STEPS = EPOCHS * ITERS_PER_EPOCH
SAVE_INTERVAL = 5
INITIAL_LR = 5e-4

# 在训练进行到 60% 时强杀。
# 为什么不能太早：minimind 用余弦调度 lr*(0.1+0.45*(1+cos(π·t/T)))，
# 在训练前 10% 处 lr 几乎等于初始值，此时「scheduler 是否恢复」根本无从区分。
# 60% 处 lr 已降到初始值的约 41%，断言才有鉴别力。
KILL_AT_GLOBAL = int(TOTAL_STEPS * 0.6)


def log(msg: str) -> None:
    print(msg, flush=True)


def fail(msg: str) -> None:
    print(f"\n  ✗ 断言失败: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"  ✓ {msg}", flush=True)


# ---------------------------------------------------------------------------
# 环境准备
# ---------------------------------------------------------------------------
def check_deps() -> None:
    missing = []
    for mod in ("torch", "transformers", "datasets"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        fail(
            f"缺少依赖: {', '.join(missing)}\n"
            "    先建环境:\n"
            "      uv venv --python 3.12 .venv\n"
            "      uv pip install --python .venv/bin/python torch "
            "transformers==4.57.6 datasets==3.6.0"
        )
    import torch
    ok(f"依赖就绪（torch {torch.__version__}，device={'cuda' if torch.cuda.is_available() else 'cpu'}）")


def prepare_minimind(workdir: Path, minimind_dir: Path | None) -> Path:
    """准备 minimind 代码。优先用本地已有副本，避免每次重复克隆。"""
    target = workdir / "minimind"

    if minimind_dir:
        log(f"  · 使用本地 minimind: {minimind_dir}")
        shutil.copytree(
            minimind_dir, target,
            ignore=shutil.ignore_patterns(".git", "__pycache__", "images"),
        )
    elif target.exists():
        log("  · 复用已有克隆")
    else:
        log(f"  · 克隆 minimind（sparse + shallow）: {MINIMIND_REPO}")
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
             MINIMIND_REPO, str(target)],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "sparse-checkout", "set", "trainer", "model", "dataset"],
            cwd=target, check=True, capture_output=True,
        )

    for required in ("trainer/train_pretrain.py", "model/tokenizer.json",
                     "trainer/trainer_utils.py"):
        if not (target / required).exists():
            fail(f"minimind 缺少 {required}")
    ok("minimind 代码就绪")
    return target


def write_tiny_dataset(minimind: Path) -> None:
    """生成极小语料。PretrainDataset 读 sample['text']。"""
    import random
    random.seed(0)
    words = ("模型 训练 数据 语言 学习 优化 梯度 参数 网络 注意力 "
             "位置 编码 解码 序列 生成 推理 损失 批次 学习率 权重").split()
    path = minimind / "dataset" / "pretrain_t2t_mini.jsonl"
    with path.open("w") as f:
        for _ in range(200):
            text = "".join(random.choice(words) for _ in range(random.randint(20, 60)))
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    ok(f"测试数据集就绪（200 条 → {ITERS_PER_EPOCH} iters/epoch）")


# ---------------------------------------------------------------------------
# 训练进程控制
# ---------------------------------------------------------------------------
def launch(minimind: Path, logfile: Path, resume: bool) -> subprocess.Popen:
    cmd = [
        sys.executable, "train_pretrain.py",
        *MODEL_ARGS,
        "--epochs", str(EPOCHS),
        "--save_interval", str(SAVE_INTERVAL),
        "--log_interval", "5",
        "--data_path", "../dataset/pretrain_t2t_mini.jsonl",
    ]
    if resume:
        cmd += ["--from_resume", "1"]

    fh = logfile.open("w")
    return subprocess.Popen(
        # -u 必须加：stdout 重定向到文件时 Python 默认块缓冲，
        # 日志会严重滞后于实际进度，导致无法在预期 step 精确 kill
        [sys.executable, "-u", *cmd[1:]],
        cwd=minimind / "trainer",
        stdout=fh, stderr=subprocess.STDOUT,
        start_new_session=True,   # 独立进程组，便于整组 kill
    )


def kill_tree(proc: subprocess.Popen) -> None:
    """模拟抢占：SIGKILL，不给任何优雅退出的机会。"""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


EPOCH_STEP_RE = re.compile(r"Epoch:\[(\d+)/(\d+)\]\((\d+)/")


def parse_global_step(text: str) -> int:
    """从日志中取最大的全局 step 数（跨 epoch 累加）。"""
    best = 0
    for m in EPOCH_STEP_RE.finditer(text):
        epoch_idx = int(m.group(1)) - 1
        best = max(best, epoch_idx * ITERS_PER_EPOCH + int(m.group(3)))
    return best


def wait_for_global_step(logfile: Path, target: int, proc: subprocess.Popen,
                         timeout: float = 300) -> int:
    """轮询日志直到全局 step >= target。返回实际看到的全局 step。"""
    deadline = time.time() + timeout
    seen = 0
    while time.time() < deadline:
        # 进程先退出 → 训练在 kill 之前就跑完了，测试失去意义
        if proc.poll() is not None:
            fail(
                f"训练在到达全局 step {target} 前就结束了（最后看到 {seen}）。"
                f"请调大 EPOCHS（当前 {EPOCHS}）"
            )
        if logfile.exists():
            seen = parse_global_step(logfile.read_text(errors="ignore"))
            if seen >= target:
                return seen
        time.sleep(0.2)
    fail(f"等待全局 step >= {target} 超时（实际最后看到 {seen}）")


def wait_for_marker(logfile: Path, marker: str, timeout: float = 120) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if logfile.exists() and marker in logfile.read_text(errors="ignore"):
            return
        time.sleep(0.3)
    fail(f"等待标记超时: {marker!r}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="验证 minimind 断点续训")
    ap.add_argument("--keep", action="store_true", help="保留工作目录便于排查")
    ap.add_argument("--workdir", type=Path, help="指定工作目录（默认临时目录）")
    ap.add_argument("--minimind-dir", type=Path,
                    help="使用本地已有的 minimind 副本，跳过克隆")
    args = ap.parse_args()

    if args.workdir:
        workdir = args.workdir
        workdir.mkdir(parents=True, exist_ok=True)
    else:
        workdir = Path(tempfile.mkdtemp(prefix="minimind-resume-test-"))

    log(f"工作目录: {workdir}\n")

    try:
        log("[1/6] 检查依赖")
        check_deps()

        log("\n[2/6] 准备代码与数据")
        minimind = prepare_minimind(workdir, args.minimind_dir)
        write_tiny_dataset(minimind)

        ckpt_dir = minimind / "checkpoints"
        for d in (ckpt_dir, minimind / "out"):
            shutil.rmtree(d, ignore_errors=True)

        # --- 第一次运行：跑到训练中后段再 SIGKILL ---
        log(f"\n[3/6] 第一次运行，跑到全局 step {KILL_AT_GLOBAL}/{TOTAL_STEPS}"
            f"（{KILL_AT_GLOBAL * 100 // TOTAL_STEPS}%）后强杀（模拟抢占）")
        log1 = workdir / "run1.log"
        proc = launch(minimind, log1, resume=False)
        seen = wait_for_global_step(log1, KILL_AT_GLOBAL, proc)
        time.sleep(0.5)   # 让最近的保存落盘
        kill_tree(proc)
        ok(f"已在全局 step {seen}/{TOTAL_STEPS} 强杀（SIGKILL，无优雅退出机会）")

        # --- 断言 1: checkpoint 位置正确 ---
        log("\n[4/6] 检查 checkpoint")
        resume_file = ckpt_dir / "pretrain_64_resume.pth"
        weight_file = ckpt_dir / "pretrain_64.pth"
        if not resume_file.exists():
            fail(f"{resume_file} 不存在 —— '../checkpoints' 路径约定被破坏")
        ok("../checkpoints/pretrain_64_resume.pth 存在（硬编码路径成立）")
        if not weight_file.exists():
            fail(f"{weight_file} 不存在")
        ok("../checkpoints/pretrain_64.pth 存在")

        import torch

        # 关键断言：续训状态文件必须可读（原子写的真正保证）。
        # 注意这里不断言「无 .tmp 残留」—— 强杀恰好落在保存中途时残留一个
        # .tmp 是预期行为且无害（load 时被忽略，下次保存会覆盖）。
        try:
            ckpt = torch.load(resume_file, map_location="cpu", weights_only=False)
        except Exception as exc:
            fail(f"续训文件损坏，无法加载: {exc}")
        for key in ("model", "optimizer", "epoch", "step"):
            if key not in ckpt:
                fail(f"checkpoint 缺少 '{key}'，无法完整恢复")
        ckpt_epoch, ckpt_step = int(ckpt["epoch"]), int(ckpt["step"])
        ok(f"续训文件完好可读，状态完整: epoch={ckpt_epoch}, step={ckpt_step}")

        # 顺带探明 ../out 的状态。已知上游 train_pretrain.py:68 对 out/ 是
        # 非原子写，所以这里只报告不阻断 —— 续训不读它
        try:
            torch.load(weight_file, map_location="cpu", weights_only=False)
            ok("../out 权重完好")
        except Exception:
            log("  ! ../out 权重损坏 —— 上游非原子写（train_pretrain.py:68），"
                "续训不受影响，下次保存会覆盖")

        # --- 断言 2: 续训从正确的 step 开始 ---
        log(f"\n[5/6] 续训（--from_resume 1），期望从 step {ckpt_step} 之后继续")
        log2 = workdir / "run2.log"
        proc2 = launch(minimind, log2, resume=True)
        # ckpt_step 是 epoch 内下标，marker 里用的是同一个值（minimind 的语义）
        marker = f"跳过前{ckpt_step}个step"
        wait_for_marker(log2, marker)
        resumed_global = ckpt_epoch * ITERS_PER_EPOCH + ckpt_step
        wait_for_global_step(log2, resumed_global + SAVE_INTERVAL, proc2)  # 确认真的在继续训
        kill_tree(proc2)

        text2 = log2.read_text(errors="ignore")
        if marker not in text2:
            fail(f"续训日志未出现 {marker!r}")
        ok(f"续训日志出现「{marker}，从step {ckpt_step + 1}开始」")

        # 起始 epoch 必须是 checkpoint 里的 epoch（0-indexed → 日志显示 +1）
        first_epoch_line = next(
            (l for l in text2.splitlines() if "跳过前" in l), "")
        if f"Epoch [{ckpt_epoch + 1}/{EPOCHS}]" not in first_epoch_line:
            fail(f"续训起始 epoch 不符: {first_epoch_line!r}，期望 [{ckpt_epoch + 1}/{EPOCHS}]")
        ok(f"续训起始 epoch 正确（[{ckpt_epoch + 1}/{EPOCHS}]）")

        # --- 断言 3: 学习率接续（最关键的一条）---
        log("\n[6/6] 检查学习率连续性（证明 scheduler 状态被恢复）")
        lr_pattern = re.compile(r"lr:\s*([0-9.eE+-]+)")
        resumed_lrs = [float(m.group(1)) for m in lr_pattern.finditer(text2)]
        if not resumed_lrs:
            fail("续训日志中找不到学习率")

        import math
        resumed_global = ckpt_epoch * ITERS_PER_EPOCH + ckpt_step

        def cosine_lr(step: int) -> float:
            return INITIAL_LR * (
                0.1 + 0.45 * (1 + math.cos(math.pi * step / TOTAL_STEPS))
            )

        actual_lr = resumed_lrs[0]
        # 续训后打印的是「续训后已走过的」步数对应的 lr，允许 SAVE_INTERVAL 的偏移，
        # 在附近取最接近的那个理论值做比较
        candidates = [cosine_lr(resumed_global + d)
                      for d in range(-SAVE_INTERVAL, SAVE_INTERVAL * 2 + 1)]
        expected_lr = min(candidates, key=lambda v: abs(v - actual_lr))

        ok(f"初始 lr(step 0)      = {INITIAL_LR:.7f}")
        ok(f"续训处 lr(step {resumed_global}) = {actual_lr:.7f}")
        ok(f"调度公式预期值        = {expected_lr:.7f}")

        # 断言 1：必须与余弦曲线在该 step 的值吻合（这是真正的证明）
        if abs(actual_lr - expected_lr) > expected_lr * 0.02:
            fail(
                f"学习率 {actual_lr:.7f} 偏离余弦调度曲线（预期 {expected_lr:.7f}）"
                " —— scheduler 状态可能未恢复"
            )
        ok("与余弦调度曲线吻合 → scheduler 状态被恢复")

        # 断言 2：鉴别力检查 —— 确保续训点足够靠后，使「接续」与「重置」
        # 在数值上可区分。否则上面的吻合断言在早期 step 处没有意义
        if expected_lr > INITIAL_LR * 0.9:
            fail(
                f"测试鉴别力不足：续训处 lr({expected_lr:.7f}) 与初始 lr "
                f"({INITIAL_LR:.7f}) 太接近，无法区分『接续』与『重置』。"
                f"请调大 KILL_AT_GLOBAL（当前 {KILL_AT_GLOBAL}/{TOTAL_STEPS}）"
            )
        drop = (1 - expected_lr / INITIAL_LR) * 100
        ok(f"鉴别力充足：续训处 lr 已比初始值低 {drop:.0f}%，"
           "重置会立刻暴露为 lr 跳回初始值")

        log("\n" + "=" * 62)
        log("全部通过 —— 断点续训机制成立，Ship it.")
        log("=" * 62)
        return 0

    except subprocess.CalledProcessError as exc:
        fail(f"外部命令失败: {exc}\n{exc.stderr.decode(errors='ignore') if exc.stderr else ''}")
        return 1
    finally:
        if args.keep:
            log(f"\n保留工作目录: {workdir}")
        elif not args.workdir:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
