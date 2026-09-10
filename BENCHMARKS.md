# GPU 训练速度实测

**为什么实测而不是查规格表**：官方 TFLOPS 对 64M 这种小模型没有参考价值。
模型太小，瓶颈会转移到数据加载和 kernel 启动开销上 ——
H100 在 7B 模型上比 4090 快 3 倍，在 64M 上可能只有 1.5 倍。
只有同一套代码、同一份数据、同一个 batch 下测出的 steps/sec 才能比较。

## 测试条件

| 项 | 值 |
|---|---|
| 代码 | [minimind](https://github.com/jingyaogong/minimind) `74f3eca` |
| 模型 | MiniMind 64M（`hidden_size=768`, `num_hidden_layers=8`） |
| 数据 | `pretrain_t2t_mini.jsonl`（1.24GB） |
| 超参 | `--epochs 2 --batch_size 32 --learning_rate 5e-4` |
| 采样 | 训练稳定后取 60 秒窗口，读日志中 step 推进量 |
| 脚本 | `scripts/benchmark_gpu.py` |

> 单卡测试。多卡受 DDP 扩展效率影响，需要单独测。

## 结果

| GPU | 单价 $/hr | steps/s | 跑完 2 epochs | 总成本 | 相对速度 |
|---|---|---|---|---|---|
| *(待填)* | | | | | |

**跑完 2 epochs** 指 `iters × 2` 步（DDP 下 iters 为每卡分片后的值）。

## 复现方法

```bash
# 单个 GPU
python scripts/benchmark_gpu.py --gpu "NVIDIA GeForce RTX 4090" --count 1

# 多卡
python scripts/benchmark_gpu.py --gpu "NVIDIA H100 NVL" --count 2
```

脚本会：创建 Pod → 等训练稳定 → 采样 60 秒 → 算 steps/sec → **终止 Pod**。
无论成败都会终止，避免漏计费。

---

## 已知的价格事实（2026-09 实测）

**竞价和按需价格完全相同。** 从 RunPod API 查到的 `minimumBidPrice` 与
`uninterruptablePrice` 对每一款 GPU 都相等：

| GPU | 竞价 | 按需 |
|---|---|---|
| RTX 4090 | $0.34 | $0.34 |
| RTX 3090 | $0.22 | $0.22 |
| A100 SXM | $1.39 | $1.39 |
| H100 SXM | $2.69 | $2.69 |

> **这意味着用 Spot 是纯负收益** —— 同样的价格承担被抢占的风险。
> 实测验证：一轮 Spot 跑到 14 分钟被抢占，Pod 直接从 API 消失（404），
> 容器盘连同全部进度丢失。

**供应量信号**（`maxGpuCountCommunityCloud`，可同时开出的最大卡数）：

| GPU | 可开卡数 |
|---|---|
| L40 / A5000 / **H100 NVL** | 10 |
| 4090 / 5090 / A100 SXM / 6000 Ada / H200 SXM | 8 |
| H100 SXM / H100 PCIe | 1 |
| B300 / RTX PRO 4000 / MIG 系列 | 0（当前无货） |
