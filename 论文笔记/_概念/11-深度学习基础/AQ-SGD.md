---
type: concept
aliases: [AQ-SGD, Activation-Quantized SGD]
---

# AQ-SGD

## 定义

AQ-SGD 是一种用于 pipeline 并行训练的 activation 量化方案，通过误差补偿（error compensation）机制缓解量化误差对收敛的影响，是 TAH-Quant 的主要对比基线。

## 核心要点

1. **误差补偿**: 缓存历史量化误差并在下一次量化时补偿，减少累积偏差
2. **存储开销**: 需在每台机器上缓存一份 activation，大数据集下显存压力显著
3. **多 epoch 依赖**: 误差补偿在多 epoch 训练中更有效；单 epoch 预训练时效果有限
4. **收敛性**: 有理论收敛保证，但比 TAH-Quant 实验吞吐量低约 22%

## 代表工作

- [[TAH-Quant]]: 无需误差补偿缓存，在多场景下超越 AQ-SGD 的吞吐量和精度

## 相关概念

- [[Activation Quantization|激活量化]]
- [[Pipeline Parallelism|流水线并行]]
- [[Stochastic Gradient Descent|SGD]]
