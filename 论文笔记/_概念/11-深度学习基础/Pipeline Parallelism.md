---
type: concept
aliases: [流水线并行, Pipeline Parallel, PP]
---

# Pipeline Parallelism

## 定义

流水线并行是一种模型并行策略，将大型神经网络按层切分为多个连续 stage，分别部署在不同设备（机器）上，各 stage 之间通过传输 activation（前向）和 gradient（后向）协同训练。

## 数学形式

设模型 $f = f^{(K)} \circ \cdots \circ f^{(1)}$，分配到 $K$ 台机器：

$$
a^{(k)} = f^{(k)}(a^{(k-1)}), \quad k = 1, \ldots, K
$$

机器 $k$ 仅需存储其 stage 参数 $\theta^{(k)}$，跨机通信量为 $|a^{(k)}|$（activation 大小）。

## 核心要点

1. **通信内容**: 前向传 activation，后向传 activation gradient
2. **通信成为瓶颈**: 带宽受限时（跨地域训练），activation 通信耗时可超过计算时间
3. **micro-batching**: 将 batch 切成 micro-batch 流水线执行，提升 GPU 利用率（减少 bubble）
4. **2-stage 到多 stage**: 典型为 2 台机器做前后两段，多阶段进一步切分

## 代表工作

- [[TAH-Quant]]: 针对慢网络 pipeline 并行的 activation 量化压缩方案
- GPipe、PipeDream: 经典流水线并行系统

## 相关概念

- [[Activation Quantization|激活量化]]
- [[Stochastic Gradient Descent|SGD]]
