---
type: concept
aliases: [激活量化, Activation Compression, 激活压缩]
---

# Activation Quantization

## 定义

激活量化（Activation Quantization）是将神经网络前向传播中的中间 activation 张量从浮点精度（FP32/FP16）压缩为低比特整数表示（INT8/INT4/INT3）的技术，用于减少内存占用或跨设备通信量。

## 数学形式

对 activation 张量 $a$，标准均匀量化为：

$$
\hat{a} = \text{clip}\!\left(\left\lfloor \frac{a}{s} \right\rceil + z,\ 0,\ 2^b - 1\right)
$$

反量化：

$$
\tilde{a} = s \cdot (\hat{a} - z)
$$

其中 $s$ 为 scale，$z$ 为 zero-point，$b$ 为比特数。

## 核心要点

1. **训练 vs 推理**: 推理量化（PTQ/QAT）关注模型精度；训练时量化关注收敛性
2. **分组量化**: 按 tile/channel 分组独立计算 scale，减少极值影响
3. **动态范围问题**: activation 分布比权重更宽、更不规则，量化难度更大
4. **通信压缩**: pipeline 并行中用低比特量化减少跨机传输数据量

## 代表工作

- [[TAH-Quant]]: 结合分块量化、熵引导比特分配、Hadamard 变换实现 3–4 bit activation 量化
- [[AQ-SGD]]: 基于误差补偿的 activation 量化

## 相关概念

- [[Group Quantization|分组量化]]
- [[Hadamard Transform|Hadamard 变换]]
- [[Shannon熵]]
- [[Pipeline Parallelism|流水线并行]]
- [[训练后量化]]
