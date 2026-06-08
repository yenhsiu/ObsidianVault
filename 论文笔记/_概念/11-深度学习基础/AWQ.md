---
type: concept
aliases: [Activation-aware Weight Quantization, 激活感知权重量化]
---

# AWQ（Activation-aware Weight Quantization）

## 定义

一种后训练权重量化方法，通过识别并保护对激活值影响最大的权重通道（salient channels），使用 per-channel 缩放将重要通道对齐到量化友好范围，从而在低 bit 宽下保持精度。

## 数学形式

$$
\hat{\mathbf{W}} = Q(\mathbf{W} \cdot \mathbf{s}^{-1}) \cdot \mathbf{s}
$$

其中 $\mathbf{s}$ 为每通道缩放因子，由激活统计确定，对重要通道放大以降低量化误差。

## 核心要点

1. **激活感知**: 不直接量化激活，而是利用激活分布统计来指导权重缩放
2. **Salient Channel 保护**: 激活均值大的通道权重对输出影响大，需优先保护
3. **通用性**: 适用于 Dense LLM，但不感知多模态或 MoE 结构

## 代表工作

- [[VEQ]]: 将 AWQ 作为基线，在 MoE VLM 量化上显著超越 AWQ

## 相关概念

- [[训练后量化|PTQ]]
- [[GPTQ]]
- [[MBQ]]
