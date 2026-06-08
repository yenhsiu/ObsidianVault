---
type: concept
aliases: [分组量化, Tile-Wise Quantization, Channel-Wise Quantization, 通道量化]
---

# Group Quantization

## 定义

分组量化（Group Quantization）是量化技术的一种变体，将张量沿某一维度（通常是 channel 维）切分为若干小组（group/tile），每组独立计算 scale 和 zero-point，从而为局部区域提供最优动态范围。

## 数学形式

设张量 $a \in \mathbb{R}^{B \times S \times C}$，沿 channel 维按 tile 大小 $G$ 分组：

$$
\text{group } j = a[:, :, jG : (j+1)G], \quad j = 0, 1, \ldots, \lceil C/G \rceil - 1
$$

每组独立量化：

$$
s_j = \frac{\max(|a_j|)}{2^{b-1} - 1}, \quad \hat{a}_j = \text{round}\!\left(\frac{a_j}{s_j}\right)
$$

## 核心要点

1. **优于全局量化**: 避免少数极大值决定整体 scale，大幅降低量化误差
2. **开销**: 每组需额外存储 scale/zero-point，元数据约占 $2 \times 16 / (G \times b)$ 的额外比特
3. **tile 大小权衡**: $G$ 越小精度越高但元数据开销越大；$G=32$–$64$ 常为最优点
4. **与 Hadamard 结合**: tile 内先做 Hadamard 变换抑制离群值，再做分组量化

## 代表工作

- [[TAH-Quant]]: 将 activation 按 $G=32$ 分 tile，配合熵引导比特分配实现 3–4 bit 量化
- [[GPTQ]]: 对权重矩阵使用分组量化

## 相关概念

- [[Activation Quantization|激活量化]]
- [[Hadamard Transform|Hadamard 变换]]
- [[训练后量化]]
