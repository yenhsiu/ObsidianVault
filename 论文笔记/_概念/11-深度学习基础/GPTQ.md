---
type: concept
aliases: [Generative Pre-trained Transformer Quantization]
---

# GPTQ

## 定义

一种基于近似 Hessian 信息的逐层（layer-wise）权重量化方法，通过最优化量化误差补偿（使用 OBQ/Cholesky 分解）在极低 bit 宽（3-4 bit）下实现大语言模型的高精度量化。

## 数学形式

逐列量化：对每一列权重量化后，用 Hessian 信息补偿剩余列的误差：

$$
\delta \mathbf{W} = -\frac{w_q - w}{[H^{-1}]_{qq}} \cdot H^{-1}_{:,q}
$$

其中 $w_q$ 为量化后的权重值，$H$ 为该层的 Hessian 矩阵近似。

## 核心要点

1. **逐层量化**: 将全模型量化问题分解为逐层的最优化子问题，降低计算复杂度
2. **Hessian 近似**: 用 $XX^{\top}$ 近似 Fisher 信息矩阵，再通过 Cholesky 分解高效求逆
3. **误差补偿**: 量化某列后，立即更新剩余列以补偿引入的量化误差

## 代表工作

- [[VEQ]]: VEQ-MA 在 GPTQ 框架上扩展，引入模态感知的增强 Hessian 矩阵

## 相关概念

- [[训练后量化|PTQ]]
- [[Hessian矩阵]]
- [[AWQ]]
