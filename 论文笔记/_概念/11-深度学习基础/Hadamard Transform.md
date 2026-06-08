---
type: concept
aliases: [Hadamard 变换, WHT, Walsh-Hadamard Transform]
---

# Hadamard Transform

## 定义

Hadamard 变换（Walsh-Hadamard Transform）是一种正交线性变换，将输入向量均匀地"扩散"到所有输出维度，常用于将局部离群值重新分散，从而降低向量的动态范围。

## 数学形式

$$
\hat{x} = \frac{1}{\sqrt{G}} H_G x
$$

其中 $H_G$ 为 $G \times G$ Hadamard 矩阵，满足 $H_G H_G^\top = G I$，递归定义为：

$$
H_1 = [1], \quad H_{2n} = \begin{bmatrix} H_n & H_n \\ H_n & -H_n \end{bmatrix}
$$

## 核心要点

1. **正交性**: 变换保范数，$\|\hat{x}\|_2 = \|x\|_2$，不引入信息损失
2. **离群值分散**: 若 $x$ 中有极大值，变换后该能量被均匀分配到所有维度，降低动态范围
3. **计算高效**: 快速 Hadamard 变换复杂度 $O(G \log G)$，但 $G$ 小时直接矩阵乘法 $O(G^2)$ 更快
4. **Pivot Swapping**: 配合置换矩阵 $P_d$ 将离群元素先移至 index 0 再变换，提升数值稳定性

## 代表工作

- [[TAH-Quant]]: 用于 activation 量化前的离群值抑制，配合 pivot swapping 将离群点能量均匀扩散

## 相关概念

- [[Activation Quantization|激活量化]]
- [[Group Quantization|分组量化]]
- [[Shannon熵]]
