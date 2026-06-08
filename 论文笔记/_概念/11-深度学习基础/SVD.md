---
type: concept
aliases: [Singular Value Decomposition, 奇异值分解]
---

# SVD

## 定义

将任意矩阵 $A \in \mathbb{R}^{m \times n}$ 分解为三个矩阵乘积的数值分解方法：$A = U\Sigma V^T$，其中 $U, V$ 为正交矩阵，$\Sigma$ 为奇异值对角矩阵。

## 数学形式

$$
A = U \Sigma V^T, \quad U \in \mathbb{R}^{m \times m}, \quad \Sigma = \text{diag}(\sigma_1, \ldots, \sigma_r), \quad V \in \mathbb{R}^{n \times n}
$$

奇异值满足 $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r \geq 0$，$r = \text{rank}(A)$。

## 核心要点

1. **低秩近似**: 取前 $k$ 个奇异值可得最优秩-$k$ 近似（Eckart–Young 定理）
2. **[[有效秩]]计算**: erank 基于奇异值概率分布的指数熵定义
3. **计算成本**: 标准 SVD 复杂度 $O(\min(m,n) \cdot mn)$；当 $N \ll d$ 时可用协方差法降低复杂度
4. **深度学习应用**: 模型压缩、矩阵分解、[[有效秩]]度量、LoRA 等

## 代表工作

- [[AgilePruner]]: 用协方差特征值替代 SVD 奇异值，高效计算视觉 token 的 erank（ICLR 2026）

## 相关概念

- [[有效秩]]
- [[Frobenius范数]]
