---
type: concept
aliases: [Frobenius Norm, F-范数, 矩阵 F 范数]
---

# Frobenius 范数

## 定义

矩阵所有元素平方和的平方根，是矩阵的一种常用度量范数，等价于将矩阵视为向量后的 L2 范数。

## 数学形式

$$
\|\mathbf{A}\|_F = \sqrt{\sum_{i,j} a_{ij}^2} = \sqrt{\text{tr}(\mathbf{A}^{\top}\mathbf{A})}
$$

## 核心要点

1. **量化误差度量**: 在 PTQ 中广泛用于衡量量化前后权重输出的差异（$\|\mathbf{W}\mathbf{X} - \hat{\mathbf{W}}\mathbf{X}\|_F^2$）
2. **优化目标**: 最小化 Frobenius 范数误差等价于最小化所有输出维度的 MSE 之和
3. **计算性质**: $\|\mathbf{A}\|_F^2 = \text{tr}(\mathbf{A}\mathbf{A}^{\top})$，与 Hessian 矩阵推导密切相关

## 代表工作

- [[VEQ]]: 量化重建损失 $\mathcal{L} = \sum_i S_i \|\mathbf{W}_i\mathbf{X}_i - \hat{\mathbf{W}}_i\mathbf{X}_i\|_F^2$ 使用 Frobenius 范数

## 相关概念

- [[训练后量化|PTQ]]
- [[Hessian矩阵]]
