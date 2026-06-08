---
type: concept
aliases: [SGD, 随机梯度下降, Momentum SGD]
---

# Stochastic Gradient Descent

## 定义

随机梯度下降（SGD）是深度学习最基础的优化算法，用随机采样的 mini-batch 梯度代替完整梯度，迭代更新模型参数以最小化期望损失。

## 数学形式

标准带动量的 SGD（Heavy Ball）：

$$
\min_{x \in \mathbb{R}^d} \mathbb{E}_{\xi \sim \mathcal{D}}[F(x;\xi)]
$$

$$
m^t = (1-\beta_1) m^{t-1} + \beta_1 g^t, \quad g^t = \nabla F(x^t; \xi^t)
$$

$$
x^{t+1} = x^t - \eta m^t
$$

收敛率（光滑非凸目标）：

$$
\frac{1}{T}\sum_{t=1}^T \mathbb{E}[\|\nabla f(x^t)\|^2] = \mathcal{O}\!\left(\frac{1}{\sqrt{T}}\right)
$$

## 核心要点

1. **无偏性**: $\mathbb{E}[g^t] = \nabla f(x^t)$，随机梯度是真实梯度的无偏估计
2. **方差**: $\mathbb{E}[\|g^t - \nabla f(x^t)\|^2] \leq \sigma^2$，方差有界
3. **动量**: 指数移动平均平滑梯度噪声，加速收敛
4. **收敛保证**: 在光滑性 + 有界方差假设下，O(1/√T) 为非凸问题的最优速率

## 代表工作

- [[TAH-Quant]]: 量化梯度下，证明 TAH-Quant 保持与 vanilla SGD 相同的 O(1/√T) 收敛速率

## 相关概念

- [[Activation Quantization|激活量化]]
- [[Pipeline Parallelism|流水线并行]]
