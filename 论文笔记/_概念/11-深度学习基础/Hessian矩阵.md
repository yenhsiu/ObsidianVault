---
type: concept
aliases: [Hessian, Hessian Matrix, 海森矩阵]
---

# Hessian 矩阵

## 定义

损失函数关于模型参数的二阶偏导数矩阵，描述损失曲面的曲率，用于衡量参数扰动（如量化误差）对模型输出的影响程度。

## 数学形式

$$
H_{ij} = \frac{\partial^2 \mathcal{L}}{\partial w_i \partial w_j}
$$

在量化中，Hessian 近似为输入激活的外积：

$$
H \approx XX^{\top}
$$

其中 $X$ 为对应层的输入激活矩阵。

## 核心要点

1. **量化重要性度量**: 对角元素 $H_{ii}$ 越大，该参数对量化误差越敏感
2. **计算近似**: 精确 Hessian 计算代价高，实际中用 $XX^{\top}$（Fisher 信息矩阵近似）代替
3. **增强 Hessian（VEQ-MA）**: 通过加权 $\tilde{H} = XCX^{\top}$ 引入 token 重要性差异，更准确估计量化敏感性

## 代表工作

- [[GPTQ]]: 首次在大模型量化中用 Hessian 指导权重更新
- [[VEQ]]: VEQ-MA 提出增强 Hessian，融入路由亲和度与模态信息

## 相关概念

- [[训练后量化|PTQ]]
- [[GPTQ]]
