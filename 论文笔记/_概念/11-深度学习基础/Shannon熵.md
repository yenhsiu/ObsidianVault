---
type: concept
aliases: [Shannon Entropy, 信息熵, Information Entropy]
---

# Shannon熵

## 定义

离散概率分布 $p$ 的不确定性度量，由 Claude Shannon 于 1948 年提出。熵越高，分布越均匀（不确定性越大）；熵越低，分布越集中（不确定性越小）。

## 数学形式

$$
H(p) = -\sum_i p_i \log p_i, \quad \sum_i p_i = 1, \quad p_i \geq 0
$$

取值范围：$H(p) \in [0, \log N]$，均匀分布时取最大值 $\log N$，退化分布时取最小值 $0$。

## 核心要点

1. **信息论基础**: 是信源编码、信道容量等信息论核心概念的基础
2. **在深度学习中的应用**: 损失函数（交叉熵）、知识蒸馏、注意力分布分析
3. **与[[有效秩]]的关系**: erank 本质是奇异值分布的指数熵，两者思想相通
4. **[[注意力熵]]**: 将 Shannon 熵应用于视觉编码器注意力权重分布，量化图像复杂度

## 代表工作

- Shannon (1948): "A mathematical theory of communication" — 原始定义
- [[AgilePruner]]: 用注意力熵衡量视觉 token 注意力分布集中度（ICLR 2026）

## 相关概念

- [[有效秩]]
- [[注意力熵]]
