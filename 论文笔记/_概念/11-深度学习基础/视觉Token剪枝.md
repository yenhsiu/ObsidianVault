---
type: concept
aliases: [Visual Token Pruning, 视觉token压缩, Token Pruning]
---

# 视觉Token剪枝

## 定义

在大型[[视觉语言模型]]推理阶段，将视觉编码器（如 [[CLIP]] ViT）产生的冗余视觉 token 剪除，只保留信息量最高的 $K$ 个，从而降低 LLM 部分的计算复杂度。

## 数学形式

$$
X' = \text{Select}(X, K) \subset X, \quad |X'| = K \ll N
$$

其中 $X \in \mathbb{R}^{N \times d}$ 为全量视觉 token，$K$ 为保留数（如 32/64/128）。

## 核心要点

1. **动机**: LLM self-attention 计算量为 $O(N^2)$，减少视觉 token 数可显著降低 FLOPs 和显存
2. **两大范式**: 
   - **基于注意力**: 保留 CLS token 注意力权重高的 token（显著区域）
   - **基于多样性**: 保留特征分布多样的 token（覆盖面广）
3. **权衡**: 注意力剪枝幻觉少但召回低；多样性剪枝召回高但易产生[[幻觉]]
4. **自适应方案**: 根据图像复杂度（[[有效秩]]、[[注意力熵]]）动态选择策略

## 代表工作

- [[AgilePruner]]: 基于 erank 的自适应相似度阈值剪枝（ICLR 2026）
- [[DivPrune]]: 多样性导向剪枝（CVPR'25）
- [[FasterVLM]]: 注意力导向剪枝
- [[VisPruner]]: 注意力导向方法（ICCV'25）

## 相关概念

- [[视觉语言模型]]
- [[有效秩]]
- [[注意力熵]]
- [[幻觉]]
