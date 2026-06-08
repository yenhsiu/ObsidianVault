---
type: concept
aliases: [Contrastive Language-Image Pre-training, CLIP ViT]
---

# CLIP

## 定义

OpenAI 提出的对比学习视觉-语言预训练模型，通过最大化配对图文的余弦相似度、最小化不配对图文的相似度来训练视觉编码器和文本编码器。在大量[[视觉语言模型]]中作为视觉骨干。

## 数学形式

对比损失（InfoNCE）：

$$
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(v_i, t_i)/\tau)}{\sum_{j=1}^{N} \exp(\text{sim}(v_i, t_j)/\tau)}
$$

其中 $v_i, t_i$ 为第 $i$ 对图文的视觉/文本嵌入，$\tau$ 为温度系数。

## 核心要点

1. **视觉编码器**: 通常为 ViT（Vision Transformer），输出 patch-level token（如 LLaVA-1.5 使用 CLIP ViT-L/14，产生 576 个视觉 token）
2. **在 LVLM 中的角色**: 作为视觉特征提取器，其输出 token 直接输入 LLM
3. **[[视觉Token剪枝]]的对象**: AgilePruner 等方法剪枝的正是 CLIP 输出的 576 个视觉 token
4. **CLS token**: 特殊分类 token，其注意力权重常被用于衡量各视觉 token 的重要性

## 代表工作

- Radford et al. (2021): "Learning transferable visual models from natural language supervision" — 原始 CLIP
- [[LLaVA]]: 以 CLIP ViT 作为视觉编码器的经典 LVLM
- [[AgilePruner]]: 对 CLIP 输出的视觉 token 进行自适应剪枝（ICLR 2026）

## 相关概念

- [[视觉语言模型]]
- [[视觉Token剪枝]]
- [[LLaVA]]
