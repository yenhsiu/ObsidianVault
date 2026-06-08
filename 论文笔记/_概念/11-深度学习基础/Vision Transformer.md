---
type: concept
aliases: [ViT, Vision Transformer, 视觉Transformer]
---

# Vision Transformer（ViT）

## 定义

Vision Transformer（ViT）是将原始 Transformer 架构直接应用于图像分类任务的模型，将图像切分为固定大小的 patch 序列，通过 self-attention 捕捉全局上下文，在大规模预训练后达到与 CNN 相当甚至更优的性能。

## 数学形式

将输入图像 $\mathbf{I} \in \mathbb{R}^{H \times W \times C}$ 切分为 $N$ 个 patch，线性投影后加入 class token $\mathbf{x}_{\text{cls}}$ 和位置编码：

$$
\mathbf{Z}_0 = [\mathbf{x}_{\text{cls}};\ \mathbf{x}_p^1 \mathbf{E};\ \ldots;\ \mathbf{x}_p^N \mathbf{E}] + \mathbf{E}_{\text{pos}}
$$

$$
\mathbf{Z}_l = \text{MSA}(\text{LN}(\mathbf{Z}_{l-1})) + \mathbf{Z}_{l-1}
$$

$$
\mathbf{Z}_l' = \text{MLP}(\text{LN}(\mathbf{Z}_l)) + \mathbf{Z}_l
$$

最终取 class token $\mathbf{Z}_L^0$ 用于分类。

## 核心要点

1. **Patch 嵌入**: 将 $P \times P$ 的图像块线性映射为 $D$ 维向量，$N = HW/P^2$ 为序列长度
2. **Class Token（[CLS]）**: 额外附加的可学习 token，聚合全图信息用于分类
3. **Multi-Head Self-Attention（MSA）**: 捕捉所有 patch 间的全局依赖，计算复杂度 $\mathcal{O}(N^2 D)$
4. **Layer Normalization（Pre-LN）**: ViT 采用 pre-normalization，在每个子层前做 LayerNorm
5. **GELU 激活**: MLP block 使用 [[GELU]] 激活函数（区别于 Transformer 中的 ReLU）
6. **Scale 敏感性**: 与 CNN 不同，ViT 激活值在不同 token 和 channel 间动态范围差异极大，给量化带来挑战

## 代表工作

- [An Image is Worth 16x16 Words (ViT, ICLR 2021)](https://arxiv.org/abs/2010.11929): 原始 ViT 论文
- [DeiT (ICML 2021)](https://arxiv.org/abs/2012.12877): 知识蒸馏增强的高效 ViT
- [Swin Transformer (ICCV 2021)](https://arxiv.org/abs/2103.14030): 分层窗口 attention 的 ViT 变体
- [[TokenBitViT]]: Token 级动态比特宽度量化，针对 ViT 激活 token 间动态范围差异

## 相关概念

- [[训练后量化]]: ViT 部署压缩的主要手段
- [[Group Quantization]]: 处理 ViT channel-wise 分布变化的分组量化
- [[GELU]]: ViT MLP block 使用的激活函数
- [[混合精度量化]]: 对不同 token/层分配不同比特宽度
- [[Activation Quantization]]: ViT 推理量化的核心挑战
