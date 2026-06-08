---
type: concept
aliases: [LLaVA-1.5, Large Language and Vision Assistant]
---

# LLaVA

## 定义

Liu et al. (2023) 提出的视觉指令调优模型，将 [[CLIP]] 视觉编码器与大型语言模型（LLaMA / Vicuna）通过 MLP projector 连接，是目前众多视觉 token 剪枝研究的标准基础模型。

## 核心架构

- **视觉编码器**: CLIP ViT-L/14，输出 576 个视觉 token（LLaVA-1.5 使用 336px 输入）
- **连接层**: 2 层 MLP Projector 将视觉特征映射到语言空间
- **语言模型**: Vicuna-7B / 13B（LLaVA-1.5）

## 核心要点

1. **视觉 token 数量**: 576（LLaVA-1.5 标准配置），是[[视觉Token剪枝]]研究的主要压缩对象
2. **指令跟随**: 通过视觉指令调优数据训练，支持开放域视觉问答和图像描述
3. **基准地位**: 大量视觉 token 剪枝论文（AgilePruner、DivPrune、VisPruner 等）均以 LLaVA-1.5-7B 为主要测试基线
4. **多版本**: LLaVA-1.5-7B（主流）、LLaVA-1.5-13B、LLaVA-NeXT（支持高分辨率）

## 代表工作

- Liu et al. (2023): "Visual instruction tuning" — 原始 LLaVA
- Liu et al. (2024): "Improved baselines with visual instruction tuning" — LLaVA-1.5
- [[AgilePruner]]: 在 LLaVA-1.5-7B 上验证自适应视觉 token 剪枝（ICLR 2026）

## 相关概念

- [[视觉语言模型]]
- [[CLIP]]
- [[视觉Token剪枝]]
