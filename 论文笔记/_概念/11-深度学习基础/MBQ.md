---
type: concept
aliases: [Modality-Balanced Quantization]
---

# MBQ（Modality-Balanced Quantization）

## 定义

针对视觉语言模型（VLM）的后训练量化方法，通过平衡视觉与文本 token 在量化校准中的贡献来缓解跨模态分布异质性导致的精度损失。

## 核心要点

1. **跨模态平衡**: 识别视觉与文本 token 的分布差异，在量化目标中进行平衡加权
2. **VLM 专用**: 相比 AWQ/GPTQ，专门针对多模态输入设计
3. **局限性**: 未考虑 MoE 结构中的专家异质性，在 MoE VLM 上存在性能瓶颈

## 代表工作

- [[VEQ]]: 在 MBQ 基础上进一步引入 MoE 专家感知，W3 配置下超越 MBQ 2.04-3.09%

## 相关概念

- [[训练后量化|PTQ]]
- [[视觉语言模型|VLM]]
- [[AWQ]]
- [[GPTQ]]
