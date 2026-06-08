---
type: concept
aliases: [CHAIR, CHAIR Hallucination Evaluation, Caption Hallucination Assessment with Image Relevance]
---

# CHAIR评测

## 定义

Rohrbach et al. (2018) 提出的对象[[幻觉]]评估基准，通过对比图像描述中提到的对象与图像中实际存在的对象，定量衡量视觉语言模型的幻觉程度。

## 数学形式

$$
C_I = \frac{|\{\text{hallucinated objects}\}|}{|\{\text{all mentioned objects}\}|}, \quad C_S = \frac{|\{\text{captions with hallucinated objects}\}|}{|\{\text{all captions}\}|}
$$

- $C_I$（CHAIR_I）: 对象级幻觉率，值越低越好
- $C_S$（CHAIR_S）: 句子级幻觉率，值越低越好
- Recall: 图像中真实对象被描述覆盖的比例，值越高越好

## 核心要点

1. **评测对象**: 图像描述（captioning）中的对象[[幻觉]]
2. **数据集基础**: 基于 COCO 验证集（约 500 张图像），利用 COCO 标注的对象类别
3. **幻觉-召回 tradeoff**: 减少幻觉（低 Cs/Ci）通常以牺牲 Recall 为代价；理想方法需兼顾两者
4. **与[[POPE]]的区别**: POPE 是二分类 VQA 形式（"Is there a dog?"），CHAIR 是生成级别评测

## 代表工作

- Rohrbach et al. (2018): "Object hallucination in image captioning" — 原始 CHAIR 基准
- [[AgilePruner]]: 在 CHAIR 上验证自适应剪枝对幻觉的改善（ICLR 2026）

## 相关概念

- [[幻觉]]
- [[POPE]]
- [[视觉语言模型]]
