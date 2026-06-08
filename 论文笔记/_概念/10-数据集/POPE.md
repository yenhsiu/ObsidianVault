---
type: concept
aliases: [Polling-based Object Probing Evaluation, POPE Benchmark]
---

# POPE

## 定义

面向视觉语言模型对象[[幻觉]]评估的二分类问答基准。通过询问"图像中是否存在某对象？"来测试模型是否会捏造不存在的对象，设计了 adversarial、popular、random 三种负样本采样策略。

## 核心要点

1. **评测形式**: 二分类 Yes/No VQA，准确率越高说明幻觉越少
2. **三种难度**: 
   - Random：随机采样不存在的对象
   - Popular：采样在 COCO 中高频出现的对象（更难）
   - Adversarial：采样与图像中存在对象共现率高的对象（最难）
3. **与[[CHAIR评测]]的互补**: POPE 评测是否存在特定对象（判别），CHAIR 评测生成描述中的幻觉（生成）
4. **复杂图像代表**: 在 AgilePruner 分析中，POPE 对应高[[有效秩]]的复杂图像，多样性剪枝在此表现更优

## 代表工作

- Li et al. (2023): "Evaluating object hallucination in large vision-language models" — 原始 POPE
- [[AgilePruner]]: 分析 POPE 上注意力 vs 多样性剪枝的性能规律，验证图像复杂度假设（ICLR 2026）

## 相关概念

- [[幻觉]]
- [[CHAIR评测]]
- [[视觉语言模型]]
- [[有效秩]]
