---
title: "AgilePruner: An Empirical Study of Attention and Diversity for Adaptive Visual Token Pruning in Large Vision-Language Models"
method_name: "AgilePruner"
authors: [Changwoo Baek, Jouwon Song, Sohyeon Kim, Kyeongbo Kong]
year: 2026
venue: ICLR 2026
tags: [visual-token-pruning, vision-language-model, hallucination, token-efficiency, attention-entropy]
zotero_collection: N/A
image_source: online
arxiv_html: https://arxiv.org/html/2603.01236v1
created: 2026-06-08
---

# 论文笔记：AgilePruner

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Pusan National University, LG Electronics |
| 日期 | March 2026 |
| 项目主页 | [cvsp-lab.github.io/AgilePruner](https://cvsp-lab.github.io/AgilePruner) |
| 对比基线 | [[DivPrune]], [[FasterVLM]], [[VisPruner]] |
| 链接 | [arXiv](https://arxiv.org/abs/2603.01236) |

---

## 一句话总结

> 通过[[有效秩]]來評估多項指標和[[注意力熵]]來調查visual token的處理機制並分析紛紛方法的優缺點。
> 定量分析結果發現，
> 	1. 很多多樣性導向的剪枝實際保留的多樣行遠低於預期，
> 	2. 使用 CHAIR 資料集表明，它們保留的多樣性與幻覺頻率的增加密切相關。
> 	3. 進一步觀察到attention-based的方法在簡單的image上有更有效，多樣性導像則跟是那些行複雜的token feature
> 	4. 基於上述的視點，我們證明要依據image的複雜度來調整目前現有的兩種token pruning手法才能一致的提升效能。
> 	5. 同時图像复杂度决定了应采用哪种[[视觉Token剪枝]]策略，并据此设计了一个自适应相似度阈值剪枝方法 AgilePruner，在 9 项多模态基准上以极低计算开销超越现有方法。

---

## 核心贡献

1. **实证诊断**: 首次用[[有效秩]]（erank）量化不同剪枝范式对特征多样性的保留程度，发现"多样性导向的剪枝方法实际保留的多样性远低于预期"，并与[[幻觉]]发生率存在正相关。
2. **图像复杂度依赖性**: 揭示了"注意力集中 vs. 多样性分布"这一数据级偏好规律——简单图像适合基于注意力的剪枝，复杂图像适合基于多样性的剪枝。
3. **自适应剪枝**: 将上述经验规律转化为可操作的 Adaptive Similarity Thresholding，无需训练，仅凭 erank 动态调节每个 token 的选取阈值，实现了在幻觉减少与召回率之间的最优平衡。

---

## 问题背景

### 要解决的问题

[[视觉语言模型]]（如 LLaVA-1.5）在处理高分辨率图像时会生成多达 576 个视觉 token，导致 LLM 的 self-attention 计算复杂度呈二次方增长。[[视觉Token剪枝]]是缓解这一瓶颈的主流方式，但不同剪枝范式各有优劣，且缺乏统一的理论框架来理解其差异。

### 现有方法的局限

- **基于注意力的剪枝**（如 [[FasterVLM]]、[[PruMerge+]]）：关注 CLS token 的注意力权重，倾向于选择视觉显著区域，有選到重複選擇的問題。但在需要综合多样化视觉证据的复杂场景下表现较弱。
- **基于多样性的剪枝**（如 [[DivPrune]]）：通过保留特征多样性来覆盖更多视觉信息，但实际上诱发更多[[幻觉]]（CHAIR 分数更差），原因是被保留的"多样"token 包含了低质量背景信息。
- hybrid
課題：不管是哪一種手法，目前都還是缺少足夠的特徵化
1. 如何真實保留特徵以及空間
2. 保留下的token的特性如何影響幻覺
3. 是否不同的image適用的token pruning方法定律成立
### 本文的动机
解決上述課題，本論文進行了兩部分的分析（？？？？）
1. 使用effective rank（e-rank）量化保留的多樣化 並檢視erank跟幻覺以及圖片形體的關係
2. 分析在圖像複雜度他們的有效性如何shift的
分析顯示：
- 多樣化的剪枝手法，保留的多樣性遠低於預期。同時保留越高的多型反而增加了幻覺的產生。
- 圖片的複雜性，attention based適用於簡單的圖像（精華的token被集中在一小部分）devisty based適用於複雜的圖片，語意的訊息被分布在各方token
两种范式的本质差异可通过[[注意力熵]]和[[有效秩]]两个度量来量化。如果能找到将图像复杂度（erank/熵）与剪枝策略选择相绑定的规律，就可以构建出自适应方案，避免固定策略带来的偏差。

基於分析
- 測試觀察到的現象可不可以直接被轉換成實際的加強
- image-aware adjustment
- threshold-based pruning procedure
	- The method explores tokens in descending attention order and removes redundant tokens based on similarity, using an adaptively set threshold informed by image-level complexity
	- 利用attention score 排序然後在用相似性移除多餘的token，同時依據image的複雜度來動態調整threshold
		- 在個task上有最好的結果，同時減輕了幻覺的傾向
		- -》說明不僅於分型在實際應用上也有效
- 貢獻：
	- 提出e-rank 評估手法
	- 有一致的結果，證明了有效性
	- 
Q ： effective rank

---

## 方法详解

### 模型架构

AgilePruner 是一个**无训练（training-free）**的自适应视觉 token 剪枝框架，作用于 [[视觉语言模型]] 的视觉编码器输出后、LLM 输入前：

- **输入**: 视觉 token 序列 $X \in \mathbb{R}^{N \times d_l}$（来自 [[CLIP]] 视觉编码器）
- **度量计算**: 计算当前图像的 [[有效秩]] $\text{erank}_{\text{input}}$ 与训练集均值 $\text{erank}_{\text{avg}}$ 之比，反映图像复杂度
- **核心模块**: Adaptive Similarity Thresholding — 基于 [[注意力熵]] 排序 + 自适应余弦距离阈值过滤
- **输出**: 压缩后的 $K$ 个视觉 token（K 可为 32/64/128）
- **额外开销**: erank 计算约 3.65 ms/样本（单 RTX 4090，批大小=1）

### 核心模块

#### 模块1: Attention Entropy 测量

**设计动机**: 利用[[注意力熵]]评估 token 注意力分布的集中程度，作为图像"简单性"的代理指标。

**具体实现**:
- 提取视觉编码器最后一层的 head-averaged 注意力权重 $\alpha \in \mathbb{R}^N$
- 排除 CLS token 的自注意力后归一化：$p_i = \alpha_i / \sum_{j \neq \text{CLS}} \alpha_j$
- 计算[[Shannon熵]]：$H(p) = -\sum_i p_i \log p_i$，低熵 → 注意力集中 → 图像简单

#### 模块2: Effective Rank (erank) 计算

**设计动机**: [[有效秩]]衡量 token 嵌入分布的"维度利用率"，高 erank 表示特征在多个方向均匀分布（图像复杂）。

**具体实现**:
- 对 token 矩阵 $X \in \mathbb{R}^{N \times d_l}$ 进行 [[SVD]] 分解，提取奇异值 $\{\sigma_i\}_{i=1}^{L}$（$L = \min(N, d_l)$）
- 归一化：$q_i = \sigma_i / \sum_{j=1}^{L} \sigma_j$
- 计算 erank：$\text{erank}(A) = \exp\left(-\sum_{i=1}^{L} q_i \log q_i\right)$
- **高效计算**（Appendix A）: 当 $N \ll d_l$ 时，改用协方差矩阵 $C = XX^T \in \mathbb{R}^{N \times N}$，取特征值 $\lambda(C)$ 代替 SVD，复杂度从 $O(ND^2)$ 降至 $O(N^2 D + N^3)$

#### 模块3: Adaptive Similarity Thresholding（核心剪枝步骤）

**设计动机**: 将图像复杂度（erank 比值）动态映射到选取阈值 $\tau$：复杂图像使用宽松阈值（保留更多多样性），简单图像使用严格阈值（保留高注意力 token）。

**具体实现（三步迭代选取）**:
1. 按注意力分数降序排列所有 token
2. 选取排名最高的 token，移除与它[[余弦相似度]]距离小于 $\tau_i$ 的邻居
3. 移至下一个未被剪枝的最高排名 token，重复直到达到目标数量 $K$

---

## 关键公式

### 公式1: [[注意力熵|注意力概率归一化]]

$$
p_i = \frac{\alpha_i}{\sum_{j \neq \text{CLS}} \alpha_j}, \quad \sum_i p_i = 1
$$

**含义**: 将视觉编码器 CLS token 对各视觉 token 的注意力权重归一化为概率分布，用于后续熵计算。

**符号说明**:
- $\alpha \in \mathbb{R}^N$: head-averaged 注意力分数（排除 CLS 自注意力）
- $p_i$: 第 $i$ 个 token 的归一化注意力概率

### 公式2: [[Shannon熵|注意力熵]]

$$
H(p) = -\sum_i p_i \log p_i
$$

**含义**: 量化注意力分布的集中程度。低熵表示注意力高度集中于少数 token（图像简单）；高熵表示注意力均匀分散（图像复杂）。

**符号说明**:
- $H(p)$: 注意力熵，值越低注意力越集中
- $p_i$: 归一化注意力概率（见公式1）

### 公式3: [[有效秩|奇异值归一化]]

$$
L = \min(N, d_l), \quad q_i = \frac{\sigma_i}{\sum_{j=1}^{L} \sigma_j}, \quad q_i \in \mathbb{R}^L
$$

**含义**: 对 token 嵌入矩阵做 [[SVD]] 分解后，将奇异值归一化为概率分布，为 erank 计算做准备。

**符号说明**:
- $\sigma_i$: 第 $i$ 个奇异值（降序）
- $N$: token 数量；$d_l$: embedding 维度
- $L$: 有效奇异值数量

### 公式4: [[有效秩|有效秩定义]]

$$
\text{erank}(A) = \exp\left(-\sum_{i=1}^{L} q_i \log q_i\right)
$$

**含义**: 有效秩本质上是奇异值分布的指数熵，取值范围 $[1, L]$。低 erank 说明嵌入集中在少数维度（低多样性）；高 erank 说明维度利用均匀（高多样性）。

**符号说明**:
- $q_i$: 归一化奇异值概率（见公式3）
- $L$: 奇异值数量
- $\text{erank} \in [1, L]$

### 公式5: [[CHAIR评测|CHAIR 幻觉度量]]

$$
C_I = \frac{|\{\text{hallucinated objects}\}|}{|\{\text{all mentioned objects}\}|}, \quad C_S = \frac{|\{\text{captions with hallucinated objects}\}|}{|\{\text{all captions}\}|}
$$

**含义**: $C_I$ 衡量所有提及对象中幻觉对象的比例；$C_S$ 衡量包含幻觉的描述比例。两者均越低越好。

**符号说明**:
- $C_I$: 对象级幻觉率（CHAIR_I）
- $C_S$: 描述级幻觉率（CHAIR_S）

### 公式6: [[视觉Token剪枝|自适应相似度阈值]]

$$
\tau_i = \text{order}_i \times \left(\frac{\text{erank}_{\text{input}}}{\text{erank}_{\text{avg}}} \times 0.01\right)
$$

**含义**: 阈值随 token 排名线性增大（越靠后的 token 选取越宽松），并根据当前图像相对于数据集平均复杂度的比值进行整体缩放；最终阈值由 $\tau_{\max}$ 上限约束。

**符号说明**:
- $\text{order}_i$: 当前 token 在注意力排序中的名次（1-based）
- $\text{erank}_{\text{input}}$: 当前图像的有效秩
- $\text{erank}_{\text{avg}}$: 训练集平均有效秩
- $0.01$: 基础步长超参数

### 公式7: [[有效秩|高效 erank 计算（协方差法）]]

$$
C = XX^T, \quad S = \sqrt{\lambda(C)}, \quad p_i = \frac{S_i}{\sum_j S_j}, \quad \text{erank}(X) = \exp\left(-\sum_i p_i \log p_i\right)
$$

**含义**: 当 token 数 $N \ll$ 特征维度 $d_l$ 时，用协方差矩阵特征值替代 SVD 奇异值，计算复杂度从 $O(ND^2)$ 降至 $O(N^2 D + N^3)$，使实时推理可行。

**符号说明**:
- $X \in \mathbb{R}^{N \times d_l}$: token 嵌入矩阵
- $C \in \mathbb{R}^{N \times N}$: 协方差矩阵
- $\lambda(C)$: $C$ 的特征值

---

## 关键图表

### Figure 1: 注意力熵与有效秩概览

![Figure 1 - Overview](https://arxiv.org/html/2603.01236v1/x1.png)

**说明**: 展示 AgilePruner 分析框架：从[[注意力熵]]（注意力集中度）和[[有效秩]]（token 嵌入多样性）两个维度评估视觉编码器。左侧为低熵/低 erank 的简单图像（注意力集中，适合注意力剪枝）；右侧为高熵/高 erank 的复杂图像（特征分散，适合多样性剪枝）。

### Figure 2: DivPrune vs FasterVLM 响应模式对比

![Figure 2 - Response comparison](https://arxiv.org/html/2603.01236v1/x2.png)

**说明**: 定性比较两种剪枝范式的生成差异。[[DivPrune]]（多样性导向）的描述更全面但易产生[[幻觉]]；[[FasterVLM]]（注意力导向）更聚焦安全但可能遗漏信息。AgilePruner 在两者间取得平衡。

### Figure 3: 跨数据集与图像复杂度的多样性 vs 注意力分析

![Figure 3 - Diversity vs attention analysis](https://arxiv.org/html/2603.01236v1/x3.png)

**说明**: (a) 高 erank 方法（多样性导向）在复杂数据集（POPE）上更优；低 erank 方法在简单数据集（ScienceQA）上更优。(b) 简单图像的[[注意力熵]]和 erank 均低，适合注意力剪枝；复杂图像反之。这是 AgilePruner 设计的核心经验依据。

### Figure 4: 相似度阈值 τ 的效果

![Figure 4 - Similarity threshold effect](https://arxiv.org/html/2603.01236v1/x4.png)

**说明**: 展示 $\tau$ 对 token 选取的影响。严格阈值（低 $\tau$）→ 倾向于选取高注意力 token；宽松阈值（高 $\tau$）→ 增加选取 token 的多样性。AgilePruner 通过公式6根据图像 erank 自适应调整 $\tau$。

### Figure 5 (Appendix B.2): MME 数据集上注意力熵与 erank 的相关性

![Figure 5 - Entropy vs erank scatter](https://arxiv.org/html/2603.01236v1/image.png)

**说明**: MME 数据集上注意力熵与 erank 的散点图，验证两者呈正相关（皮尔逊相关系数 > 0.8）。表明两种度量均可作为自适应阈值的输入信号，但 erank 在细粒度区分上更稳定（见 Table 11）。

---

### Table 1: POPE 上保留 64 个 token 的平均 erank

| 方法 | Erank |
|------|-------|
| PruMerge+ | 10.91 |
| VisionZip | 14.02 |
| VisPruner | 14.35 |
| DivPrune | 21.84 |

**关键发现**: DivPrune 虽标榜多样性，其保留 token 的 erank（21.84）仍远低于完整 token 集，说明"多样性导向"方法实际上并未完整保留特征多样性。

---

### Table 2: CHAIR 幻觉评测（64 tokens）

| 方法 | Cs↓ | Ci↓ | Recall↑ | Len |
|------|-----|-----|---------|-----|
| LLaVA-1.5-7B（全量） | 51.0 | 13.9 | 78.7 | 101.4 |
| **注意力导向方法** | | | | |
| FasterVLM (arXiv'24) | 45.4 | 13.5 | 69.3 | 94.0 |
| PruMerge+ (ICCV'25) | 45.2 | 15.6 | 66.7 | 91.4 |
| Vispruner (ICCV'25) | 49.8 | 15.0 | 72.6 | 96.7 |
| **多样性导向方法** | | | | |
| DivPrune (CVPR'25) | 57.4 | 18.0 | 76.4 | 101.1 |
| FPSPruner | 58.6 | 18.6 | 76.0 | 100.5 |

**关键发现**: 多样性导向方法（DivPrune）保留更高 Recall（76.4），但幻觉率（Cs=57.4）反而比原始模型更高。注意力导向方法幻觉少但 Recall 损失明显。

---

### Table 3: 注意力基选取比例 R 对 CHAIR 的影响

| R | Cs↓ | Ci↓ | Recall↑ | Len | Mean erank | Mean attn. |
|---|-----|-----|---------|-----|-----------|-----------|
| 0（纯多样性） | 57.4 | 18.0 | 76.4 | 101.1 | 21.14 | 0.0035 |
| 0.25 | 50.8 | 16.8 | 74.5 | 97.6 | 14.98 | 0.0065 |
| 0.50 | 46.2 | 14.5 | 73.7 | 95.5 | 14.38 | 0.0072 |
| 0.75 | 45.2 | 14.1 | 70.5 | 94.0 | 13.58 | 0.0076 |

**关键发现**: 随着注意力基比例增加，幻觉率持续下降但 Recall 也随之下降。说明纯策略无法同时优化两者，需要自适应混合。

---

### Table 4a/b: 简单 vs 复杂图像数据集的注意力熵与 erank

**(a) 简单图像数据集**

| 指标 | MME | ScienceQA | OCR | Numerical Cal. |
|------|-----|-----------|-----|----------------|
| 注意力熵 | 4.61 | 4.47 | 4.39 | 4.45 |
| Erank | 78 | 58 | 49 | 74 |
| 注意力法得分 | 140 | 55 | 100 | 69.51 |
| 多样性法得分 | 130 | 40 | 80 | 67.53 |

**(b) 复杂图像数据集**

| 指标 | MME | POPE | Position | Scene | Count |
|------|-----|------|----------|-------|-------|
| 注意力熵 | 4.90 | 4.86 | 4.82 | 4.87 | — |
| Erank | 109 | 103 | 102 | 106 | — |
| 注意力法得分 | 105 | 157 | 120 | 77.4 | — |
| 多样性法得分 | 111 | 168 | 140 | 86.0 | — |

**关键发现**: 简单图像（低熵低 erank）→ 注意力法胜出；复杂图像（高熵高 erank）→ 多样性法胜出。这一一致性规律支撑了自适应设计的合理性。

---

### Table 5: 多样性+注意力方法组合性能（将 AgilePruner 的自适应阈值插入现有方法）

**128 tokens**

| 方法 | GQA | SQAIMG | POPE | MME | Rel. |
|------|-----|--------|------|-----|------|
| LLaVA-1.5-7B | 61.9 | 69.5 | 85.9 | 1862 | 100.0% |
| BAT (CVPR'23) | 58.6 | 69.3 | 85.3 | 1737 | 96.75% |
| BAT + Inverse | 58.4 | 69.1 | 85.0 | 1734 | 96.46% |
| BAT + Ours | 58.8 | 69.4 | 86.8 | 1782 | **97.91%** |
| Vispruner (ICCV'25) | 58.2 | 69.0 | 84.6 | 1768 | 96.72% |
| Vispruner + Inverse | 57.9 | 68.8 | 85.3 | 1744 | 96.35% |
| Vispruner + Ours | 58.6 | 69.1 | 85.5 | 1787 | **97.32%** |

**64 tokens**

| 方法 | GQA | SQAIMG | POPE | MME | Rel. |
|------|-----|--------|------|-----|------|
| BAT (CVPR'23) | 56.6 | 68.8 | 81.5 | 1683 | 93.91% |
| BAT + Ours | 56.9 | 69.1 | 83.5 | 1692 | **94.85%** |
| Vispruner (ICCV'25) | 55.4 | 69.1 | 80.4 | 1650 | 92.78% |
| Vispruner + Ours | 55.9 | 69.3 | 81.5 | 1671 | **93.76%** |

**关键发现**: 将 AgilePruner 的自适应阈值替换掉现有方法的固定阈值后，BAT 和 VisPruner 性能均一致提升，验证了方法的通用性。

---

### Table 6: 自适应 vs 固定组合策略

**128 tokens**

| 方法 | GQA | SQAIMG | POPE | MME | Rel. |
|------|-----|--------|------|-----|------|
| LLaVA-1.5-7B | 61.9 | 69.5 | 85.9 | 1862 | 100.0% |
| Divprune (CVPR'25) | 59.4 | 68.6 | 87.0 | 1707 | 96.90% |
| FasterVLM (arXiv'24) | 57.9 | 68.5 | 83.2 | 1757 | 95.80% |
| Divprune + FasterVLM | 58.4 | 68.5 | 84.9 | 1768 | 96.66% |
| Divprune + FasterVLM（自适应） | 58.9 | 68.8 | 86.0 | 1787 | **97.55%** |

**64 tokens**

| 方法 | GQA | SQAIMG | POPE | MME | Rel. |
|------|-----|--------|------|-----|------|
| Divprune (CVPR'25) | 57.5 | 68.0 | 85.5 | 1615 | 94.25% |
| FasterVLM (arXiv'24) | 55.0 | 69.0 | 77.4 | 1665 | 91.91% |
| Divprune + FasterVLM | 56.8 | 68.6 | 82.2 | 1681 | 94.08% |
| Divprune + FasterVLM（自适应） | 57.2 | 68.9 | 83.5 | 1690 | **94.87%** |

**关键发现**: 简单相加两种方法（fixed）未必优于单一方法；自适应版本才能稳定提升，证明图像级动态调节的必要性。

---

### Table 7: 9 项多模态基准主实验结果（归一化相对全量 LLaVA-1.5-7B）

**128 tokens**

| 方法 | VQAv2 | GQA | VizWiz | SQAIMG | TextVQA | POPE | MME | MMB | MMBCN | Avg |
|------|-------|-----|--------|--------|---------|------|-----|-----|--------|-----|
| LLaVA-1.5-7B（全量） | 78.5 | 61.9 | 50.1 | 69.5 | 58.2 | 85.9 | 1862 | 64.7 | 58.1 | 100.00% |
| FastV (ECCV'24) | 71.0 | 54.0 | 51.9 | 69.2 | 56.4 | 68.2 | 1490 | 63.0 | 55.9 | 92.31% |
| PDrop (CVPR'25) | 74.3 | 57.1 | 49.4 | 70.1 | 56.7 | 77.5 | 1696 | 62.3 | 55.3 | 95.17% |
| SparseVLM (ICML'25) | 75.1 | 57.3 | 49.7 | 69.0 | 56.3 | 83.1 | 1761 | 62.6 | 56.9 | 96.61% |
| PruMerge+ (ICCV'25) | 75.0 | 58.2 | 53.7 | 69.1 | 54.0 | 83.1 | 1554 | 61.8 | 55.8 | 95.64% |
| VisionZip (CVPR'25) | 75.6 | 57.6 | 51.6 | 68.7 | 56.9 | 83.3 | 1763 | 62.1 | 57.0 | 97.19% |
| VisPruner (ICCV'25) | 75.8 | 58.2 | 52.7 | 69.0 | 57.0 | 84.6 | 1768 | 62.7 | 57.3 | 98.01% |
| DivPrune (CVPR'25) | 76.0 | 59.4 | 52.8 | 68.6 | 54.5 | 85.5 | 1707 | 60.1 | 52.3 | 97.25% |
| **Ours (AgilePruner)** | **76.4** | **59.4** | **53.0** | **68.6** | **57.0** | **87.4** | 1748 | 61.8 | 55.5 | **98.04%** |

**64 tokens**

| 方法 | VQAv2 | GQA | VizWiz | SQAIMG | TextVQA | POPE | MME | MMB | MMBCN | Avg |
|------|-------|-----|--------|--------|---------|------|-----|-----|--------|-----|
| FastV (ECCV'24) | 55.9 | 46.0 | 49.1 | 70.1 | 51.6 | 35.5 | 1256 | 50.1 | 42.1 | 76.86% |
| SparseVLM (ICML'25) | 66.9 | 52.0 | 49.4 | 69.2 | 52.1 | 69.7 | 1561 | 58.3 | 49.6 | 88.60% |
| PruMerge+ (ICCV'25) | 71.3 | 55.4 | 53.7 | 69.5 | 52.0 | 75.7 | 1640 | 59.6 | 52.1 | 92.76% |
| VisionZip (CVPR'25) | 72.4 | 55.1 | 52.9 | 68.7 | 55.5 | 77.0 | 1690 | 60.1 | 55.4 | 94.46% |
| VisPruner (ICCV'25) | 72.7 | 55.4 | 53.3 | 69.1 | 55.8 | 80.4 | 1650 | 61.3 | 55.1 | 95.07% |
| DivPrune (CVPR'25) | 74.1 | 57.5 | 53.6 | 68.0 | 54.5 | 85.5 | 1615 | 60.1 | 52.3 | 95.02% |
| **Ours (AgilePruner)** | **75.5** | 57.4 | **54.0** | **68.6** | **56.0** | 84.1 | **1703** | **60.7** | **55.8** | **96.76%** |

**32 tokens**

| 方法 | VQAv2 | GQA | VizWiz | SQAIMG | TextVQA | POPE | MME | MMB | MMBCN | Avg |
|------|-------|-----|--------|--------|---------|------|-----|-----|--------|-----|
| PruMerge+ (ICCV'25) | 65.6 | 52.9 | 53.5 | 67.9 | 49.2 | 66.7 | 1550 | 55.1 | 45.9 | 87.01% |
| VisionZip (CVPR'25) | 67.1 | 51.8 | 52.4 | 69.1 | 53.1 | 69.4 | 1579 | 57.0 | 50.3 | 89.41% |
| Vispruner (ICCV'25) | 67.7 | 52.2 | 53.0 | 69.2 | 53.9 | 72.7 | 1538 | 58.4 | 52.7 | 90.75% |
| DivPrune (CVPR'25) | 71.2 | 54.9 | 53.3 | 68.6 | 52.9 | 81.5 | 1594 | 57.6 | 49.1 | 92.16% |
| **Ours (AgilePruner)** | **74.0** | **54.1** | **53.4** | **69.0** | **54.5** | **80.1** | **1603** | **60.4** | **53.6** | **94.02%** |

---

### Table 8: CHAIR 幻觉完整评测（64 / 128 tokens）

| 方法 | Cs↓(64) | Ci↓(64) | Recall↑(64) | Len(64) | Cs↓(128) | Ci↓(128) | Recall↑(128) | Len(128) |
|------|---------|---------|------------|---------|---------|---------|-------------|---------|
| LLaVA-1.5-7B | 51.0 | 13.9 | 78.7 | 101.4 | 51.0 | 13.9 | 78.7 | 101.4 |
| FasterVLM (arXiv'24) | 45.4 | 13.5 | 69.3 | 94.0 | 45.8 | 13.3 | 75.4 | 97.0 |
| PruMerge+ (ICCV'25) | 45.2 | 15.6 | 66.7 | 91.4 | 46.8 | 14.4 | 71.5 | 95.2 |
| Vispruner (ICCV'25) | 49.8 | 15.0 | 72.6 | 96.7 | 52.8 | 15.4 | 77.1 | 98.7 |
| DivPrune (CVPR'25) | 57.4 | 18.0 | 76.4 | 101.1 | 58.6 | 18.1 | 78.4 | 103.1 |
| FPSPruner | 58.6 | 18.6 | 76.0 | 100.5 | 59.4 | 18.8 | 81.1 | 104.1 |
| **Ours (AgilePruner)** | **52.2** | **15.9** | **75.7** | **99.1** | **54.4** | **16.5** | **78.1** | **101.1** |

**关键发现**: AgilePruner 在幻觉（Cs/Ci）和召回率（Recall）之间取得最佳平衡，既避免了多样性方法的高幻觉，也保留了比注意力方法更高的信息覆盖率。

---

### Table 9 (Appendix A): 效率对比（单 RTX 4090，TextVQA）

| 方法 | 保留 Token | FLOPs (T) | 延迟 (ms/样本) | GPU 内存 (GB) | 准确率 |
|------|-----------|-----------|--------------|-------------|-------|
| Vanilla LLaVA-1.5-7B | 576 | 3.14 | 172 | 13.60 | 58.2 |
| PDrop (CVPR'25) | 64 | 0.51 | 128 | 13.30 | 55.0 |
| SparseVLM (ICML'25) | 64 | 0.52 | 129 | 16.26 | 55.2 |
| DivPrune (CVPR'25) | 64 | 0.48 | 110 | 13.30 | 55.8 |
| VisPruner (ICCV'25) | 64 | 0.48 | 115 | 13.30 | 55.4 |
| **Ours (AgilePruner)** | 64 | **0.48** | **115** | **13.30** | **56.0** |

**关键发现**: AgilePruner 将 FLOPs 从 3.14T 降至 0.48T（减少 89%），延迟与 VisPruner 持平，额外的 erank 计算仅增加约 3.65 ms（~3.2% 开销）。

---

### Table 10 (Appendix A): erank 计算开销

| 批大小 | 1 | 3 | 5 | 10 |
|-------|---|---|---|-----|
| erank 开销 (ms) | 3.65 | 10.21 | 16.84 | 33.40 |

---

### Table 11 (Appendix B.2): erank 阈值 vs 注意力熵阈值对比

| 方法 | GQA | SQAIMG | POPE | MME | TextVQA | Rel. |
|------|-----|--------|------|-----|---------|------|
| **128 tokens** | | | | | | |
| LLaVA-1.5-7B | 61.9 | 69.5 | 85.9 | 1862 | 58.2 | 100.0% |
| Erank-based（Ours） | 59.4 | 68.6 | **87.4** | 1748 | 57.0 | **98.13%** |
| 注意力熵-based | 59.4 | 68.6 | 87.0 | 1721 | 56.9 | 98.06% |
| **64 tokens** | | | | | | |
| Erank-based（Ours） | 57.4 | 68.6 | 84.1 | 1703 | 56.0 | **95.97%** |
| 注意力熵-based | 57.7 | 68.7 | 83.2 | 1690 | 56.0 | 95.84% |

**关键发现**: erank 与注意力熵性能接近（差距 <0.2%），但 erank 在 POPE 等复杂场景上略优，且在输入扰动下更鲁棒（见 Appendix E）。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| VQAv2 | ~200k QA | 开放域视觉问答 | 测试 |
| GQA | ~22M QA | 组合式视觉推理 | 测试 |
| VizWiz | ~31k | 无障碍场景，图像质量低 | 测试 |
| SQAIMG | ~21k | 科学题含图 | 测试 |
| TextVQA | ~45k | 图像中的文字识别 | 测试 |
| POPE | ~9k | 对象[[幻觉]]评估 | 测试 |
| MME | — | 细粒度多模态理解 | 测试 |
| MMBench | — | 多语言视觉语言 | 测试 |
| MMBench-CN | — | 中文版 MMBench | 测试 |
| CHAIR | ~500 images | 对象[[幻觉]]量化评测 | 测试 |

### 实现细节

- **基础模型**: LLaVA-1.5-7B（主实验）; LLaVA-1.5-13B, LLaVA-NeXT-7B, Qwen2.5-VL-7B（附录）
- **视觉 Token 数量**: 576（全量）→ 32 / 64 / 128（剪枝后）
- **erank 均值基准**: 在训练集上预先统计 $\text{erank}_{\text{avg}}$
- **硬件**: 单 RTX 4090（效率测试）
- **无需训练**: AgilePruner 完全无训练，推理时自适应

### 可视化结果

Figure 2 和 Figure F（Appendix）的定性分析显示，AgilePruner 能同时保留场景中的显著对象（如 FasterVLM）和背景多样信息（如 DivPrune），在细粒度推理任务（counting、position）上尤为明显。

---

## 批判性思考

### 优点

1. **分析严谨**: 用 erank 和注意力熵两个客观度量统一了两种剪枝范式的对比框架，实验设计完整（简单/复杂图像分组验证）。
2. **方法轻量**: 无需训练，额外计算开销极小（约 3.65 ms），可无缝插入现有 LVLM 推理流水线。
3. **通用性好**: 将自适应阈值插入 BAT、VisPruner 等不同方法均带来一致提升，说明发现具有普适性。
4. **ICLR 接收质量**: 实验基准覆盖全面（9 个基准 × 3 个 token 数量），并附有详尽的消融实验和效率分析。

### 局限性

1. **基于 LLaVA 验证为主**: 虽有 Qwen2.5-VL 附录实验，但主实验均以 LLaVA-1.5 为基础，未系统验证在 interleaved 图文 / 多图场景的泛化性。
2. **erank 基准值依赖训练集统计**: $\text{erank}_{\text{avg}}$ 需要在训练集上预计算，跨分布或新场景下可能失效。
3. **阈值超参数**: $\tau_{\max}$ 和 0.01 步长参数未做充分的敏感性分析。
4. **没有比较 KV cache 级别的压缩方案**: 只比较了视觉 token 剪枝，未涉及 KV cache 合并等正交优化。

### 潜在改进方向

1. 将自适应阈值扩展到 LLM 层内的 visual token 动态退出（layer-wise pruning）
2. 结合 token merging（如 ToMe）替代单纯剪枝，进一步降低信息损失
3. 在视频 LVLM 场景验证时序 erank 的变化规律

### 可复现性评估

- [ ] 代码开源（项目主页有，但论文未明确 code 链接）
- [ ] 预训练模型（基于 LLaVA-1.5-7B，公开可获取）
- [x] 训练细节完整（无训练，仅推理参数）
- [x] 数据集可获取（均为公开基准）

---

## 关联笔记

### 基于

- [[视觉语言模型]]: 本文聚焦于 LVLM 的视觉 token 压缩
- [[LLaVA]]: 主实验基础模型
- [[CLIP]]: 视觉编码器来源，token 来自 CLIP ViT 输出

### 对比

- [[DivPrune]]: 多样性导向剪枝代表方法（CVPR'25）
- [[FasterVLM]]: 注意力导向剪枝代表方法
- [[VisPruner]]: 注意力导向方法（ICCV'25）
- [[PruMerge+]]: 注意力+合并方法（ICCV'25）
- [[VisionZip]]: 另一注意力导向方法（CVPR'25）

### 方法相关

- [[视觉Token剪枝]]: 核心研究主题
- [[有效秩]]: 关键多样性度量
- [[注意力熵]]: 关键集中度度量
- [[CHAIR评测]]: 幻觉评估基准

### 硬件/数据相关

- [[POPE]]: 对象幻觉评估数据集

---

## 速查卡片

> [!summary] AgilePruner (ICLR 2026)
> - **核心**: 发现图像复杂度决定注意力 vs 多样性剪枝的优劣；据此提出自适应阈值剪枝
> - **方法**: erank 度量图像复杂度 → Adaptive Similarity Thresholding 动态调节 token 选取
> - **结果**: 128 tokens 达到 98.04% 全量性能；64 tokens 达到 96.76%，幻觉与召回兼顾最优
> - **代码**: [cvsp-lab.github.io/AgilePruner](https://cvsp-lab.github.io/AgilePruner)

---

*笔记创建时间: 2026-06-08*
