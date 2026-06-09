---
title: "Prune Redundancy, Preserve Essence: Vision Token Compression in VLMs via Synergistic Importance-Diversity"
method_name: "PruneSID"
authors: [Zhengyao Fang, Pengyuan Lyu, Chengquan Zhang, Guangming Lu, Jun Yu, Wenjie Pei]
year: 2026
venue: ICLR 2026
tags: [vision-token-compression, vlm-efficiency, token-pruning, pca, non-maximum-suppression, training-free, multimodal]
zotero_collection: _待整理
image_source: mixed
arxiv_html: https://arxiv.org/html/2603.09480
created: 2026-06-05
---

# 论文笔记：PruneSID — Vision Token Compression in VLMs

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Harbin Institute of Technology (Shenzhen)、Peng Cheng Laboratory |
| 日期 | March 2026 |
| 项目主页 | — |
| 对比基线 | [[VisionZip]]、[[FastV]]、[[ToMe]] |
| 链接 | [arXiv](https://arxiv.org/abs/2603.09480) / [Code](https://github.com/ZhengyaoFang/PruneSID) |

---

## 一句话总结

> PruneSID 是一个 training-free 的视觉 token 压缩框架，用 [[主成分分析|PCA]] 驱动的语义分组加 [[非极大值抑制|NMS]] 去冗余，在 11.1% token 保留率下达到 96.3% 原始精度，推理预填充速度提升 7.8×。

---

## 核心贡献

1. **Principal Semantic Components Analysis (PSCA)**: 对 token 维度而非特征维度做 PCA 分解，把视觉 token 聚类为语义一致的组，确保覆盖多样概念。
2. **Intra-Group Non-Maximum Suppression (NMS)**: 在每个语义组内用自适应相似度阈值去除冗余 token，保留最具代表性的 token。
3. **Information-Aware Dynamic Compression Ratio**: 根据图像信息量自动调节每张图像的 token 保留预算，复杂图像多保留、简单图像压缩更猛。

---

## 问题背景

### 要解决的问题

现代 [[视觉语言模型|VLM]] 为了提升视觉细节感知能力，使用高分辨率图像编码，每张图像会产生 576–2880 个视觉 token。这些大量的 token 导致 [[自回归语言模型|LLM]] 的注意力计算复杂度呈平方增长，严重拖慢推理速度。

### 现有方法的局限

| 范式 | 代表方法 | 局限 |
|------|----------|------|
| **注意力引导** | [[FastV]]、[[SparseVLM]]、[[VisionZip]] | 保留高注意力 token，但可能丢弃上下文背景信息 |
| **去重复性** | [[DART]]、[[DivPrune]] | 移除相似 token，但可能丢弃语义重要的区域 |

两类方法都只关注单一维度（重要性或多样性），无法同时优化二者。

### 本文的动机

语义重要 token 中存在大量冗余（同一区域多个相似 token），而纯去重则可能误删重要 token。PruneSID 提出在 **语义分组内部** 做去冗余，天然平衡语义覆盖率与信息多样性。

---

## 方法详解

### 模型架构

PruneSID 是一个 **training-free、plug-and-play** 的压缩模块，插入在 [[Vision Transformer|ViT]] 编码器之后、[[大语言模型|LLM]] 之前：

- **输入**: [[ViT]] 编码后的视觉 token 序列 $\mathbf{X} \in \mathbb{R}^{T \times D}$（$T$ = token 数，$D$ = 特征维度）
- **Stage 1**: [[主成分分析|PSCA]] — 语义感知 token 分组
- **Stage 2**: [[非极大值抑制|NMS]] — 组内冗余去除
- **Dynamic Ratio**: 信息感知动态压缩率
- **输出**: 压缩后的 token 序列 $\widetilde{\mathbf{X}} \in \mathbb{R}^{N \times D}$，$N \ll T$

![[PruneSID_fig1.png|600]]

**说明**: PSCA 先通过低秩 PCA 分解把视觉 token 聚类为语义一致的组，再由组内 NMS 用自适应相似度阈值 $\tau$ 去除冗余 token，保留最具信息量的代表性 token。

---

### 核心模块

#### 模块 1: Principal Semantic Components Analysis (PSCA)

**设计动机**: 利用 [[主成分分析|PCA]] 识别跨 token 的全局语义方向，而非传统 PCA 在特征维度上的方差分析。

**具体实现**:

1. 对 token 特征用 [[Sigmoid 激活函数|Sigmoid]] 归一化并中心化

$$
\mathbf{X}_{\text{ctr}} = \sigma(\mathbf{X}) - \mu, \quad \mu = \frac{1}{T}\sum_{i=1}^{T}\sigma(\mathbf{x}_i)
$$

2. 对转置矩阵做低秩 [[SVD 分解|SVD]]（在 token 维度而非特征维度上）

$$
\mathbf{X}_{\text{ctr}}^\top \approx \mathbf{U}\mathbf{S}\mathbf{V}^\top
$$

3. 每个 token 分配到其贡献最大的主成分方向所在的组

$$
g(i) = \arg\max_j \|\mathbf{V}_{i,j}\|
$$

**关键洞察**: $\mathbf{V}$ 的列向量代表主语义方向，每行 $\mathbf{V}_i$ 表示第 $i$ 个 token 对各语义成分的贡献，按最大绝对值分配组别可保证语义一致性。

**层选择**: 实验发现 [[Vision Transformer|ViT]] 中间偏后层（第 22 层）的特征最适合做 PSCA 分组，能更好地捕捉语义结构。

---

#### 模块 2: Intra-Group NMS

**设计动机**: 在同一语义组内，相邻 token 在空间或语义上高度相似，存在大量冗余，用 [[非极大值抑制|NMS]] 策略可在保留代表性 token 的同时消除冗余。

**具体实现**:

1. 每个 token 的重要性分数 = 对所在组主方向的贡献：$s_i = |\mathbf{V}_{i,g(i)}|$
2. 按 $s_i$ 降序排列 token
3. 贪心保留：若当前 token 与已保留 token 的最大[[余弦相似度]]低于阈值 $\tau$，则保留

$$
\text{sim}(\mathbf{x}_i, \mathbf{x}_j) = \frac{\mathbf{x}_i^\top \mathbf{x}_j}{\|\mathbf{x}_i\| \cdot \|\mathbf{x}_j\|}
$$

4. 全局冗余分数（图像级平均相似度）：

$$
\rho = \frac{2}{T(T-1)}\sum_{i=1}^{T}\sum_{j=i+1}^{T} \text{sim}(\mathbf{x}_i, \mathbf{x}_j)
$$

5. 自适应阈值：$\tau = \lambda \cdot \rho$，其中 $\lambda = N/32$（$N$ 为全局 token 预算）
6. 各组按比例分配 token 名额后，取各组 top-$n_k$ token 合并输出

$$
\widetilde{\mathbf{X}} = \bigcup_{k=1}^{K} \text{Top}_{n_k}(\widetilde{G}_k)
$$

---

#### 模块 3: Information-Aware Dynamic Compression Ratio

**设计动机**: 固定压缩率对所有图像一刀切——语义丰富的图像被过度压缩，语义简单的图像仍有冗余。

**具体实现**:

1. 图像信息分数 $\phi = 1 - \rho$（越大表示 token 间差异越大，语义越丰富）

$$
\phi = 1 - \rho
$$

2. 根据 $\phi$ 动态分配 token 数 $N'$，信息量大的图保留更多 token，信息量小的图压缩更激进

---

### 理论基础：PSCA–NMS 的信息论视角

通过[[容斥原理]]，保留子集 $S'$ 的有效信息下界为：

$$
\text{Inform}(S') \geq \sum_{s_i \in S'} I(s_i) - \sum_{s_i, s_j \in S'} R(s_i, s_j)
$$

优化目标是在固定预算 $N$ 下最大化有效信息：

$$
\max_{\|S'\|=N} \sum_{s_i \in S'} I(s_i) \quad \text{s.t.} \quad R(s_i, s_j) \leq \epsilon
$$

- **PSCA** 优化第一项：选择对主语义方向贡献最大的 token（最大化 $\sum I(s_i)$）
- **NMS** 优化约束：抑制相似 token，强制多样性（满足 $R(s_i, s_j) \leq \epsilon$）

---

## 关键公式

### 公式 1: [[Sigmoid 激活函数|Token 归一化与中心化]]

$$
\mathbf{X}_{\text{ctr}} = \sigma(\mathbf{X}) - \mu, \quad \mu = \frac{1}{T}\sum_{i=1}^{T}\sigma(\mathbf{x}_i)
$$

**含义**: 对视觉 token 做 Sigmoid 归一化并减去均值，消除量纲差异，使 PCA 分解聚焦语义变化。

**符号说明**:
- $\sigma(\cdot)$: Sigmoid 激活函数，将特征映射到 $(0, 1)$
- $T$: token 总数
- $\mu$: token 均值（逐维度）

---

### 公式 2: [[SVD 分解|低秩 SVD 分解]]

$$
\mathbf{X}_{\text{ctr}}^\top \approx \mathbf{U}\mathbf{S}\mathbf{V}^\top
$$

**含义**: 对转置矩阵（token 维度为主轴）做低秩 SVD，$\mathbf{V}$ 的行向量编码每个 token 对各主语义成分的贡献。

**符号说明**:
- $\mathbf{U} \in \mathbb{R}^{D \times K}$: 特征空间的左奇异向量
- $\mathbf{S} \in \mathbb{R}^{K \times K}$: 奇异值对角矩阵
- $\mathbf{V} \in \mathbb{R}^{T \times K}$: token 在主语义空间的投影（分组依据）
- $K$: 语义组数，$K = \lfloor N/4 \rfloor$

---

### 公式 3: [[PCA 分组|Token 语义组分配]]

$$
g(i) = \arg\max_j \|\mathbf{V}_{i,j}\|
$$

**含义**: 每个 token 分配到其在主语义方向上贡献绝对值最大的组，实现语义感知的离散分组。

**符号说明**:
- $g(i)$: token $i$ 的组别
- $\mathbf{V}_{i,j}$: token $i$ 对第 $j$ 个主成分的投影值

---

### 公式 4: [[余弦相似度|Token 间相似度]]

$$
\text{sim}(\mathbf{x}_i, \mathbf{x}_j) = \frac{\mathbf{x}_i^\top \mathbf{x}_j}{\|\mathbf{x}_i\| \cdot \|\mathbf{x}_j\|}, \quad \forall \mathbf{x}_i, \mathbf{x}_j \in \mathbf{X}
$$

**含义**: 用余弦相似度度量两个 token 的语义相似程度，作为 NMS 的判别依据。

---

### 公式 5: [[图像冗余度|全局冗余分数]]

$$
\rho = \frac{2}{T(T-1)}\sum_{i=1}^{T}\sum_{j=i+1}^{T} \text{sim}(\mathbf{x}_i, \mathbf{x}_j)
$$

**含义**: 计算所有 token 对的平均余弦相似度，量化图像整体冗余程度；$\rho$ 越高表示 token 越相似、图像越简单。

---

### 公式 6: [[图像信息分数|信息分数]]

$$
\phi = 1 - \rho
$$

**含义**: 图像信息量的度量，$\phi$ 越高表示语义越丰富，应分配更多 token 预算。

---

### 公式 7: [[信息论|有效信息下界]]（理论）

$$
\text{Inform}(S') \geq \sum_{s_i \in S'} I(s_i) - \sum_{s_i, s_j \in S'} R(s_i, s_j)
$$

**含义**: 用[[容斥原理]]将子集的有效信息量下界分解为语义重要性之和减去冗余惩罚项，理论上指导 PSCA + NMS 的联合优化目标。

**符号说明**:
- $I(s_i)$: token $s_i$ 的语义重要性
- $R(s_i, s_j)$: token $s_i$ 与 $s_j$ 的冗余量

---

### 公式 8: [[优化目标|Token 保留优化问题]]

$$
\max_{\|S'\|=N} \sum_{s_i \in S'} I(s_i) \quad \text{s.t.} \quad R(s_i, s_j) \leq \epsilon
$$

**含义**: 在固定 token 预算 $N$ 下，最大化保留子集的语义重要性总和，同时约束任意两个保留 token 之间的冗余度不超过 $\epsilon$。

---

## 关键图表

### Figure 1: 视觉 Token 压缩范式对比

![[PruneSID_fig2.png|600]]

**说明**: (a) 原始图像；(b) 注意力引导方法保留高注意力 token，但丢弃上下文背景；(c) 去重复性方法通过相似度剪枝移除冗余，但可能误删高注意力的语义重要区域；(d) PruneSID 的语义分组引导方法同时平衡语义重要性与信息多样性。

---

### Figure 2: 多 Benchmark 性能对比（LLaVA-1.5）

![[PruneSID_fig3.png|600]]

**说明**: 在 LLaVA-1.5 的多个视觉语言 benchmark 上，PruneSID 在不同 token 保留量下均超越 [[VisionZip]]、[[FastV]] 等基线，尤其在极低 token 保留率（64 tokens，11.1%）时优势最显著（+1.9 pp）。

---

### Figure 3: 方法框架概览

![[PruneSID_fig4.png|600]]

**说明**: PSCA 通过低秩 PCA 分解将视觉 token 聚类为语义一致的组；组内 NMS 用自适应阈值 $\tau$ 去除冗余，保留最具代表性的 token。

---

### Figure 4: ViT 层特征消融（PSCA 分组）

![[PruneSID_fig5.png|600]]

**说明**: 消融不同 [[Vision Transformer|ViT]] 层特征用于 PSCA 分组的效果，中后期层（第 22 层）表现最优，说明深层特征具有更好的语义抽象能力。

---

### Figure 5: 信息分数分布直方图

![[PruneSID_fig6.png|600]]

**说明**: MMMU 和 GQA 两个 benchmark 的图像信息分数分布直方图。MMMU 包含更多语义丰富的图像（如图表、数学题），信息分数分布整体偏高；GQA 场景图像更均匀。

---

### Figure 6: 方法局限性可视化

![[PruneSID_fig7.png|600]]

**说明**: 在需要细粒度局部细节的场景下，PruneSID 可能因区域内冗余判断而误删关键 token，导致 VLM 回答不完整。这是任务无关（task-agnostic）压缩的固有局限。

---

### Figure 7: 推理效率对比（LLaVA-NeXT）

![[PruneSID_fig8.png|600]]

**说明**: 在 POPE 数据集上，PruneSID（160 tokens）与 [[VisionZip]] 效率相当，但 POPE F1 从 86.6% 提升到 89.0%；相比原始模型（2880 tokens），预填充速度提升 **7.8×**。

---

### Table 1: LLaVA-1.5 主实验结果

**设定 A: 192 tokens（66.7% 压缩率）**

| 方法 | GQA | MMBench | MME | POPE | ScienceQA | VQAv2 | TextVQA | MMMU | SEED | VizWiz | LLaVA-B | 平均 |
|------|-----|---------|-----|------|-----------|-------|---------|------|------|--------|---------|------|
| Vanilla (576 tokens) | 61.9 | 64.7 | 1862 | 85.9 | 69.5 | 78.5 | 58.2 | 36.3 | 60.5 | 54.3 | 66.8 | 100% |
| PruneSID | 60.1 | 63.7 | 1791 | 86.9 | 68.5 | 76.8 | 56.7 | 36.1 | 59.0 | 55.4 | 65.1 | 98.5% |
| PruneSID-Dyn | 60.2 | 63.8 | 1797 | 87.1 | 69.1 | 76.8 | 56.9 | 36.8 | 59.0 | 55.5 | 65.1 | **98.6%** |

**设定 B: 64 tokens（88.9% 压缩率）**

| 方法 | GQA | MME | POPE | ScienceQA | 平均 |
|------|-----|-----|------|-----------|------|
| VisionZip (CVPR'25) | 55.1 | 1690 | 77.0 | 69.0 | 94.0% |
| PruneSID-Dyn | **57.2** | **1734** | **84.1** | 68.1 | **96.3%** |

**关键发现**: 极低 token 保留率（11.1%）下，PruneSID 超越 VisionZip 1.9 pp，说明语义分组 + NMS 在强压缩下优势更明显。

---

### Table 2: LLaVA-NeXT 主实验结果（2880 tokens 基础）

| 设定 | Token 数 | VisionZip | PruneSID | 提升 |
|------|---------|-----------|----------|------|
| 22.2% 保留 | 640 | 97.5% | **98.4%** | +0.9% |
| 11.1% 保留 | 320 | 94.3% | **95.8%** | +1.5% |
| 5.6% 保留 | 160 | 90.3% | **92.8%** | +2.5% |

**关键发现**: 压缩率越高，PruneSID 的优势越大（最高 +2.5 pp）。

---

### Table 3: Mini-Gemini 泛化实验

| 方法 | GQA | MMBench | MME | POPE | ScienceQA | VQAv2 | MMMU | SEED-I | 平均 |
|------|-----|---------|-----|------|-----------|-------|------|--------|------|
| Vanilla (576 tokens) | 62.4 | 69.3 | 1841 | 85.8 | 70.7 | 80.4 | 36.1 | 69.7 | 100% |
| PruneSID (64 tokens) | 58.3 | 63.1 | 1735 | 76.0 | 70.6 | 75.2 | 37.2 | 63.6 | 94.4% |

---

### Table 4: Video-LLaVA 视频理解结果

| 方法 | TGIF | MSVD | MSRVTT | ActivityNet | 平均 |
|------|------|------|--------|-------------|------|
| Video-LLaVA (原始) | 47.1 | 70.7 | 59.2 | 43.1 | 100% |
| VisionZip | 42.4 | 63.5 | 52.1 | 43.0 | 93.2% |
| PruneSID (6.6% 保留) | **45.8** | **67.1** | **53.3** | **43.1** | **95.5%** |

**关键发现**: 仅保留 6.6% token，视频理解精度达到原始的 95.5%，超越 VisionZip 2.3 pp，证明跨模态泛化能力。

---

### Table 5: 消融实验 — Token 分组策略

| 方法 | 64 tokens（GQA/MME/POPE/SQA） | 平均精度 |
|------|-------------------------------|---------|
| Random Grouping | 56.2 / 1707 / 79.6 / 67.5 | 94.8% |
| KMeans Grouping | 56.5 / 1630 / 82.8 / 67.8 | 95.6% |
| PruneSID (PSCA) | **57.1 / 1733 / 83.8 / 67.8** | **96.8%** |

**关键发现**: PSCA 优于随机分组（+2.0 pp）和 KMeans（+1.2 pp），验证了在 token 维度做 PCA 的语义感知优势。

---

### Table 6: 消融实验 — 组数 K 的影响

| K 值 | 保留 64 | 保留 128 | 保留 192 |
|-----|---------|---------|---------|
| K=8 | 93.1% | 96.0% | 96.2% |
| K=16 | 95.1% | 96.1% | 96.9% |
| K=32 | 94.7% | 97.0% | 97.9% |
| K=48 | 94.1% | 96.3% | **98.3%** |

**关键发现**: 本文采用 $K = \lfloor N/4 \rfloor$ 的自适应设置，平衡计算开销与语义覆盖度。

---

### Table 7: 效率分析（LLaVA-NeXT 7B，POPE 数据集）

| 方法 | Token 数 | 推理时间 | 预填充时间 | POPE F1 |
|------|---------|---------|---------|---------|
| Vanilla | 2880 | 254ms | 218ms | 86.4 |
| VisionZip (160 tokens) | 160 | 84ms | 27.8ms | 86.6% |
| PruneSID (160 tokens) | 160 | **89ms** | **27.8ms** | **89.0%** |

**关键发现**: 预填充速度提升 **7.8×**（218ms → 27.8ms），与 VisionZip 效率持平但精度更高（89.0% vs 86.6%）。

---

## 实验

### 评测 Benchmark

| 模态 | Benchmark | 说明 |
|------|-----------|------|
| 图像 | GQA, MMBench, MME, POPE, ScienceQA, VQAv2, TextVQA, MMMU, SEED-Bench, VizWiz, LLaVA-Bench | 多维度图像理解 |
| 视频 | TGIF, MSVD, MSRVTT, ActivityNet | 视频问答 |

### 实现细节

- **ViT 层选择**: 第 22 层特征用于 PSCA 分组（中后期层最优）
- **组数**: $K = \lfloor N/4 \rfloor$（自适应，随 token 预算变化）
- **NMS 缩放因子**: $\lambda = N/32$
- **特征归一化**: Sigmoid 激活 + L2 归一化
- **token 名额分配**: 按组大小比例分配（$\lfloor |G_k| / \sum_j |G_j| \cdot N \rfloor$）
- **方法属性**: Training-free，无需额外训练，即插即用

### 可视化结果

在低 token 保留率下，PruneSID 能保留语义丰富区域（前景物体、文字）的同时去除背景冗余 token。局限性出现在细粒度文字识别等场景，此时关键局部细节可能因区域内相似度过高而被错误剪枝。

---

## 批判性思考

### 优点

1. **Training-free，部署成本低**: 不需要微调任何模型参数，可以即插即用地集成到任何现有 VLM 中。
2. **理论有支撑**: 用[[容斥原理]]从信息论角度推导了 PSCA+NMS 的联合优化目标，方法设计有理论依据。
3. **跨模态泛化**: 在图像（LLaVA-1.5、LLaVA-NeXT、Mini-Gemini）和视频（Video-LLaVA）上均有效，证明方法的普适性。
4. **极端压缩下优势突出**: 在 5.6%–11.1% token 保留率下超越 VisionZip 最多 2.5 pp。

### 局限性

1. **Task-agnostic 压缩**: 不考虑具体任务或问题指令，在需要细粒度局部细节（如文字识别、计数）的场景可能失效。
2. **全局相似度计算开销**: 计算全局冗余分数 $\rho$ 需要 $O(T^2)$ 的 token 对相似度计算，token 数量极大时开销显著。
3. **动态压缩需要额外超参数**: 动态 token 分配策略引入了新的超参数，实际部署中需要调参。

### 潜在改进方向

1. **指令感知压缩**: 将用户问题或任务类型纳入 token 重要性评估，在细粒度任务上针对性保留关键区域。
2. **端到端联合训练**: 在 training-free 基础上引入轻量 adapter 微调，进一步提升压缩-精度权衡。
3. **近似 $O(T^2)$ 优化**: 用局部采样或分块计算替代全局冗余分数的精确计算，提升大分辨率图像效率。

### 可复现性评估

- [x] 代码开源（GitHub: ZhengyaoFang/PruneSID）
- [ ] 预训练模型（无需，training-free 方法）
- [x] 训练细节完整（超参数、层选择均有消融实验）
- [x] 数据集可获取（所有 benchmark 均为公开数据集）

---

## 关联笔记

### 基于

- [[VisionZip]]: 最直接的对比基线，同样是 LLM prefilling 前的 token 压缩
- [[FastV]]: 注意力引导 token 剪枝的代表方法
- [[Vision Transformer]]: 视觉编码器基础架构，PSCA 在 ViT 特征上操作

### 对比

- [[VisionZip]]: 注意力引导方法，PruneSID 在强压缩下显著超越
- [[DART]]: 去重复性方法，PruneSID 的 NMS 思路与之相关但在语义组内操作
- [[DivPrune]]: 多样性感知剪枝，PruneSID 与之目标相似但方法不同
- [[ToMe]]: Token Merging，合并而非剪枝，不同范式

### 方法相关

- [[主成分分析]]: PSCA 的核心技术，但 PruneSID 做了 token 维度的非标准 PCA
- [[非极大值抑制]]: NMS 的核心操作，从目标检测借鉴到 token 压缩
- [[余弦相似度]]: 用于 NMS 的 token 相似度度量

### 数据集相关

- [[POPE]]: 物体幻觉评测 benchmark，效率分析的主要测试集
- [[MMMU]]: 多学科多模态理解 benchmark，信息分数分布分析的对象

---

## 速查卡片

> [!summary] PruneSID (ICLR 2026)
> - **核心**: Training-free 视觉 token 压缩，同时优化语义重要性和信息多样性
> - **方法**: PSCA（token 维度 PCA 分组）+ Intra-Group NMS（组内自适应去冗余）+ 动态压缩率
> - **结果**: 11.1% token 保留 → 96.3% 精度；预填充速度 7.8×
> - **代码**: [github.com/ZhengyaoFang/PruneSID](https://github.com/ZhengyaoFang/PruneSID)

---

*笔记创建时间: 2026-06-05*
