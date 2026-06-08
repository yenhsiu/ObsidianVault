---
title: "VEQ: Modality-Adaptive Quantization for MoE Vision-Language Models"
method_name: "VEQ"
authors: [Guangshuo Qin, Zhiteng Li, Zheng Chen, Weihang Zhang, Linghe Kong, Yulun Zhang]
year: 2026
venue: ICML
tags: [quantization, mixture-of-experts, vision-language-model, post-training-quantization, model-compression, multimodal]
zotero_collection: N/A
image_source: online
arxiv_html: https://arxiv.org/html/2602.01037v1
created: 2026-06-07
---

# 论文笔记：VEQ: Modality-Adaptive Quantization for MoE Vision-Language Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Shanghai Jiao Tong University |
| 日期 | February 2026 |
| 项目主页 | N/A |
| 对比基线 | [[MBQ]], [[AWQ]], [[GPTQ]] |
| 链接 | [arXiv](https://arxiv.org/abs/2602.01037) / [Code](https://github.com/guangshuoqin/VEQ) |

---

## 一句话总结

> VEQ 是首个同时考虑多模态异质性与 MoE 专家重要性差异的 VLM 后训练量化框架，在 3-bit 配置下分别为 Kimi-VL 和 Qwen3-VL 带来 2.04% 和 3.09% 的平均精度提升。

---

## 核心贡献

1. **首个 MoE VLM 量化框架**: 首次同时解决多模态异质性与 MoE 结构特性，填补了领域空白
2. **VEQ-ME（模态专家感知量化）**: 依据视觉/文本 token 的路由频率为专家赋予重要性分数，在重建损失中优先降低核心专家的误差
3. **VEQ-MA（模态亲和感知量化）**: 融合路由器亲和度与模态敏感性来增强 Hessian 矩阵计算，更准确地估计量化误差

---

## 问题背景

### 要解决的问题

[[混合专家模型|MoE（Mixture-of-Experts）]] 架构的 [[视觉语言模型|VLM（Vision-Language Model）]] 参数量庞大，部署成本极高。[[训练后量化|Post-Training Quantization（PTQ）]] 可在不重新训练的前提下压缩模型，但现有 PTQ 方法针对普通 LLM 设计，无法同时处理 MoE VLM 的两大结构挑战。

### 现有方法的局限

- **VLM 量化方法**（[[MBQ]]、Q-VLM 等）：关注视觉与文本 token 分布差异，但完全忽视 MoE 的专家稀疏结构
- **MoE LLM 量化方法**（MoEQuant、MxMoE 等）：处理专家稀疏性，但未考虑多模态输入的跨模态异质性
- **通用 PTQ**（[[AWQ]]、[[GPTQ]]）：将模型视为整体，忽视 MoE 和模态的双重异质性

### 本文的动机

作者通过实验观察到两类关键异质性：

1. **跨模态差异**：文本 token 的平均梯度幅度是视觉 token 的 **22.4×**，这意味着两类 token 对量化误差的敏感度截然不同
2. **专家异质性**：并非所有专家均等激活，少数"热门专家"以高路由置信度主导输出，而不同专家对视觉/文本 token 具有模态特异性

---

## 方法详解

### 模型架构

VEQ 是一个 [[训练后量化|后训练量化]] 框架，应用于 [[混合专家模型|MoE]] 型 [[视觉语言模型|VLM]] 的 FFN 专家层。VEQ 不修改模型结构，仅优化量化参数。框架包含两个独立但互补的组件：

- **输入**: 校准数据集（视觉 + 文本 token）
- **核心模块 1**: [[VEQ-ME]]：基于专家激活频率的重要性加权
- **核心模块 2**: [[VEQ-MA]]：基于路由亲和度的增强 [[Hessian矩阵|Hessian]] 计算
- **输出**: 量化后的模型权重（W3A16 或 W4A16 配置）

### 核心模块

#### 模块1: VEQ-ME（Modality-Expert-Aware Quantization）

**设计动机**: 利用 [[专家路由|Expert Routing]] 的激活统计，识别对视觉/文本 token 分别最重要的专家，在量化重建损失中给予更高权重。

**具体实现**:
- 统计校准集中每个专家被文本 token 和视觉 token 路由的次数（$N_i^{\text{text}}$，$N_i^{\text{vis}}$）
- 计算梯度敏感因子 $\gamma$（文本与视觉梯度幅度之比）和数量归一化因子 $\beta$（文本与视觉 token 数之比）
- 合并为专家重要性分数 $S_i$，对重建损失加权优化

#### 模块2: VEQ-MA（Modality-Affinity-Aware Quantization）

**设计动机**: 利用 [[路由器亲和度|Router Affinity]]（路由概率 $p_j$）和模态标识符 $\alpha_j$，增强 [[Hessian矩阵|Hessian 矩阵]] 对不同 token 重要性的刻画。

**具体实现**:
- 标准 Hessian 矩阵将所有 token 等权计算 $H = XX^{\top}$
- VEQ-MA 为每个 token $j$ 赋予重要性权重 $c_j = p_j \cdot \alpha_j$，其中 $\alpha_j = \gamma$（文本 token）或 $1$（视觉 token）
- 构造增强 Hessian $\tilde{H} = XCX^{\top}$，用于 GPTQ 风格的逐层量化

---

## 关键公式

### 公式1: [[VEQ-ME|专家重要性分数]]

$$
S_i = \gamma \cdot N_i^{\text{text}} + \beta \cdot N_i^{\text{vis}}
$$

**含义**: 综合梯度敏感性和 token 数量差异，为第 $i$ 个专家计算重要性分数。

**符号说明**:
- $S_i$: 第 $i$ 个专家的重要性分数
- $N_i^{\text{text}}$: 校准集中第 $i$ 个专家被文本 token 路由的次数
- $N_i^{\text{vis}}$: 校准集中第 $i$ 个专家被视觉 token 路由的次数
- $\gamma = \|\nabla_{\text{text}}\| / \|\nabla_{\text{vis}}\|$: 梯度敏感因子（文本梯度幅度 ÷ 视觉梯度幅度，≈ 22.4）
- $\beta = T_{\text{text}} / T_{\text{vis}}$: 数量归一化因子（文本 token 总数 ÷ 视觉 token 总数）

---

### 公式2: [[训练后量化|标准量化重建损失]]

$$
\mathcal{L}_{\text{Standard}} = \sum_{i=1}^{M} \|\mathbf{W}_i \mathbf{X}_i - \hat{\mathbf{W}}_i \mathbf{X}_i\|_F^2
$$

**含义**: 对所有 $M$ 个专家，最小化量化前后激活输出的 Frobenius 范数差，作为量化优化目标。

**符号说明**:
- $M$: MoE 层中专家总数
- $\mathbf{W}_i$: 第 $i$ 个专家的原始权重矩阵
- $\hat{\mathbf{W}}_i$: 第 $i$ 个专家的量化权重矩阵
- $\mathbf{X}_i$: 第 $i$ 个专家接收的输入激活
- $\|\cdot\|_F$: [[Frobenius范数|Frobenius 范数]]

---

### 公式3: [[VEQ-ME|加权量化重建损失]]

$$
\mathcal{L}_{\text{Weighted}} = \sum_{i=1}^{M} S_i \cdot \|\mathbf{W}_i \mathbf{X}_i - \hat{\mathbf{W}}_i \mathbf{X}_i\|_F^2
$$

**含义**: 将专家重要性分数 $S_i$ 作为权重引入量化损失，使优化优先降低高重要性专家的量化误差。

**符号说明**:
- $S_i$: 第 $i$ 个专家的重要性分数（由公式1计算）
- 其余符号同公式2

---

### 公式4: [[VEQ-MA|增强 Hessian 矩阵]]

$$
\tilde{H} = X \mathbf{C} X^{\top}
$$

其中 $\mathbf{C} = \text{diag}(c_1, c_2, \ldots, c_T)$ 为对角权重矩阵。

**含义**: 通过对 token 赋予不同权重，使 Hessian 矩阵更准确地反映各 token 对量化误差的实际影响。

**符号说明**:
- $X \in \mathbb{R}^{d \times T}$: 输入激活矩阵，$T$ 为 token 数
- $\mathbf{C}$: 对角权重矩阵，对角元素为各 token 的重要性权重 $c_j$
- $\tilde{H}$: 增强 Hessian 矩阵，用于替换标准 GPTQ 中的 $XX^{\top}$

---

### 公式5: [[路由器亲和度|Token 重要性权重]]

$$
c_j = p_j \cdot \alpha_j, \quad \alpha_j = \begin{cases} \gamma & \text{if } x_j \text{ is text token} \\ 1 & \text{if } x_j \text{ is vision token} \end{cases}
$$

**含义**: 将路由器对该 token 的置信度（$p_j$）与模态敏感因子（$\alpha_j$）相乘，得到每个 token 的综合重要性权重。

**符号说明**:
- $p_j$: 路由器对 token $j$ 的激活置信度（softmax 概率值）
- $\alpha_j$: 模态敏感因子，文本 token 取 $\gamma$（≈ 22.4），视觉 token 取 1
- $\gamma$: 与公式1中定义一致

---

## 关键图表

### Figure 1: 3-bit 量化下的 zero-shot 性能对比

![Figure 1](https://arxiv.org/html/2602.01037v1/x1.png)

**说明**: Kimi-VL-Instruct 在 W3A16 配置下的 zero-shot 性能。VEQ-ME 和 VEQ-MA 在所有 benchmark 上均优于 RTN、AWQ、MBQ、GPTQ 基线，展示了 VEQ 在极低 bit 宽下的鲁棒性。

---

### Figure 2: 跨模态激活特征对比

| 文本 Token 激活分布 | 视觉 Token 激活分布 |
|---|---|
| ![Figure 2a](https://arxiv.org/html/2602.01037v1/x2.png) | ![Figure 2b](https://arxiv.org/html/2602.01037v1/x3.png) |

**说明**: 文本 token（左）和视觉 token（右）的激活频率分布对比。文本 token 呈现更尖锐的高频激活峰，而视觉 token 分布更平坦，定性展示了跨模态异质性。

---

### Figure 3: VEQ 框架总览

![Figure 3](https://arxiv.org/html/2602.01037v1/x4.png)

**说明**: VEQ 框架的整体结构。（1）VEQ-ME 根据路由统计为每个专家动态赋予重要性分数，优先降低核心专家的重建损失；（2）VEQ-MA 整合 token-专家亲和度与模态敏感性，构建增强 Hessian 矩阵，用于更精确的量化参数优化。

---

### Figure 4: 梯度幅度分析（128 COCO 样本）

![Figure 4](https://arxiv.org/html/2602.01037v1/x5.png)

**说明**: 在 128 个 COCO 数据集样本上统计的文本与视觉 token 梯度幅度。文本 token 的梯度范数显著高于视觉 token，平均比值高达 **22.4**，为 VEQ 中梯度敏感因子 $\gamma$ 的设计提供了实验依据。

---

### Figure 5: 单样本梯度详细分析（Sample 88）

![Figure 5](https://arxiv.org/html/2602.01037v1/x6.png)

**说明**: 代表性样本（Sample 88）的梯度逐 token 可视化。文本与视觉梯度比约达 15，证实了文本信息在量化误差中的主导性地位。

---

### Figure 6: 专家亲和度模式可视化

| Vision Act (Set A) | Vision Act (Set B) | Text Act (Set A) | Text Act (Set B) |
|---|---|---|---|
| ![Figure 6a](https://arxiv.org/html/2602.01037v1/x7.png) | ![Figure 6b](https://arxiv.org/html/2602.01037v1/x8.png) | ![Figure 6c](https://arxiv.org/html/2602.01037v1/x9.png) | ![Figure 6d](https://arxiv.org/html/2602.01037v1/x10.png) |

**Layer 13 专家激活分布**:

![Figure 6e](https://arxiv.org/html/2602.01037v1/x11.png)

**说明**: (a)-(d) 展示了视觉和文本 token 在不同输入样本下的专家激活分布，体现了专家激活的稀疏性与模态特异性——视觉 token 倾向于路由到特定专家集合，而文本 token 路由到另一批专家。(e) 展示 Kimi-VL-Instruct 第 13 层的专家激活特性，少数"热门专家"以高置信度主导输出。

---

### Figure 7: 超参数敏感性分析

| VEQ-ME 参数敏感性 | VEQ-MA 参数敏感性 |
|---|---|
| ![Figure 7a](https://arxiv.org/html/2602.01037v1/x12.png) | ![Figure 7b](https://arxiv.org/html/2602.01037v1/x13.png) |

**说明**：
- **VEQ-ME（左）**：保持 $\gamma$ 与 $\beta$ 的相对比例不变时，方法对绝对数值的尺度变化不敏感（尺度不变性），说明相对比例是量化误差最小化的关键
- **VEQ-MA（右）**：减小 $\lambda$（路由置信度权重）会导致 PPL 上升，$\lambda=1$ 时达到最低 PPL 2.2526，验证了路由置信度对精准量化的重要性

---

### Table 1: 主实验结果（W3 & W4 零样本精度）

#### Kimi-VL-Instruct

| 方法 | 配置 | MMMU | AI2D | InfoVQA | TextVQA | RealWorldQA | ScienceQA | VizWiz | MMBench | MME-RW | **Avg.** |
|------|------|------|------|---------|---------|-------------|-----------|--------|---------|--------|------|
| BF16 | — | 51.11 | 83.65 | 83.38 | 85.93 | 66.41 | 92.10 | 69.00 | 82.73 | 58.22 | 74.73 |
| RTN | W3 | 37.00 | 69.14 | 59.68 | 74.58 | 56.47 | 79.06 | 58.51 | 66.06 | 46.13 | 60.74 |
| AWQ | W3 | 41.78 | 70.05 | 61.49 | 74.04 | 58.17 | 81.84 | 61.37 | 71.13 | 50.44 | 63.37 |
| MBQ | W3 | 39.76 | 70.89 | 58.86 | 74.35 | 58.74 | 81.82 | 60.87 | 71.13 | 49.91 | 62.93 |
| GPTQ | W3 | 40.33 | 75.49 | 64.14 | 64.30 | 58.82 | 83.87 | 55.61 | 73.02 | 45.41 | 62.33 |
| VEQ-ME | W3 | 44.56 | 71.57 | 62.33 | 73.36 | 58.95 | 82.93 | 58.90 | 71.91 | 51.46 | 64.00 |
| **VEQ-MA** | **W3** | 42.56 | **75.65** | **64.48** | **78.30** | 58.30 | **84.85** | **63.10** | **73.42** | 48.03 | **65.41** |
| RTN | W4 | 48.44 | 82.38 | 80.16 | 84.60 | 66.80 | 90.83 | 69.17 | 80.67 | 52.48 | 72.84 |
| AWQ | W4 | 49.00 | 81.74 | 79.01 | 83.37 | 66.27 | 90.92 | 69.07 | 80.58 | 53.70 | 72.63 |
| MBQ | W4 | 48.89 | 81.70 | 78.49 | 83.18 | 64.58 | 90.29 | 69.03 | 80.93 | 55.08 | 72.46 |
| GPTQ | W4 | 50.00 | 82.32 | 80.04 | 84.63 | 64.84 | 91.39 | 68.86 | 81.44 | 53.14 | 72.96 |
| VEQ-ME | W4 | 49.22 | 81.41 | 79.75 | 83.69 | 66.36 | 91.06 | 69.17 | 80.84 | 54.20 | 72.86 |
| **VEQ-MA** | **W4** | **50.30** | **82.55** | **80.37** | **84.71** | **66.41** | **91.56** | 68.91 | **81.67** | **55.61** | **73.57** |

#### Qwen3-VL-30B-A3B-Instruct

| 方法 | 配置 | MMMU | AI2D | InfoVQA | TextVQA | RealWorldQA | ScienceQA | VizWiz | MMBench | MME-RW | **Avg.** |
|------|------|------|------|---------|---------|-------------|-----------|--------|---------|--------|------|
| BF16 | — | 73.67 | 86.27 | 81.43 | 81.31 | 65.49 | 93.52 | 83.27 | 85.91 | 59.92 | 78.98 |
| RTN | W3 | 57.33 | 77.10 | 44.96 | 68.64 | 43.01 | 89.08 | 64.07 | 78.60 | 45.73 | 63.17 |
| AWQ | W3 | 58.89 | 73.61 | 44.62 | 67.93 | 45.36 | 88.40 | 61.75 | 77.06 | 42.38 | 62.22 |
| MBQ | W3 | 50.56 | 71.44 | 40.15 | 64.23 | 53.82 | 87.31 | 59.48 | 76.46 | 45.27 | 60.97 |
| GPTQ | W3 | 64.89 | 74.11 | 47.27 | 71.82 | 54.25 | 88.23 | 63.39 | 69.39 | 44.08 | 64.16 |
| VEQ-ME | W3 | 60.56 | 73.38 | 44.95 | 69.91 | 53.73 | 88.40 | 62.47 | 77.75 | 44.24 | 63.93 |
| **VEQ-MA** | **W3** | **65.89** | **79.15** | 47.14 | **72.74** | **54.77** | **89.55** | **69.46** | **82.30** | 43.25 | **67.14** |
| RTN | W4 | 63.11 | 83.33 | 62.62 | 79.77 | 45.23 | 91.51 | 71.08 | 85.13 | 57.31 | 70.89 |
| AWQ | W4 | 65.20 | 81.22 | 58.32 | 77.40 | 63.14 | 91.06 | 69.73 | 85.22 | 57.31 | 72.07 |
| MBQ | W4 | 69.67 | 81.41 | 59.36 | 78.99 | 64.31 | 91.51 | 70.78 | 83.68 | 56.10 | 72.87 |
| GPTQ | W4 | 72.67 | 82.93 | 62.54 | 80.18 | 53.46 | 93.23 | 71.32 | 84.53 | 57.96 | 73.20 |
| VEQ-ME | W4 | 68.78 | 82.47 | 59.53 | 78.57 | 65.32 | 91.61 | 70.97 | 83.56 | 56.79 | 73.07 |
| **VEQ-MA** | **W4** | **71.56** | **82.95** | **62.86** | **81.10** | 61.70 | **92.64** | **72.42** | **85.88** | **58.12** | **74.36** |

**关键发现**: VEQ-MA 在 W3（最具挑战性的极低 bit 宽）下优势最为显著：Kimi-VL 提升 2.04%，Qwen3-VL 提升 3.09%。W4 下提升较小但依然一致（0.61% 和 1.16%）。

---

### Table 2: VEQ-ME 消融实验

| 配置 | MMMU | InfoVQA | ScienceQA | Avg. |
|------|------|---------|-----------|------|
| w/o γ（γ=1） | 43.44 | 60.39 | 82.24 | 62.02 |
| w/o β（β=1） | 43.89 | 60.34 | 82.67 | 62.30 |
| **VEQ-ME（完整）** | **44.56** | **62.33** | **82.93** | **63.27** |

**关键发现**: 梯度因子 $\gamma$ 和数量因子 $\beta$ 对文本密集型任务（InfoVQA）影响更大；移除任一因子均导致性能下降，两者缺一不可。

---

### Table 3: VEQ-MA 消融实验

| 配置 | MMMU | InfoVQA | ScienceQA | Avg. |
|------|------|---------|-----------|------|
| w/o p（p=1） | 42.33 | 64.30 | 83.35 | 63.63 |
| w/o α（α=1） | 41.33 | 63.72 | 82.34 | 62.46 |
| **VEQ-MA（完整）** | **42.56** | **64.48** | **84.85** | **63.96** |

**关键发现**: 模态指示符 $\alpha$（区分视觉/文本 token）对性能影响更大（移除后均值下降 1.5%）；路由置信度 $p$ 亦不可缺少，两者共同构成增强 Hessian 的完整信息来源。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| MMMU | — | 多学科多模态理解 | 测试 |
| AI2D | — | 图表理解 | 测试 |
| InfoVQA | — | 信息图表问答 | 测试 |
| TextVQA | — | 含文字图像问答 | 测试 |
| RealWorldQA | — | 真实世界理解 | 测试 |
| ScienceQA | — | 科学知识多模态 | 测试 |
| VizWiz | — | 视障辅助场景 | 测试 |
| MMBench | — | 多维度 VLM 评测 | 测试 |
| MME-RealWorld | — | 真实世界多模态 | 测试 |
| COCO | 128 samples | 通用视觉数据 | 校准集（用于统计梯度） |

### 实现细节

- **量化目标**: 权重量化（Weight-Only），激活保持 BF16（W3A16, W4A16）
- **校准集**: 128 个 COCO 数据集样本
- **基准模型**: Kimi-VL-Instruct（月之暗面）、Qwen3-VL-30B-A3B-Instruct（阿里）
- **对比方法**: RTN（朴素取整）、[[AWQ]]、[[GPTQ]]、[[MBQ]]
- **代码**: 即将开源于 GitHub

### 可视化结果

- Figure 4/5 直观展示了文本 vs. 视觉梯度差异（22.4×），为方法提供了强有力的动机
- Figure 6 揭示了 MoE 专家的模态特异性聚类现象，为专家差异化量化提供了依据
- Figure 7 的超参数分析证明了 VEQ-ME 的尺度不变性和 VEQ-MA 中路由置信度的重要性

---

## 批判性思考

### 优点

1. **问题定义清晰**: 首次系统性地将 MoE 专家异质性与多模态异质性同时纳入量化框架，问题抓得准
2. **方法轻量高效**: VEQ-ME 和 VEQ-MA 均以统计量（梯度幅度、路由频率）驱动，无需额外训练，额外计算开销极小
3. **实验覆盖广**: 9 个 benchmark、2 个大型模型（Kimi-VL 和 Qwen3-VL）、W3/W4 双配置，结果可信度高

### 局限性

1. **校准集依赖**: $\gamma$ 和 $\beta$ 需在校准集上统计梯度幅度，不同校准集可能导致不同的分数分布，存在一定不稳定性
2. **专家重要性静态估计**: 当前方案是基于校准集的离线统计，无法适应推理时动态输入的变化（不同查询下路由分布可能不同）
3. **仅评测两个模型**: 实验局限于 Kimi-VL 和 Qwen3-VL，能否推广到其他 MoE VLM（如 Mixtral-VL 等）仍未验证
4. **权重量化限制**: 仅处理权重量化（W3A16/W4A16），未涉及激活量化（W4A4 等），对推理吞吐量的实际提升有限

### 潜在改进方向

1. **动态专家重要性**: 结合在线路由统计，实现推理时自适应的专家重要性评估
2. **激活量化扩展**: 将跨模态差异意识扩展到激活量化（activation quantization），实现更激进的压缩
3. **跨模型验证**: 在更多 MoE VLM 架构上验证泛化性

### 可复现性评估

- [ ] 代码开源（承诺开源，尚未发布）
- [ ] 预训练模型（依赖公开 checkpoint）
- [x] 训练细节完整（PTQ 流程描述清晰）
- [x] 数据集可获取（校准集 COCO 公开，benchmark 均公开）

---

## 关联笔记

### 基于

- [[GPTQ]]: 逐层权重量化框架，VEQ-MA 的 Hessian 矩阵构建基于 GPTQ 框架
- [[AWQ]]: 激活感知权重量化，主要对比基线之一

### 对比

- [[MBQ]]: 当前 VLM 量化 SOTA，关注跨模态分布异质性但忽略 MoE 结构
- [[AWQ]]: 通用 PTQ 方法，将模型视为整体
- [[GPTQ]]: 通用逐层量化，无模态/专家感知能力

### 方法相关

- [[混合专家模型|MoE（Mixture-of-Experts）]]: VEQ 的应用目标架构
- [[训练后量化|Post-Training Quantization (PTQ)]]: VEQ 所属的量化类别
- [[Hessian矩阵|Hessian Matrix]]: VEQ-MA 增强 Hessian 构建的核心数学工具
- [[路由器亲和度|Router Affinity]]: VEQ-MA 的关键信号来源
- [[专家路由|Expert Routing]]: MoE 中决定 token 分配的机制

### 硬件/数据相关

- [[Kimi-VL-Instruct]]: 主要测试模型之一（月之暗面）
- [[Qwen3-VL]]: 主要测试模型之一（阿里）

---

## 速查卡片

> [!summary] VEQ: Modality-Adaptive Quantization for MoE VLMs
> - **核心**: 首个同时解决多模态异质性 + MoE 专家异质性的 VLM 后训练量化框架
> - **方法**: VEQ-ME（专家重要性加权损失）+ VEQ-MA（增强 Hessian 矩阵）
> - **结果**: W3A16 下 Kimi-VL +2.04%，Qwen3-VL +3.09%（vs. 最强基线）
> - **代码**: https://github.com/guangshuoqin/VEQ

---

*笔记创建时间: 2026-06-07*
