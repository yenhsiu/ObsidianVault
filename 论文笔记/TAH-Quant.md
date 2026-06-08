---
title: "TAH-Quant: Effective Activation Quantization in Pipeline Parallelism over Slow Network"
method_name: "TAH-Quant"
authors: [Guangxin He, Yuan Cao, Yutong He, Tianyi Bai, Kai Chen, Kun Yuan, Binhang Yuan]
year: 2025
venue: arXiv
tags: [activation-quantization, pipeline-parallelism, distributed-training, llm-training, quantization]
zotero_collection: _待整理
image_source: online
arxiv_html: https://arxiv.org/html/2506.01352v2
created: 2026-06-08
---

# 论文笔记：TAH-Quant: Effective Activation Quantization in Pipeline Parallelism over Slow Network

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 多所机构联合（含 Binhang Yuan 组） |
| 日期 | June 2025 |
| 项目主页 | — |
| 对比基线 | [[AQ-SGD]] |
| 链接 | [arXiv](https://arxiv.org/abs/2506.01352) / Code: — |

---

## 一句话总结

> TAH-Quant 将分块量化、熵引导自适应比特分配和 Hadamard 变换三者组合，把 pipeline 并行训练的 activation 压缩到 3–4 bits，在慢网络下实现最高 4.3× 吞吐量提升，同时保持与 FP16 基线统计等价的收敛性。

---

## 核心贡献

1. **Tile-Wise Group Quantization（分块量化）**: 沿 channel 维度将 activation 张量切成小 tile，每块独立维护 scale/zero-point，杜绝极值主导量化范围
2. **Entropy-Guided Adaptive Bit Allocation（熵引导自适应比特分配）**: 用 [[Shannon熵]] 评分动态给高熵 tile 分配 INT4、低熵 tile 分配 INT3，细化到 token 内 tile 粒度
3. **Hadamard-Based Outlier Suppression（Hadamard 离群值抑制）**: 检测 activation 中的离群点后局部施加 [[Hadamard Transform|Hadamard 变换]]，将极值重新分散到全部分量，再做量化

---

## 问题背景

### 要解决的问题

在跨机器 [[Pipeline Parallelism|流水线并行]] 训练大语言模型时，stage 之间需要传输 activation（前向）和 activation gradient（后向）。当跨地域、学术机构或个人之间合作时，网络带宽往往仅有 100 Mbps–1 Gbps，通信成为严重瓶颈。

### 现有方法的局限

- **AQ-SGD** 等方法需缓存历史 activation 做误差补偿（error compensation），在大数据集下存储开销大，且只能在多 epoch 训练中有效
- 纯低比特量化（INT4/INT3）在 activation 有离群值时误差大，收敛不稳定
- 先前 Hadamard 方案多用于权重量化，未针对 pipeline 通信场景下的 activation 设计

### 本文的动机

把 tile 粒度的分组量化、熵驱动的比特分配与 Hadamard 变换三者组合，可在无需误差补偿缓存的前提下，将 activation 压缩到 ~3.8 bits，并给出 O(1/√T) 的理论收敛保证。

---

## 方法详解

### 整体流程

**TAH-Quant** 作用在前向传播的 activation 通信阶段，将形状为 $B \times S \times C$ 的 activation 张量按 channel 分成大小为 $G$ 的 tile，依次经过三个模块处理后打包为 uint8 字节流跨机传输。

- **输入**: activation $a \in \mathbb{R}^{B \times S \times C}$，tile 大小 $G$
- **输出**: 3–4 bit 的压缩字节流
- **后向**: gradient 使用朴素定点 6–8 bit 压缩（计算效率优先，非算法关键）

### 核心模块

#### 模块 1：分块量化（Tile-Wise Group Quantization）

**设计动机**: 单一量化范围会被极值主导，tile 粒度让每个局部区域自适应动态范围。

**具体实现**:
- 将 $B \times S \times C$ 沿 channel 切成 $\lceil C/G \rceil$ 个 tile，每块独立计算 scale $s$ 和 zero-point $z$
- 支持 INT4（16 级）和 INT3（8 级）两种精度，由下一模块决定分配

#### 模块 2：熵引导自适应比特分配（Entropy-Guided Adaptive Bit Allocation）

**设计动机**: 不同 tile 的激活分布复杂度差异大——高熵 tile 需要更多比特保留信息，低熵 tile 用 INT3 已够用。

**具体实现**:
- 对每个 tile $\tilde{a}_{i,j'}$，计算归一化幅度分布 $p_k$（见公式 1）
- 计算 [[Shannon熵]] $\mathcal{H}(\tilde{a}_{i,j'})$（见公式 2）
- 对同一 token 内的 $\lceil C/G \rceil$ 个 tile 按熵值排序，top-$p$% 分配 INT4，其余分配 INT3（默认 $p=80$）

#### 模块 3：Hadamard 离群值抑制（Hadamard-Based Outlier Suppression）

**设计动机**: activation 中存在少量极大值（离群点），用比率启发式检测后局部做 [[Hadamard Transform|Hadamard 变换]]，把极值"摊平"。

**具体实现**:
- 对 tile 的最大值 $\alpha^{(1)}$ 和次大值 $\alpha^{(2)}$ 计算比率 $r$（见公式 3）
- 若 $r > \tau$（默认 $\tau=2.0$），将离群元素通过 pivot swapping 移到 index 0，再施加 Hadamard 变换 $\dot{\alpha} = \alpha P_d \cdot \frac{1}{\sqrt{G}} H_G$（见公式 4）
- 变换后极值被分散到所有分量，再做分块量化

---

## 关键公式

### 公式 1：[[Shannon熵|归一化幅度分布]]

$$
p_k = \frac{|\tilde{a}_{i,j',k}|}{\|\tilde{a}_{i,j'}\|_1 + \varepsilon}, \quad k = 1, \ldots, A
$$

**含义**: 将 tile 内每个元素的绝对值归一化为概率分布，用于后续熵计算

**符号说明**:
- $\tilde{a}_{i,j',k}$：第 $i$ 个 batch、第 $j'$ 个 tile、第 $k$ 个 channel 的 activation 值
- $A$：tile 大小 $G$
- $\varepsilon$：数值稳定项

### 公式 2：[[Shannon熵|逐 Tile 熵]]

$$
\mathcal{H}(\tilde{a}_{i,j'}) = \sum_{k=1}^{A} p_k \log(p_k + \varsigma)
$$

**含义**: 衡量该 tile 激活分布的复杂度，熵高则分配 INT4，熵低则分配 INT3

**符号说明**:
- $\varsigma$：对数稳定项（防止 $\log 0$）

### 公式 3：[[Hadamard Transform|离群值检测比率]]

$$
r = \frac{|\alpha^{(1)}|}{|\alpha^{(2)}| + \varrho}
$$

**含义**: 判断 tile 中最大值是否远超次大值，是则视为离群点

**符号说明**:
- $\alpha^{(1)}, \alpha^{(2)}$：tile 中绝对值最大和次大的元素
- $\varrho$：稳定项
- $\tau = 2.0$：判断阈值，$r > \tau$ 时触发 Hadamard 变换

### 公式 4：[[Hadamard Transform|带 Pivot 交换的 Hadamard 变换]]

$$
\dot{\alpha} = \alpha P_d \cdot \frac{1}{\sqrt{G}} H_G
$$

**含义**: 先用置换矩阵 $P_d$ 将离群元素移到 index 0，再做 Hadamard 变换，将极值均匀扩散

**符号说明**:
- $P_d$：pivot 置换矩阵，把离群元素移到首位
- $H_G$：$G \times G$ Hadamard 矩阵
- $\frac{1}{\sqrt{G}}$：归一化系数

### 公式 5–7：[[Stochastic Gradient Descent|随机优化目标与动量更新]]

$$
\min_{x \in \mathbb{R}^d} \mathbb{E}_{\xi \sim \mathcal{D}}[F(x;\xi)]
$$

$$
m^t = (1-\beta_1)m^{t-1} + \beta_1 \hat{g}^t
$$

$$
x^{t+1} = x^t - \eta m^t
$$

**含义**: 标准带动量的 [[Stochastic Gradient Descent|SGD]] 更新，$\hat{g}^t$ 为量化后的 gradient 估计

**符号说明**:
- $\beta_1$：动量衰减系数
- $\eta$：学习率
- $\hat{g}^t$：TAH-Quant 量化后的 gradient（前向用 TAH-Quant，后向用朴素定点压缩）

### 公式 8–10：[[Activation Quantization|量化误差约束（Assumption 4.4）]]

$$
\|\hat{g}^t - g^t\|^2 \leq (1-\delta)\|g^t\|^2
$$

$$
\|\mathbb{E}_{\xi^t}[\hat{g}^t] - \nabla f(x^t)\|^2 \leq (1-\delta)\|\nabla f(x^t)\|^2
$$

**含义**: 量化误差相对原 gradient 有界，$\delta$ 约为 0.6–0.9，实验验证成立

**符号说明**:
- $\delta$：量化精度系数，越大表示量化越精确
- $g^t$：未量化的 stochastic gradient

### 定理（Convergence Rate）

TAH-Quant 在 $T$ 步后满足：

$$
\frac{1}{T} \sum_{t=1}^{T} \mathbb{E}[\|\nabla f(x^t)\|^2] = \mathcal{O}\!\left(\frac{1}{\sqrt{T}}\right)
$$

**含义**: 收敛速率与标准 SGD 一致，量化不影响理论最优性

---

## 关键图表

### Figure 1：Assumption 4.4 的实验验证

![Figure 1a: step-wise tile=64](https://arxiv.org/html/2506.01352v2/x1.png)
![Figure 1b: step-wise tile=32](https://arxiv.org/html/2506.01352v2/x2.png)
![Figure 1c: full-dataset tile=64](https://arxiv.org/html/2506.01352v2/x3.png)
![Figure 1d: full-dataset tile=32](https://arxiv.org/html/2506.01352v2/x4.png)

**说明**: 训练过程中 per-step gradient 的相对量化误差始终低于 0.4，batch 平均后低于 0.1，实验验证 Assumption 4.4 成立（$\delta \approx 0.6$–$0.9$）

### Figure 2：训练收敛与端到端吞吐量

**收敛曲线（上排）**：
![Figure 2a: GPT2-XL WikiText2 fine-tuning](https://arxiv.org/html/2506.01352v2/x5.png)
![Figure 2b: GPT2-XL ArXiv21 fine-tuning](https://arxiv.org/html/2506.01352v2/x6.png)
![Figure 2c: Qwen2.5-3B code SFT](https://arxiv.org/html/2506.01352v2/x7.png)
![Figure 2d: Qwen2.5-3B open-domain SFT](https://arxiv.org/html/2506.01352v2/x8.png)

**吞吐量（下排）**：
![Figure 2e: 100Mbps task (i)](https://arxiv.org/html/2506.01352v2/x9.png)
![Figure 2f: 100Mbps task (ii)](https://arxiv.org/html/2506.01352v2/x10.png)
![Figure 2g: 500Mbps task (i)](https://arxiv.org/html/2506.01352v2/x11.png)
![Figure 2h: 500Mbps task (ii)](https://arxiv.org/html/2506.01352v2/x12.png)

**说明**: TAH-Quant 收敛曲线紧贴 FP16 基线；在 100 Mbps 下比 AQ-SGD 快 22%，比 FP32 快 4.3×

### Figure 3：消融研究

![Figure 3a: 自适应比特分配效果](https://arxiv.org/html/2506.01352v2/x13.png)
![Figure 3b: Hadamard 变换效果](https://arxiv.org/html/2506.01352v2/x14.png)

**说明**: 去掉自适应比特分配收敛慢 5–10%；去掉 Hadamard 变换对离群值场景影响显著

### Figure 4：LLaMA-8B 大规模预训练

![Figure 4a: LLaMA-8B 训练 loss](https://arxiv.org/html/2506.01352v2/x15.png)
![Figure 4b: LLaMA-8B 验证困惑度](https://arxiv.org/html/2506.01352v2/x16.png)

**说明**: 在 2.5B token 预训练中，TAH-Quant 训练 loss 和验证困惑度与 FP16 重合，验证可扩展性

### Table 1：LLaMA-3.2-1B 预训练（SlimPajama-6B）

| Metric | 1B tokens | 2B | 3B | 4B | 5B | 6B |
|--------|-----------|----|----|----|----|-----|
| Train Loss (FP16) | 3.778 | 3.326 | 3.149 | 3.032 | 2.954 | 2.893 |
| Train Loss (TAH-Quant) | 3.781 | 3.325 | 3.147 | 3.031 | 2.953 | 2.891 |
| Val PPL (FP16) | 39.18 | 28.54 | 25.36 | 23.03 | 21.42 | 20.67 |
| Val PPL (TAH-Quant) | 38.47 | 28.51 | 25.30 | 22.99 | 21.41 | 20.64 |

**关键发现**: 6B token 训练全程，TAH-Quant 验证困惑度与 FP16 统计等价，差值在噪声范围内

### Table 2：Qwen2.5-3B 下游任务评估

| Model | AVG | ARC | TruthfulQA | WinoGrande | HumanEval |
|-------|-----|-----|-----------|-----------|-----------|
| Qwen (no SFT) | 51.13 | 68.67 | 39.63 | 48.85 | 47.35 |
| SFT-FP16 | 59.08 | 69.38 | 66.46 | 50.49 | 50.00 |
| **SFT-TAH-Quant** | **59.32** | **70.00** | **67.68** | 49.61 | 49.91 |

**关键发现**: 经 TAH-Quant 压缩通信训练的模型在 4 项下游 benchmark 上与 FP16 SFT 模型无显著差异

### Table 3：GPT2-XL 吞吐量（tokens/s）

| 带宽 | FP32（无压缩）| AQ-SGD | TAH-Quant（4-bit 前向，6-bit 后向）| 相对 FP32 倍速 | 相对 AQ-SGD 倍速 |
|------|-------------|--------|----------------------------------|-------------|----------------|
| 1 Gbps | 2600 | 4749 | 5650 | 2.17× | 1.19× |
| 500 Mbps | 2482 | 4311 | 5749 | 2.32× | 1.33× |
| 300 Mbps | 1761 | 4369 | 5120 | 2.91× | 1.17× |
| 100 Mbps | 751 | 3310 | 4045 | 5.39× | 1.22× |

**关键发现**: 带宽越低收益越大；100 Mbps 下对 FP32 加速 5.4×，对 AQ-SGD 加速 1.22×

### Table 4：Tile 大小消融（Qwen2.5-3B）

| Tile Size G | MMLU | ARC |
|-------------|------|-----|
| 8 | 64.60 | 50.34 |
| **32（推荐）** | **64.88** | **49.91** |
| 128 | 64.34 | 49.66 |

**关键发现**: $G=32$ 在精度和元数据开销之间取得最优平衡；过大 tile 组内异质性高，精度下降

### Table 5：吞吐量 vs 微批次大小（500 Mbps，全局 batch=32）

| Micro-batch | 无压缩（tokens/s）| TAH-Quant（tokens/s）| 加速倍数 |
|-------------|-----------------|---------------------|---------|
| 1 | 2406 | 4289 | 1.78× |
| 2 | 1710 | 4016 | 2.35× |
| 4 | 1317 | 3023 | 2.30× |
| 8 | 981 | 2099 | 2.14× |

### Table 6：吞吐量 vs 微批次大小（1 Gbps，全局 batch=32）

| Micro-batch | 无压缩（tokens/s）| TAH-Quant（tokens/s）| 加速倍数 |
|-------------|-----------------|---------------------|---------|
| 1 | 3449 | 5177 | 1.50× |
| 2 | 2576 | 4655 | 1.81× |
| 4 | 2003 | 3475 | 1.73× |
| 8 | 1485 | 2436 | 1.64× |

### Table 7：离群值阈值 $\tau$ 敏感性（训练 loss）

| $\tau$ 设置 | Step 100 | Step 200 | Step 300 | Step 400 | Step 500 |
|------------|----------|----------|----------|----------|----------|
| 0（总是变换）| 2.97 | 2.79 | 2.63 | 2.60 | 2.60 |
| **2（默认）** | **2.72** | **2.63** | **2.59** | **2.56** | **2.55** |
| 4 | 2.73 | 2.65 | 2.61 | 2.58 | 2.57 |
| ∞（从不变换）| 2.82 | 2.72 | 2.68 | 2.64 | 2.63 |

**关键发现**: $\tau=2$ 最优；过激（$\tau=0$）破坏正常分布，过保守（$\tau=\infty$）无法抑制离群值

### Table 8：INT4/INT3 比例效果（训练 loss）

| 配置 | Step 50 | Step 100 | Step 200 | Step 300 | Step 500 |
|------|---------|----------|----------|----------|----------|
| 100% INT4 | 2.78 | 2.70 | 2.61 | 2.56 | 2.51 |
| **80% INT4 + 20% INT3（默认）** | — | — | — | — | — |
| 50% INT4 + 50% INT3 | 2.82 | 2.74 | 2.66 | 2.62 | 2.57 |
| 100% INT3 | 2.85 | 2.77 | 2.69 | 2.65 | 2.60 |

**关键发现**: 混合精度（80% INT4）在通信效率和数值稳定性间取得最佳平衡；纯 INT2 出现训练不稳定

---

## 算法流程

### Algorithm 1：TAH-Quant 在两级 Pipeline 中的应用

```
初始化：Machine a 参数 x^(a)，Machine b 参数 x^(b)，优化器 ρ

For t = 1 to T:
  采样批次 ξ_t

  [前向传播]
  Machine a 计算 activation，用 TAH-Quant 量化：Q_TAH(a(ξ_t, x_t^(a)))
  Machine a → Machine b 发送量化 activation 字节流
  Machine b 反量化，继续前向计算

  [后向传播]
  Machine b 计算 activation gradient，用朴素定点压缩：Q_Naive(∇_a(F∘b)|_ξt)
  Machine b → Machine a 发送压缩 gradient
  Machine a 反量化 gradient

  [参数更新]
  Machine a 用 ĝ^t 更新 x^(a)
  Machine b 用 ĝ^t 更新 x^(b)

输出：最终权重 x = (x_T^(a), x_T^(b))
```

**设计要点**: 前向用 TAH-Quant（~3.8 bits），后向用简单定点（6–8 bits）；对称设计保证梯度流的鲁棒性

---

## 实验

### 数据集

| 数据集 | 规模 | 用途 |
|--------|------|------|
| WikiText-2 | 标准 LM | GPT2-XL fine-tuning |
| ArXiv21 | 学术文本 | GPT2-XL fine-tuning |
| Open-domain SFT | — | Qwen2.5-3B instruction-tuning |
| Code SFT | — | Qwen2.5-3B instruction-tuning |
| SlimPajama-6B | 6B tokens | LLaMA-3.2-1B pretraining |
| Proof-Pile | 2.5B tokens | LLaMA-8B pretraining |

### 实现细节

- **模型**: GPT2-XL (1.5B)、Qwen2.5-3B、LLaMA-3.2-1B、LLaMA-8B
- **量化配置**: 默认 Tile size $G=32$，80% INT4 + 20% INT3，后向 6-bit 朴素定点，阈值 $\tau=2.0$
- **有效比特数**: ~4.41 bits/element（含元数据）
- **计算开销**: 约 +1% 额外训练步时间
- **网络仿真**: 使用 Linux `tc` 工具限速 100 Mbps / 300 Mbps / 500 Mbps / 1 Gbps
- **评测 benchmark**: ARC、TruthfulQA、WinoGrande、HumanEval

---

## 批判性思考

### 优点

1. **无需误差补偿缓存**：相比 AQ-SGD，不缓存历史 activation，内存开销小，支持单 epoch 训练
2. **完整收敛理论**：给出 O(1/√T) 收敛率证明，与 vanilla SGD 等价
3. **实现轻量**：Hadamard 变换 O(G²) 开销小，整体约 +1% 训练时间
4. **组合设计合理**：三个模块相互独立，消融清晰，各自贡献可量化

### 局限性

1. **仅验证两阶段 Pipeline**：Algorithm 1 只展示了 2-stage 分区，多阶段（>2 GPU）的实验缺失
2. **后向使用朴素定点**：后向 gradient 仍用 6–8 bit 定点，非本文创新，通信瓶颈在前向时有效，后向密集场景可能成为新瓶颈
3. **网络仿真非真实慢网络**：限速测试不等于真实跨地域网络（延迟、抖动、丢包未考虑）
4. **Tile 大小固定**：$G$ 在训练中不自适应，不同层的激活分布差异未被利用

### 潜在改进方向

1. **动态 tile 大小**：根据层类型（attention vs FFN）自适应调整 $G$
2. **多级 Pipeline 扩展**：验证 4/8 stage pipeline 下的收益与开销
3. **后向也用 TAH-Quant**：统一前后向压缩框架
4. **与 FSDP/ZeRO 结合**：在数据并行 + 流水线并行混合场景下评估

### 可复现性评估

- [ ] 代码开源（论文中未提供代码链接）
- [ ] 预训练模型（未提供）
- [x] 训练细节完整（超参数、数据集均有说明）
- [x] 数据集可获取（WikiText-2、SlimPajama 均公开）

---

## 关联笔记

### 基于

- [[AQ-SGD]]: 主要对比基线，基于误差补偿的 activation 量化方法
- [[Pipeline Parallelism|流水线并行]]: 本文的目标应用场景
- [[Stochastic Gradient Descent|SGD]]: 收敛分析的基础框架

### 对比

- [[AQ-SGD]]: 需要 activation 缓存，TAH-Quant 无此限制
- [[GPTQ]]: 权重量化方案，TAH-Quant 针对 activation 通信
- [[AWQ]]: 同样用于减少量化误差，但面向推理而非训练通信

### 方法相关

- [[Activation Quantization|激活量化]]: 核心技术方向
- [[Hadamard Transform|Hadamard 变换]]: 离群值抑制核心工具
- [[Shannon熵]]: 自适应比特分配的度量
- [[Group Quantization|分组量化]]: Tile-wise 量化的基础

---

## 速查卡片

> [!summary] TAH-Quant
> - **核心**: Pipeline 并行训练中的 activation 量化，解决慢网络通信瓶颈
> - **方法**: Tile 分组量化 + 熵引导 INT4/INT3 自适应分配 + Hadamard 离群值抑制
> - **结果**: 3–4 bits，慢网络下最高 4.3× 吞吐量提升，收敛等价 FP16
> - **代码**: 未开源

---

*笔记创建时间: 2026-06-08*
