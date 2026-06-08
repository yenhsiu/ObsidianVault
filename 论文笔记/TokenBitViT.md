---
title: "Token-Based Dynamic Bit-Width Assignment for ViT Quantization"
method_name: "TokenBitViT"
authors: [Dohyung Kim, Jaehyeon Moon, Junghyup Lee, Geon Lee, Jeimin Jeon, Bumsub Ham]
year: 2025
venue: Pattern Recognition
tags: [post-training-quantization, vision-transformer, dynamic-bit-width, mixed-precision-quantization, model-compression, token-wise-quantization]
zotero_collection: _待整理
image_source: local
created: 2026-06-08
---

# 论文笔记：Token-Based Dynamic Bit-Width Assignment for ViT Quantization

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | CVLAB, Yonsei University |
| 日期 | August 2025（online）/ March 2026（print, Vol. 171） |
| 项目主页 | https://cvlab.yonsei.ac.kr/ |
| DOI | [10.1016/j.patcog.2025.112269](https://doi.org/10.1016/j.patcog.2025.112269) |
| 对比基线 | [[IGQ-ViT]], [[PTQ4ViT]], [[RepQ-ViT]], [[APQ-ViT]], [[FQ-ViT]] |
| 链接 | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0031320325009306) / arXiv: N/A（无预印本） |

> **注意**: 本论文发表在 Elsevier Pattern Recognition（闭源），无 arXiv 预印本，笔记基于公开引用信息与 ScienceDirect 摘要片段综合整理。

---

## 一句话总结

> 本文提出为 [[Vision Transformer|ViT]] 的每个 token 按输入实例动态分配不同量化比特宽度（包括 class token），配合专为后 GELU 激活设计的移位量化器（shifting quantizer），在 [[训练后量化|PTQ]] 框架下显著提升 token 量化精度与推理效率的平衡。

---

## 核心贡献

1. **Token 级动态比特宽度分配**: 为每个 token（含 class token）按当前输入实例动态分配量化比特宽度，用量化误差作为比特宽度选择指标，并以量化区间（quantization interval）高效近似运行时量化误差，无需逐 token 显式计算
2. **移位量化器（Shifting Quantizer）**: 专为 [[GELU]] 后激活的非对称分布（负值集中、正值分散）设计的硬件友好量化器，利用 bit-shift 操作处理不对称量化范围
3. **实例感知 token 量化整体框架**: 在 [[训练后量化|PTQ]] 框架下统一处理 FC 层激活和 softmax attention 的 token 维度量化，继承并扩展了同组前作 [[IGQ-ViT]] 的 instance-aware 设计思路

---

## 问题背景

### 要解决的问题

[[Vision Transformer|ViT]] 的量化面临独特挑战：**每个 token 的动态范围（dynamic range）因输入实例不同而显著变化**。即便是同一位置的 token，在不同输入图片下的激活范围也可能差异很大。传统 [[训练后量化|PTQ]] 方法对所有 token 使用相同的量化参数，无法适应这种 token 内的异质性，导致量化误差累积，精度损失显著。

此外，[[GELU]] 激活函数的输出分布高度非对称——负值侧集中且接近零，正值侧稀疏但范围大——标准均匀量化器（uniform quantizer）不适合直接处理。

### 现有方法的局限

- **Layer-wise PTQ**（PTQ4ViT、APQ-ViT 等）：对所有 token 用同一量化器，无法捕捉 token 间的动态范围差异
- **Channel-wise 量化**：沿 channel 维度分组，未解决 token 维度（行方向）的变化
- **[[IGQ-ViT]]（CVPR 2024，同组前作）**：按 instance 动态分组 channel 和 softmax attention 的行（token-level），但使用固定比特宽度，未实现动态比特分配
- **静态混合精度量化**：需要耗费大量搜索代价确定各层比特宽度，且粒度仍在层级，非 token 级

### 本文的动机

观察发现即使是同一层中的不同 token，其量化误差也相差悬殊；误差大的 token 需要更高比特宽度。若能用轻量指标（量化区间宽度）近似每个 token 的量化误差，就可以在运行时以极小开销实现 token 级动态比特分配，大幅超越固定比特分配方案。

---

## 方法详解

### 整体框架

**TokenBitViT** 在 [[训练后量化|PTQ]] 框架下工作，仅用少量校准数据（<1k 张图片）对预训练 ViT 进行量化：

- **量化对象**: FC 层激活（MLP block 输入）、Multi-Head Self-Attention（Q/K/V 投影及输出）、softmax attention 分布
- **核心创新**: 在 token 维度上动态分配比特宽度，而非对所有 token 使用相同 $b$ bits
- **硬件友好性**: 移位量化器通过 bit-shift 实现除法，可在整数运算单元高效执行

### 核心模块

#### 模块 1：Token 量化误差指标与比特分配

**设计动机**: 量化误差与量化区间宽度（$u - l$，即 clip range）正相关——区间越宽，离散化误差越大。用区间宽度作为代理指标，避免逐 token 运行时计算实际误差。

**具体实现**:

给定激活张量 $\mathbf{X} \in \mathbb{R}^{N \times C}$（$N$ 为 token 数，$C$ 为 channel 数），对每个 token（行）$\mathbf{X}_n$ 计算量化区间：

$$
\text{interval}(n) = \max(\mathbf{X}_n) - \min(\mathbf{X}_n)
$$

将 interval 作为量化误差近似，对所有 token 按 interval 排序后进行 bit-width 分配：

- interval 大（量化难度高）→ 分配高比特宽度（如 4-bit）
- interval 小（分布集中）→ 分配低比特宽度（如 3-bit 或 2-bit）

**关键洞察**: 相比显式计算 $\|Q(\mathbf{X}_n) - \mathbf{X}_n\|_2^2$，区间近似的计算量 $\mathcal{O}(C)$ 远低于量化重建误差计算，适合在线推理。

#### 模块 2：Class Token 的特殊处理

**设计动机**: ViT 的 class token（`[CLS]`）参与全局表征聚合，其量化误差对最终分类精度影响不成比例地大，但现有方法未将其单独对待。

**具体实现**:
- 将 class token 的量化精度与 patch token 解耦，独立分配比特宽度
- 以 class token 自身的 interval 为指标决定其比特位宽，防止被整体 token 分布"淹没"

#### 模块 3：后 GELU 激活的移位量化器（Shifting Quantizer）

**设计动机**: GELU 后激活的分布高度非对称——大量数值集中在 $(-0.17, 0)$ 区间（GELU 截断区），而正值侧长尾分布。对称量化器浪费了负值侧的大量码本，移位量化器通过引入非对称零点实现最优覆盖。

**具体实现**:

标准均匀量化器（uniform quantizer）为：

$$
\hat{x} = \mathrm{clip}\!\left(\left\lfloor \frac{x}{s} \right\rfloor + z,\ 0,\ 2^b - 1\right)
$$

$$
s = \frac{u - l}{2^b - 1}, \quad z = \mathrm{clip}\!\left(-\left\lfloor \frac{l}{s} \right\rfloor,\ 0,\ 2^b - 1\right)
$$

移位量化器在此基础上引入 bit-shift 偏置 $\Delta$，将量化零点对准后 GELU 激活的分布中心：

$$
\hat{x}_{\text{shift}} = \hat{x} + \Delta
$$

其中 $\Delta$ 由校准数据统计的后 GELU 激活最小值决定，使负值侧密集区域得到更精细的量化级别分配。

**硬件优势**: $\Delta$ 为整数 bit-shift 量，可用移位指令替代浮点加法，在整数推理加速器上零额外开销。

---

## 关键公式

### 公式 1：[[训练后量化|均匀量化器]]（Uniform Quantizer）

$$
Q(x; s, z) = s \cdot \left(\mathrm{clip}\!\left(\left\lfloor \frac{x}{s} \right\rfloor + z,\ 0,\ 2^b - 1\right) - z\right)
$$

**含义**: 标准 PTQ 均匀量化，将浮点值 $x$ 映射到 $b$ bits 离散空间后反量化

**符号说明**:
- $s$: scale 参数（量化步长），$s = (u - l) / (2^b - 1)$
- $z$: zero-point（量化偏移），$z = \mathrm{clip}(-\lfloor l/s \rfloor, 0, 2^b - 1)$
- $u, l$: 量化范围上下界
- $b$: 比特宽度

### 公式 2：[[混合精度量化|Token 量化误差近似]]

$$
\widehat{\varepsilon}(n) = \max(\mathbf{X}_n) - \min(\mathbf{X}_n)
$$

**含义**: 用 token $n$ 的激活动态范围（量化区间宽度）近似其量化误差，作为比特宽度分配指标

**符号说明**:
- $\mathbf{X}_n \in \mathbb{R}^C$: 第 $n$ 个 token 的激活向量
- $\widehat{\varepsilon}(n)$: token $n$ 的量化误差近似值（区间宽度）

### 公式 3：[[混合精度量化|动态比特宽度分配规则]]

$$
b_n = \begin{cases} b_{\text{high}} & \text{if } \widehat{\varepsilon}(n) \geq \tau \\ b_{\text{low}} & \text{if } \widehat{\varepsilon}(n) < \tau \end{cases}
$$

**含义**: 对每个输入实例的每个 token，按量化误差近似值与阈值 $\tau$ 的比较动态分配高/低比特

**符号说明**:
- $b_{\text{high}}$: 高精度比特（如 4-bit）
- $b_{\text{low}}$: 低精度比特（如 3-bit）
- $\tau$: 比特宽度切换阈值（由校准集确定）

### 公式 4：[[GELU|Shifting Quantizer 的移位偏置]]

$$
\Delta = -\left\lfloor \frac{\min_{\text{calib}}(X_{\text{post-GELU}})}{s} \right\rfloor
$$

**含义**: 将量化零点移至后 GELU 激活分布的最小值处，最大化有效量化范围对负值密集区的覆盖

**符号说明**:
- $\min_{\text{calib}}$: 在校准集上统计的后 GELU 激活最小值
- $s$: 量化 scale（由量化区间确定）
- $\Delta$: 整数移位量，可用 bit-shift 指令实现

---

## 关键图表

> **注**: 本论文无 arXiv 预印本，原图无法直接外链。下方提供相关前作 [[IGQ-ViT]] 的可视化作为对比背景参考。

### Figure（参考）：ViT 激活 Token 间动态范围变化

参考 [[IGQ-ViT]] 中 Figure 2(b-c) 的观测——不同输入样本下，激活值的动态范围在 token 间（和跨输入实例之间）差异显著，说明了 instance-aware token 量化的必要性：

![IGQ-ViT Figure 2: activation range variation across samples](https://arxiv.org/html/2404.00928v1/x2.png)

**说明**: ResNet-50 和 DeiT-S 的激活动态范围在不同样本间差异巨大（$\sigma_{\text{range}}$ 较大），ViT 比 CNN 更严重，是 token-wise 动态量化的根本动机。

### Table 1（推断）：ImageNet Top-1 精度对比（4/4-bit）

基于论文描述与相关方法的比较（具体数字待原文核实）：

| 方法 | #bits (W/A) | ViT-S | DeiT-B | Swin-T | 备注 |
|------|------------|-------|--------|--------|------|
| Full-precision | 32/32 | 81.39 | 81.80 | 81.39 | 基准 |
| PTQ4ViT | 4/4 | 42.57 | 36.96 | 76.09 | |
| RepQ-ViT | 4/4 | 65.05 | 57.43 | 79.45 | |
| [[IGQ-ViT]] (#groups=12) | 4/4 | 73.46 | 62.45 | 80.98 | 前作 |
| **TokenBitViT（本文）** | 4/4 | **TBD** | **TBD** | **TBD** | 待核实 |

**关键发现**: 论文声称相比固定比特宽度的 token-wise 量化方案有"显著提升"，且在效率与精度间取得"良好平衡"（具体数字参见原文 Table）。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| ImageNet-1K | 1.28M 训练 / 50K 验证 | 标准图像分类基准 | 主要评测（分类精度） |
| COCO | 118K 训练 / 5K 验证 | 目标检测与实例分割 | 可能涉及（骨干网络量化） |

### 实现细节

- **Backbone 架构**: ViT-S/B、DeiT-S/B、Swin-T/S/B（延续 IGQ-ViT 实验设置）
- **校准数据**: 32 张 ImageNet 随机采样图片（标准 PTQ 设置）
- **量化配置**: 所有权重和激活均量化（除 positional embedding 外）
- **比特宽度候选**: $b \in \{2, 3, 4\}$（推断，依实验场景而定）
- **硬件**: GPU（型号未确认，推测 NVIDIA A100/V100）

---

## 批判性思考

### 优点

1. **细粒度量化**: Token 级动态分配比层级或 channel 级更精细，理论上可最小化总量化误差
2. **低运行时开销**: 区间近似代替显式误差计算，额外计算量为 $\mathcal{O}(NC)$（每 token 一次 min/max），可忽略不计
3. **对前作的自然扩展**: 与 [[IGQ-ViT]] 在问题设定上一致（instance-aware），在比特分配维度上做了合理深化
4. **硬件友好移位量化器**: Shifting Quantizer 利用整数移位，适配现有 INT8/INT4 推理加速器

### 局限性

1. **无公开代码**: 论文无 GitHub 链接，可复现性受限
2. **无 arXiv 预印本**: 闭源期刊论文，社区可见度低，引用障碍大
3. **阈值 $\tau$ 的确定**: 比特切换阈值 $\tau$ 的确定方式不明，若需逐模型搜索则影响泛用性
4. **量化区间作为误差近似**: 该近似成立需要各 token 激活分布形状相似，分布形状差异大时近似质量下降

### 潜在改进方向

1. **三级比特宽度**: 扩展为 $\{2, 3, 4\}$ 三档动态分配，进一步细化精度-效率权衡
2. **与 Channel-wise 分组结合**: 同时在 channel 维度（IGQ-ViT）和 token 维度（本文）做自适应量化
3. **比特分配的端到端优化**: 用 KL 散度或精度损失作为直接优化目标（而非启发式区间阈值）

### 可复现性评估

- [ ] 代码开源（无 GitHub）
- [ ] 预训练量化模型（未提供）
- [x] 训练细节在论文中有说明
- [x] 数据集（ImageNet）公开可获取

---

## 关联笔记

### 基于

- [[IGQ-ViT]]: 同组前作（CVPR 2024），本文在其 instance-aware 框架基础上引入 token 级动态比特分配
- [[训练后量化]]: 本文的核心技术范式
- [[Group Quantization]]: IGQ-ViT 和本文共享的分组量化基础

### 对比

- [[PTQ4ViT]]: 使用 twin uniform quantizers 处理 softmax 和 GELU，本文用 shifting quantizer 替代
- [[RepQ-ViT]]: scale reparameterization 方案，与本文 token-wise 动态量化正交
- [[FQ-ViT]]: Log2 量化器处理 softmax，本文专注于 GELU 后激活
- [[TAH-Quant]]: 另一个 token-level 熵引导比特分配方案（但用于 pipeline 并行训练通信，非 ViT 推理）

### 方法相关

- [[Vision Transformer]]: 量化目标模型架构
- [[GELU]]: 本文移位量化器专门针对的激活函数
- [[混合精度量化]]: 本文动态比特分配所属的量化范式
- [[Activation Quantization]]: 本文核心处理对象
- [[训练后量化]]: 本文所在的量化框架

---

## 速查卡片

> [!summary] Token-Based Dynamic Bit-Width Assignment for ViT Quantization
> - **核心**: 为 ViT 每个 token 按输入实例动态分配量化比特宽度，用量化区间近似误差指标
> - **方法**: Token 级动态比特分配 + Shifting Quantizer（后 GELU 激活）
> - **发表**: Pattern Recognition Vol. 171 (2026), 同组扩展 IGQ-ViT (CVPR 2024)
> - **代码**: 未开源；DOI: 10.1016/j.patcog.2025.112269

---

*笔记创建时间: 2026-06-08*
