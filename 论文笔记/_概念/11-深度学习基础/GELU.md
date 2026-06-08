---
type: concept
aliases: [GELU, Gaussian Error Linear Unit, 高斯误差线性单元]
---

# GELU（Gaussian Error Linear Unit）

## 定义

GELU 是一种非线性激活函数，通过将输入乘以其服从标准正态分布的累积分布函数（CDF）来实现软性非线性门控，在 Transformer 系列模型（BERT、GPT、ViT 等）的 MLP block 中被广泛采用。

## 数学形式

精确形式：

$$
\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right]
$$

常用近似形式（Hendrycks & Gimpel 2016）：

$$
\text{GELU}(x) \approx 0.5x\left(1 + \tanh\!\left[\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right]\right)
$$

## 核心要点

1. **软性门控**: 输出 $\text{GELU}(x) = x \cdot P(X \leq x)$，$X \sim \mathcal{N}(0,1)$，使接近 0 的输入被"软"抑制
2. **无硬截断**: 负值不被完全归零（不同于 ReLU），保留更多梯度流信息
3. **非对称输出分布**: 在 $x \approx -0.17$ 处有极值约 $-0.17$；负值侧密集、正值侧稀疏——量化时的核心挑战
4. **后 GELU 激活特性**: ViT MLP 中，GELU 之后的激活分布高度集中于 $(-0.17, 0)$，标准均匀量化器的码本分配效率低
5. **整数近似困难**: 精确 GELU 依赖 erf 函数，不适合整数推理；I-ViT 提出 ShiftGELU 用移位近似

## 与量化的关系

后 GELU 激活是 [[Vision Transformer|ViT]] 量化的主要难点之一：
- PTQ4ViT 引入 twin uniform quantizers 处理 GELU 输出的非对称分布
- I-ViT 提出 ShiftGELU 用整数 bit-shift 近似 GELU，支持全整数推理
- [[TokenBitViT]] 提出 Shifting Quantizer，用移位零点适配 GELU 后激活的非对称分布

## 代表工作

- Hendrycks & Gimpel (2016). *Gaussian Error Linear Units*: 提出 GELU 激活
- GPT-2 / BERT / ViT: 主流 Transformer 中的标准激活函数
- [[TokenBitViT]]: 专为后 GELU 激活设计移位量化器（Shifting Quantizer）

## 相关概念

- [[Vision Transformer]]: GELU 是其 MLP block 的激活函数
- [[Activation Quantization]]: 后 GELU 激活量化的主要挑战
- [[训练后量化]]: GELU 量化是 PTQ 方法需要处理的特殊问题
