# ActCIM-Robust

## 基于存算非线性随机扰动灵敏度感知训练的存算鲁棒性增强方法

> **Sensitivity-Guided Randomized Nonlinearity-Aware Training for Computing-in-Memory Neural Networks**

**2026 Inno CIM 存算一体高校挑战赛 参赛项目**

[![Python](https://img.shields.io/badge/Python-3.12.5-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu121-red)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green)](https://developer.nvidia.com/cuda-toolkit)
[![GPU](https://img.shields.io/badge/GPU-RTX%204060%208GB-orange)](https://www.nvidia.com)

---

## 项目简介 | Overview

存算一体（Computing-in-Memory, CIM）芯片在模拟计算过程中引入固有的非线性激活失真，导致部署于 CIM 芯片上的深度神经网络准确率严重下降。本项目系统分析了 CIM 三次多项式非线性 $f_\alpha(x) = m \cdot [\alpha \cdot (x/m)^3 + (1-\alpha) \cdot (x/m)]$ 对 ResNet-18 推理性能的影响，并提出和对比了三种非线性感知训练（NAT）方法。

CIM chips suffer from inherent nonlinear activation distortions during analog computation, severely degrading the accuracy of deployed DNNs. We systematically analyze the impact of a parameterized cubic CIM nonlinearity on ResNet-18 inference and propose three Nonlinearity-Aware Training (NAT) methods.

### 核心发现 | Key Results

| 方法 | α=0 准确率 | 最差准确率 (α=+0.8) | AURC | 不对称间隙 |
|------|-----------|---------------------|------|-----------|
| Clean (baseline) | **94.23%** | 81.25% | 0.9283 | -0.0314 |
| Random-NAT | 94.25% | 81.30% | 0.9281 | -0.0331 |
| SGR-NAT | **94.28%** | 82.06% | 0.9290 | -0.0300 |
| **Fixed-NAT** | 94.02% | **91.79%** | **0.9374** | **0.0007** |

> **Fixed-NAT 将最差情况准确率提升 10.54 个百分点**（从 81.25% → 91.79%），仅需 **5 分钟微调**，且几乎消除了正负方向不对称性。

---

## 快速开始 | Quick Start

### 环境要求 | Requirements

- **Python**: 3.12.5
- **PyTorch**: 2.5.1+cu121
- **GPU**: NVIDIA RTX 4060 8GB（或其他 CUDA 兼容 GPU）
- **OS**: Windows 11 / Linux
- **RAM**: 32GB

### 安装 | Installation

```powershell
pip install -r requirements.txt
pip install -e .
```

### 环境检查 | Environment Check

```powershell
python -m actcim_robust.cli check-env
```

### 运行测试 | Run Tests

```powershell
pytest tests/ -v
```

---

## 项目结构 | Project Structure

```
ActCIM-Robust/
├── src/actcim_robust/         # 核心代码
│   ├── cli.py                 # 命令行统一入口
│   ├── models/                # ResNet-18 for CIFAR-10
│   ├── nonlinearity/          # 非线性注入框架
│   │   ├── wrapper.py         # NonlinearInputWrapper
│   │   └── controller.py      # NonlinearityController
│   ├── training/              # 训练引擎（baseline/Fixed-NAT/Random-NAT/SGR-NAT）
│   ├── evaluation/            # 评估引擎（alpha sweep/校准）
│   ├── analysis/              # 层敏感度/误差累积分析
│   ├── visualization/         # 图表生成
│   └── data/                  # CIFAR-10 数据加载
├── configs/                   # YAML 配置文件
├── results/                   # 实验结果（JSON/CSV/PNG）
│   ├── baseline/              # Clean 基线训练（3 seeds）
│   ├── fixed_nat/             # Fixed-NAT 训练（3 seeds）
│   ├── random_nat/            # Random-NAT 训练
│   ├── sgr_nat/               # SGR-NAT 训练
│   ├── post_training/         # 统一 Alpha 扫描 & 方法对比
│   ├── analysis/              # 层敏感度 & 误差累积
│   ├── manifests/             # Checkpoint 清单
│   └── figures/               # 所有生成图表
├── tests/                     # 单元测试
├── experiments/               # 实验脚本
├── scripts/                   # 辅助脚本
├── reports/final/             # 最终交付物
│   ├── ActCIM_Robust_technical_report.md   # 技术报告（中文）
│   ├── ActCIM_Robust_paper_draft.md        # 论文初稿（英文）
│   ├── ppt_page_content.md                 # PPT内容页（中文）
│   ├── video_script.md                     # 视频脚本（中文）
│   ├── demo_runbook.md                     # 演示操作手册
│   └── final_result_audit.md               # 结果审计报告
├── pyproject.toml
├── requirements.txt
├── PROGRESS.md
├── RUN_STATUS.md
├── SUBMISSION_CHECKLIST.md
└── CHANGELOG.md
```

---

## CLI 命令参考 | CLI Commands

```powershell
# 训练 Clean 基线（~24 min）
python -m actcim_robust.cli train --config configs/baseline_full.yaml --seed 42

# Fine-Tuning Fixed-NAT（~5 min）
python -m actcim_robust.cli train --config configs/fixed_nat.yaml --seed 42 \
  --checkpoint results/baseline/seed_42/best.pt --method fixed_nat

# Fine-Tuning Random-NAT（~5.5 min）
python -m actcim_robust.cli train --config configs/random_nat.yaml --seed 42 \
  --checkpoint results/baseline/seed_42/best.pt --method random_nat

# Fine-Tuning SGR-NAT（~6.5 min）
python -m actcim_robust.cli train --config configs/sgr_nat.yaml --seed 42 \
  --checkpoint results/baseline/seed_42/best.pt --method sgr_nat

# Alpha 扫描
python -m actcim_robust.cli alpha-sweep --checkpoint results/baseline/seed_42/best.pt

# 层敏感度分析
python -m actcim_robust.cli layer-sensitivity --checkpoint results/baseline/seed_42/best.pt

# 误差累积分析
python -m actcim_robust.cli error-accumulation --checkpoint results/baseline/seed_42/best.pt

# 统一所有方法的 Alpha 扫描
python -m actcim_robust.cli unified-sweep

# 生成图表
python -m actcim_robust.cli build-figures

# 验证结果完整性
python -m actcim_robust.cli validate-results
```

---

## 实验环境 | Experimental Setup

| 项目 | 配置 |
|------|------|
| CPU | Intel Core i7-10700 @ 2.90GHz |
| GPU | NVIDIA GeForce RTX 4060 8GB GDDR6 |
| RAM | 32GB |
| OS | Windows 11 |
| Python | 3.12.5 |
| PyTorch | 2.5.1+cu121 |
| CUDA | 12.1 |

### 训练配置 | Training Configuration

**Clean 基线**：SGDR(lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)，cosine annealing + 3-epoch warmup，batch_size=128，50 epochs，AMP 混合精度。

**NAT Fine-Tuning**：SGD(lr=0.01)，cosine annealing + 1-epoch warmup，batch_size=128，10 epochs，早停 patience=10。

### 模型 | Model

- **架构**：ResNet-18 adapted for CIFAR-10（3×3 initial conv, stride=1, no initial max-pool）
- **参数量**：11,183,582（约 11.2M）
- **可注入层数**：21（20 Conv2d + 1 Linear）
- **数据集**：CIFAR-10（50k train / 10k test）

---

## 方法说明 | Method Descriptions

### Fixed-NAT（固定非线性感知训练）

微调时固定注入 $\alpha=+0.4$ 的非线性算子到所有 21 层。训练后模型在 $\alpha=+0.8$ 时准确率从 81.25% 提升至 91.79%。

### Random-NAT（随机非线性感知训练）

每次前向传播从 $U(-0.8, 0.8)$ 随机采样 $\alpha$，对所有层注入相同非线性。

### SGR-NAT（灵敏度引导随机非线性感知训练）

双分支架构 + KL 一致性正则化 + 课程学习。基于预计算的层敏感度排名选择性注入，训练注入率约 4.76%（1/21 层）。

---

## 关键实验结果 | Key Experimental Results

### 1. 非线性方向不对称性 | Directional Asymmetry

Clean 模型在 $\alpha=-0.8$ 时准确率 93.66%，在 $\alpha=+0.8$ 时准确率 81.25%——正向压缩型非线性的破坏力约为负向扩张型的 **22.8 倍**。

### 2. 非线性误差的累积放大 | Error Accumulation

- 单层注入 $\alpha=\pm 0.4$ 最大准确率下降仅 0.78%
- 21 层同时注入 $\alpha=+0.4$ 准确率降至 93.59%——非线性效应层层累积
- fc 全连接层是误差集中爆发点：$\alpha=-0.4$ 时标准差翻倍（3.63→7.16），饱和率 53.4%

### 3. Fixed-NAT 最有效 | Fixed-NAT is Most Effective

- 最差准确率：+10.54pp 提升（81.25% → 91.79%）
- AURC：+0.0091 提升（0.9283 → 0.9374）
- 不对称间隙：从 -0.0314 降至 0.0007（几乎消除）
- $\alpha=0$ 准确率仅下降 0.21pp（94.23% → 94.02%）
- 训练时间：仅 5 分钟

### 4. Random-NAT 和 SGR-NAT 增益有限 | Marginal Gains from Randomized Methods

- Random-NAT：最差准确率 +0.05pp，AURC *下降* 0.0003
- SGR-NAT：最差准确率 +0.81pp，AURC +0.0006
- 根本原因：10 epoch × lr=0.01 的微调窗口不足以让随机化方法收敛到有效鲁棒状态

### 5. 校准误差 | Calibration Error

| α | Clean ECE | Fixed-NAT ECE |
|---|-----------|---------------|
| 0.0 | 0.0326 | 0.0496 |
| +0.4 | 0.1235 | 0.0283 |
| +0.8 | 0.5602 | 0.4359 |

### 6. 多随机种子验证 | Multi-Seed Validation

| Seed | Clean Val Acc | Fixed-NAT Val Acc | Fixed-NAT Train Time |
|------|--------------|-------------------|---------------------|
| 42 | 94.84% | 94.90% | 5m01s |
| 3407 | 94.80% | 94.84% | 4m53s |
| 2026 | 94.98% | 94.98% | 4m53s |

---

## 未运行实验 | Unrun Experiments

| 实验 | 状态 | 原因 |
|------|------|------|
| ImageNet + ResNet-50 | NOT_RUN | 计算/数据资源限制 |
| VGG-16 / MobileNet 评估 | NOT_RUN | 架构多样性验证搁置 |
| Scratch训练 vs Fine-Tuning 对比 | NOT_RUN | 时间限制 |
| 对抗训练结合 | NOT_RUN | 优先级往后排 |
| CIM芯片在线校准 | NOT_RUN | 需要硬件平台 |

---

## 复现 | Reproducibility

所有实验结果可通过以下顺序完全复现：

```powershell
# 1. Clean基线训练（3 seeds, ~72 min total）
python -m actcim_robust.cli train --config configs/baseline_full.yaml --seed 42
python -m actcim_robust.cli train --config configs/baseline_full.yaml --seed 3407
python -m actcim_robust.cli train --config configs/baseline_full.yaml --seed 2026

# 2. Alpha扫描
python -m actcim_robust.cli alpha-sweep --checkpoint results/baseline/seed_42/best.pt

# 3. 层敏感度分析
python -m actcim_robust.cli layer-sensitivity --checkpoint results/baseline/seed_42/best.pt

# 4. 误差累积分析
python -m actcim_robust.cli error-accumulation --checkpoint results/baseline/seed_42/best.pt

# 5. NAT训练
python -m actcim_robust.cli train --config configs/fixed_nat.yaml --seed 42 --checkpoint results/baseline/seed_42/best.pt --method fixed_nat
python -m actcim_robust.cli train --config configs/random_nat.yaml --seed 42 --checkpoint results/baseline/seed_42/best.pt --method random_nat
python -m actcim_robust.cli train --config configs/sgr_nat.yaml --seed 42 --checkpoint results/baseline/seed_42/best.pt --method sgr_nat

# 6. 统一Alpha扫描（比较所有方法）
python -m actcim_robust.cli unified-sweep

# 7. 生成所有图表
python -m actcim_robust.cli build-figures

# 8. 验证结果完整性
python -m actcim_robust.cli validate-results
```

总预计时间：约 100 分钟（单 GPU，RTX 4060）

---

## 局限性与未来工作 | Limitations & Future Work

### 局限 | Limitations

1. 仅验证 ResNet-18 + CIFAR-10，泛化性未知
2. 三次多项式是对 CIM 真实非线性的简化
3. 仅 3 个种子，统计效力不足
4. SGR-NAT 仅尝试一种配置

### 未来工作 | Future Work

1. ImageNet + ResNet-50 大规模验证
2. 从 Scratch 训练的 NAT 方法对比
3. 实际 CIM 芯片上的硬件验证
4. 动态敏感度估计的 SGR-NAT
5. 与对抗训练、知识蒸馏的结合

---

## 许可 | License

参见 [LICENSE](LICENSE) 文件。

---

## 联系方式 | Contact

*（占位符 - 待参赛团队填写）*

- 团队名称：[待填写]
- 队长：[待填写]
- 邮箱：[待填写]
- 单位：[待填写]
