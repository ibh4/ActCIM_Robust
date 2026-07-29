# ActCIM-Robust · 存算一体芯片中非线性误差对推理精度的影响研究

> **Sensitivity-Guided Nonlinearity-Aware Training for Computing-in-Memory Neural Networks**
> 2026 Inno CIM 存算一体高校挑战赛参赛项目

存算一体（CIM）芯片的模拟计算链路会给激活引入确定性非线性失真 $f_\alpha(x) = m\cdot[\alpha(x/m)^3 + (1-\alpha)(x/m)]$。本仓库围绕该失真模型完成三项赛题任务——**敏感性分析、非线性感知训练（NAT）、鲁棒性增强**——并附完整的论文、PPT、网页演示与结果审计材料。

## 核心结果

| 方法 | Clean 准确率 (α=0) | 最差准确率 (α=+0.8) | AURC | 方向不对称差 |
|------|:---:|:---:|:---:|:---:|
| Clean 基线 | 94.23% | 81.25% | 0.9283 | −0.0314 |
| Random-NAT | 94.25% | 81.30% | 0.9281 | −0.0331 |
| SGR-NAT | 94.28% | 82.06% | 0.9290 | −0.0300 |
| **Fixed-NAT (α=+0.4)** | 94.02% | **91.79%** | **0.9374** | **+0.0007** |

- **正 α（激活压缩）是主导失效方向**：基线在 α=+0.8 掉 13pp，而 α=−0.8 仅掉 0.6pp；根源是压缩破坏判别信息，而非扰动幅度
- **Fixed-NAT 用 0.21pp 干净精度代价，将最差准确率提升 10.54pp**，并把方向不对称差归零；随机化策略（Random/SGR）仅千分位增益
- **强压缩引发欠自信型置信度崩溃**（conf 0.25 / acc 0.81），Fixed-NAT 将 ECE 从 0.560 降至 0.436
- 所有数字经过工程级审计，可由 CSV / 日志 / checkpoint 追溯（见 `ActCIM-Robust/reports/final/result_audit_statement.md`）

## 仓库结构

```
├── ActCIM-Robust/               # 主项目（可复现工程）
│   ├── src/actcim_robust/       # 框架源码：注入、训练、评估、分析、绘图
│   ├── configs/                 # YAML 实验配置（基线 / 三种 NAT / α 扫描）
│   ├── scripts/                 # 训练流水线与论文构建脚本
│   │   └── paper/               # build_docx.py / build_pptx.py（论文与PPT生成）
│   ├── tests/                   # pytest 单元与冒烟测试
│   ├── results/                 # 实验产物：α-Sweep CSV、层敏感性、图表、manifest
│   │   └── figures/paper_final/ # 10 张论文图（300DPI PNG+PDF）+ figures_manifest.json
│   ├── reports/final/           # 最终交付：论文 (MD/DOCX/PDF)、12页 PPTX、
│   │                            # references.bib、结果审计说明
│   └── data/splits/             # 训练/验证划分索引（CIFAR-10 原始数据需自行下载）
└── web_presentation/            # 14 页网页演示（录屏讲解用）
    ├── index.html               # 纯 HTML/CSS/JS，无外部依赖
    ├── video_script.md          # 10 分钟演示视频口播脚本（含时间码）
    └── export_pdf.py            # 网页 → 每页一张的 16:9 PDF
```

## 快速开始

```bash
cd ActCIM-Robust
pip install -r requirements.txt && pip install -e .

python -m actcim_robust.cli check-env      # 环境检查
python -m actcim_robust.cli test           # 运行测试

# 训练（基线 / NAT）
python -m actcim_robust.cli train --config configs/baseline.yaml --seed 42
python -m actcim_robust.cli train --config configs/fixed_nat.yaml --seed 42

# 评估与分析
python -m actcim_robust.cli alpha-sweep         # α ∈ [-0.8, 0.8] 11 点扫描
python -m actcim_robust.cli layer-sensitivity   # 逐层敏感性
python -m actcim_robust.cli error-accumulation  # 误差累积分析
python -m actcim_robust.cli build-figures       # 生成论文图表
```

### 网页演示

```bash
cd web_presentation
python3 -m http.server 8735    # 浏览器打开 http://localhost:8735
# 按 F 全屏，→/← 翻页，点击图片放大；支持 #page-N 直达
```

## 大文件说明

为控制仓库体积，以下内容**不随仓库分发**（已在 `.gitignore` 排除）：

| 内容 | 大小 | 获取方式 |
|---|---|---|
| CIFAR-10 原始数据 | 178 MB | [官网下载](https://www.cs.toronto.edu/~kriz/cifar.html) 解压至 `ActCIM-Robust/data/raw/` |
| 模型 checkpoint（16 个 `.pt`） | 85 MB/个 | 按上述命令重新训练（基线 50 epoch + NAT 微调 10 epoch），或联系作者获取 |

checkpoint 的 sha256 指纹与复评结果记录在 `ActCIM-Robust/results/manifests/checkpoint_inventory.json` 与 `reports/final/checkpoint_reverification.json`，可用于校验复现一致性。

## 主要交付物

| 交付物 | 位置 |
|---|---|
| 设计报告（20 页，MD/DOCX/PDF） | `ActCIM-Robust/reports/final/ActCIM_Robust_paper_final.*` |
| 介绍 PPT（12 页可编辑） | `ActCIM-Robust/reports/final/ActCIM_Robust_slides.pptx` |
| 论文插图（10 张 300DPI） | `ActCIM-Robust/results/figures/paper_final/` |
| 参考文献 | `ActCIM-Robust/reports/final/references.bib` |
| 结果审计说明 | `ActCIM-Robust/reports/final/result_audit_statement.md` |
| 网页演示 + 视频脚本 | `web_presentation/` |

## License

见 [ActCIM-Robust/LICENSE](ActCIM-Robust/LICENSE)。
