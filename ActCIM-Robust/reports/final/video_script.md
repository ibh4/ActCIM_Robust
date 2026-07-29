# ActCIM-Robust 答辩视频脚本（8分钟）

---

## 0:00–0:35 | 背景和问题

| 要素 | 内容 |
|------|------|
| **画面** | 标题页 + CIM芯片示意图 + 非线性失真动画 |
| **旁白** | "大家好，我们的项目是ActCIM-Robust，致力于解决存算一体芯片的非线性失真问题。CIM芯片通过在存储器内部直接计算，极大提升了能效比，但也引入了模拟域的非线性失真。一个在GPU上94.23%准确率的ResNet-18模型，部署到CIM芯片上可能暴跌到81.25%。我们的目标就是解决这个鲁棒性问题。" |
| **操作** | 无（开篇介绍） |
| **图表** | `results/figures/accuracy_vs_alpha.png` |
| **时长** | 35秒 |

---

## 0:35–1:20 | 非线性数学模型

| 要素 | 内容 |
|------|------|
| **画面** | 数学公式展示 + 函数曲线动画（α变化时曲线的形状变化） |
| **旁白** | "我们用一个三次多项式来参数化CIM非线性。公式是 f_alpha(x) = m × [α×(x/m)³ + (1-α)×(x/m)]。当α等于零时退化为线性；α大于零产生压缩效应——大信号被衰减，相当于软截断；α小于零产生扩张效应——大信号被放大。注意这个函数是奇对称的，过零点的特性不变。我们使用的α范围从负的0.8到正的0.8，覆盖了CIM芯片的典型工作区间。" |
| **操作** | 展示公式推导 |
| **图表** | `results/figures/activation_distribution_shift.png` |
| **时长** | 45秒 |

---

## 1:20–2:00 | 项目架构

| 要素 | 内容 |
|------|------|
| **画面** | 项目目录结构 + 模块关系图 |
| **旁白** | "项目采用模块化设计。核心是nonlinearity模块，包含NonlinearInputWrapper和NonlinearityController，实现了灵活的非线性注入框架。训练、评估、分析各自独立模块。命令行工具统一入口——training、alpha-sweep、layer-sensitivity、build-figures、unified-sweep、validate-results等子命令。配置文件驱动，所有结果自动保存为JSON、CSV和PNG，确保完全可复现。" |
| **操作** | 展示项目目录 + 配置文件 |
| **图表** | 项目结构图（文字） |
| **时长** | 40秒 |

---

## 2:00–2:50 | Alpha扫描结果

| 要素 | 内容 |
|------|------|
| **画面** | accuracy_vs_alpha_all_methods.png 全屏展示，依次高亮四条线 |
| **旁白** | "现在来看最核心的实验结果——Alpha扫描。这张图展示了四种方法在11个α值下的测试准确率。蓝色Clean线：α等于零时准确率94.23%，α等于正的零点八时断崖式下跌到81.25%——下降了近13个百分点。但α等于负的零点八时仍有93.66%，只降了零点五七个百分点。这是22.8倍的差距——正向和负向非线性的影响完全不对称。再看橙色Fixed-NAT线：在整个正方向远远高于其他三条线，α等于正的零点八时仍保持91.79%，比Clean高了整整10.54个百分点。" |
| **操作** | 依次高亮：Clean蓝线→SGR-NAT红线→Random-NAT绿线→Fixed-NAT橙线 |
| **图表** | `results/figures/post_training/01_accuracy_vs_alpha_all_methods.png` |
| **时长** | 50秒 |

---

## 2:50–3:40 | 层敏感性分析

| 要素 | 内容 |
|------|------|
| **画面** | layer_sensitivity_bar.png + layer_error_accumulation 对比图 |
| **旁白** | "接下来我们做了更细粒度的分析——逐层敏感度。我们对ResNet-18的21个层逐个注入α等于正负零点四的非线性，看单层扰动的影响。结果发现：单层骚扰的影响极小——最大的准确率下降只有零点七八个百分点。但21层同时注入时，准确率从94.23%降到93.59%。这说明非线性误差是累积放大的。从误差累积图可以看到——浅层和中层几乎不受影响，相对L2距离为零。但到了fc全连接层，误差急剧爆发——标准差在α等于负的零点四时从三点六三翻倍到七点一六，符号翻转率达到百分之三点四。非线性损伤的本质是层层传递、在分类层集中爆发的累积效应。" |
| **操作** | 切换layer_sensitivity_bar → layer_error_accumulation双图 |
| **图表** | `results/figures/layer_sensitivity_bar.png`, `results/figures/layer_error_accumulation_pos_04.png`, `results/figures/layer_error_accumulation_neg_04.png` |
| **时长** | 50秒 |

---

## 3:40–4:30 | Random-NAT

| 要素 | 内容 |
|------|------|
| **画面** | 回到accuracy_vs_alpha_all_methods.png，高亮绿色Random-NAT线 |
| **旁白** | "基于这些分析，我们提出了三种非线性感知训练方法，简称NAT。第一种是Random-NAT——训练时每次前向传播随机从均匀分布采样一个α值。直觉上，这应该让模型适应各种非线性强度。但从绿色线可以看到，结果几乎与Clean蓝线完全重叠。最差准确率仅从81.25%提升到81.30%，几乎无任何改善。AURC还从0.9283降到了0.9281。为什么？因为10个epoch的小学习率微调窗口太窄，随机梯度信号互相矛盾，模型根本没有学到有效的鲁棒策略。" |
| **操作** | 高亮Random-NAT绿色线 + 展示关键数字 |
| **图表** | `results/figures/post_training/01_accuracy_vs_alpha_all_methods.png`（绿色线） |
| **时长** | 50秒 |

---

## 4:30–5:30 | SGR-NAT

| 要素 | 内容 |
|------|------|
| **画面** | SGR-NAT架构图（双分支） + accuracy图高亮红色SGR-NAT线 |
| **旁白** | "第二种方法叫SGR-NAT——灵敏度引导的随机非线性感知训练。这是最复杂的方法。我们引入双分支架构：一个Clean分支做标准前向传播，另一个Nonlinear分支基于预计算的敏感度排名选择性地注入非线性。两个分支之间通过KL散度做一致性正则化，让有扰动的输出逼近无扰动的输出。还引入了课程学习——训练初期弱扰动、后期强扰动。结果呢？红色SGR-NAT线在无扰动下准确率最高，达到94.28%。但在正方向高α区域改善有限——最差准确率82.06%，只比Clean高了0.81个百分点。而且训练不稳定，最佳模型出现在第0个epoch。根本问题是训练只注入一层、推理却要面对21层——扰动空间严重不匹配。" |
| **操作** | 展示SGR-NAT架构 → 切换到accuracy图红色线 → 展示训练状态面板 |
| **图表** | `results/figures/post_training/01_accuracy_vs_alpha_all_methods.png`（红色线） |
| **时长** | 60秒 |

---

## 5:30–6:30 | Fixed-NAT & 方法对比

| 要素 | 内容 |
|------|------|
| **画面** | 对比柱状图：worst_case_accuracy + AURC + asymmetry_gap |
| **旁白** | "第三种方法Fixed-NAT是最简单的——固定α等于正零点四训练。但最简单的却最有效。让我们直接看对比：在最差准确率上，Fixed-NAT的91.79%遥遥领先于Clean的81.25%、Random-NAT的81.30%和SGR-NAT的82.06%。提升了10.54个百分点！在AURC上，Fixed-NAT的0.9374显著高于其他方法的0.928到0.929。最惊人的是不对称间隙——Clean的不对称间隙是负的零点零三一四，表明正方向性能远低于负方向。Fixed-NAT把这个间隙压缩到了零点零零零七——几乎完全消除。代价是什么？仅仅是α等于零时准确率从94.23%降到94.02%，零点二一个百分点的微小下降。" |
| **操作** | 逐一切换对比柱状图 → 展示关键数字 |
| **图表** | `results/figures/post_training/05_worst_case_accuracy_comparison.png`, `results/figures/post_training/06_aurc_all_comparison.png`, `results/figures/post_training/10_asymmetry_gap_comparison.png`, `results/figures/post_training/08_clean_accuracy_vs_worst_accuracy.png` |
| **时长** | 60秒 |

---

## 6:30–7:15 | 统计与局限性

| 要素 | 内容 |
|------|------|
| **画面** | 多种子结果表 + 局限性列表 |
| **旁白** | "我们用了三个随机种子来验证Fixed-NAT的一致性。种子42、3407和2026下，Fixed-NAT的验证准确率分别是94.90%、94.84%和94.98%，均与对应Clean基线持平或略优。训练时间方面，每个Fixed-NAT只需约5分钟——相比基线训练的24分钟，节省了79%的时间。当然，我们的工作有明确局限：第一，只验证了ResNet-18一种架构和CIFAR-10一个数据集，结论在其他模型和数据上的泛化性未知；第二，三次多项式是对CIM真实非线性的简化；第三，三个种子在统计上是不够的。第四，SGR-NAT只尝试了一种配置，参数空间远未被充分探索。" |
| **操作** | 展示多种子表 → 展示局限性 |
| **图表** | 文字表格（多种子结果） |
| **时长** | 45秒 |

---

## 7:15–8:00 | 结论

| 要素 | 内容 |
|------|------|
| **画面** | 总结要点 + 项目logo + 致谢 |
| **旁白** | "最后总结三个核心结论。第一，CIM非线性的正负方向不对称性是一个关键发现——压缩型的破坏力是扩张型的20多倍，CIM芯片设计者应该优先关注压制式失真。第二，Fixed-NAT是当前最具实用价值的鲁棒性方案——只需要5分钟的微调，就像给模型打了一针'非线性疫苗'，让最差准确率从81.25%提升到91.79%。第三，Random-NAT和SGR-NAT在微调窗口内增益有限——但这个问题值得在更大规模的设置下进一步探索，比如从Scratch训练或使用更大的微调窗口。感谢各位评委，欢迎提问。" |
| **操作** | 定格在总结页 |
| **图表** | 回放 `results/figures/post_training/01_accuracy_vs_alpha_all_methods.png` |
| **时长** | 45秒 |

---

## 附录：演示命令速查

视频中若需要展示命令行操作，可使用以下实际可运行的命令：

```powershell
# 1. 环境检查
python -m actcim_robust.cli check-env

# 2. 运行测试
pytest tests/ -v

# 3. Alpha扫描演示（快速3个alpha值）
python -m actcim_robust.cli alpha-sweep --checkpoint results/baseline/seed_42/best.pt

# 4. 层敏感度分析
python -m actcim_robust.cli layer-sensitivity --checkpoint results/baseline/seed_42/best.pt

# 5. 统一所有方法的alpha扫描（生成对比数据）
python -m actcim_robust.cli unified-sweep

# 6. 生成所有图表
python -m actcim_robust.cli build-figures

# 7. 验证结果完整性
python -m actcim_robust.cli validate-results

# 8. Fixed-NAT训练演示
python -m actcim_robust.cli train --config configs/fixed_nat.yaml --seed 42 --checkpoint results/baseline/seed_42/best.pt --method fixed_nat
```

---

## 图表路径速查

| 用途 | 路径 |
|------|------|
| Alpha扫描主图 | `results/figures/post_training/01_accuracy_vs_alpha_all_methods.png` |
| 准确率下降 | `results/figures/post_training/02_accuracy_drop_vs_alpha_all_methods.png` |
| ECE曲线 | `results/figures/post_training/03_ece_vs_alpha_all_methods.png` |
| 置信度曲线 | `results/figures/post_training/04_confidence_vs_alpha_all_methods.png` |
| 最差准确率柱状图 | `results/figures/post_training/05_worst_case_accuracy_comparison.png` |
| AURC柱状图 | `results/figures/post_training/06_aurc_all_comparison.png` |
| Clean Acc vs Worst Acc | `results/figures/post_training/08_clean_accuracy_vs_worst_accuracy.png` |
| 不对称间隙对比 | `results/figures/post_training/10_asymmetry_gap_comparison.png` |
| 层敏感度柱状图 | `results/figures/layer_sensitivity_bar.png` |
| 误差累积（正方向） | `results/figures/layer_error_accumulation_pos_04.png` |
| 误差累积（负方向） | `results/figures/layer_error_accumulation_neg_04.png` |
| 可靠性图 | `results/figures/post_training/reliability_clean_alpha_0.png` 等 |
