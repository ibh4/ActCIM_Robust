# ActCIM-Robust 结果审计说明（Result Audit Statement）

审计日期：2026-07-29　｜　审计范围：`results/`、`configs/`、`src/actcim_robust/`、训练日志与全部 checkpoint
审计原则：论文与 PPT 中的每一个数字必须可由项目内 CSV / JSON / 日志 / checkpoint 追溯；无法追溯或与代码实际行为不符的表述一律更正或删除。

---

## 1. 审计结论摘要

| # | 审计项 | 结论 |
|---|---|---|
| 1 | "推荐模型 seed 2026 vs 推荐 checkpoint seed 42" 不一致 | **已实证定论**：头条鲁棒性数字来自 **seed 42** checkpoint（§2） |
| 2 | Fixed-NAT 多种子是否有完整 Alpha Sweep | **有**（seed 42/2026/3407 各 11 点），但仅 n=3 且对比方法单种子 → 论文只作描述性一致性检查，**不宣称统计显著性** |
| 3 | 非线性注入范围 | 实际仅 **4/21 层** 生效（控制器命名碰撞），全项目训练/评估同口径（§4） |
| 4 | AURC 定义 | 梯形积分归一化（trapz/1.6），**不是**算术平均（§5） |
| 5 | 归一化因子 m | 逐张量动态 max\|x\|，非草稿所写固定常数 m=10 |
| 6 | Random-NAT 训练分布 | U(−0.5, +0.5)，非草稿所写 ±0.8（源码 + 配置双重核验） |
| 7 | 强正 α 校准失准方向 | **置信度崩溃型欠自信**（conf 0.25 / acc 0.81），非"过自信"（§6） |
| 8 | 层敏感性排序实验 | 双重退化（命名碰撞 + 128 样本分辨率 0.78 pp），**不作为逐层证据** |
| 9 | Fixed-NAT 校准分箱缺失 | 已按同协议补算（仅评估），α=0 结果与 CSV 逐位一致，交叉验证通过（§7） |
| 10 | Random/SGR best_epoch=0 | 属实：微调几乎未超过初始点，与其指标与基线千分位差异互证 |

## 2. seed 42 vs seed 2026 不一致的实证定论（核心审计项）

**问题**：技术报告称"推荐模型为 Fixed-NAT seed 2026"（其验证集准确率 94.98% 为三种子最高），但推荐 checkpoint 路径指向 `results/fixed_nat/fixed_nat/seed_42/best.pt`。

**方法**：`scripts/paper/verify_checkpoints_vs_csv.py` 重新加载 4 个 checkpoint（baseline seed42、fixed_nat seed42/2026/3407），在 CIFAR-10 全部 10 000 张测试图像上复评 α=0 与 α=+0.8，输出 `reports/final/checkpoint_reverification.json`。

**结果**（α=0 准确率指纹，四者互不相同）：

| checkpoint | α=0 复评 | α=0 CSV | α=+0.8 复评 | α=+0.8 CSV |
|---|---|---|---|---|
| baseline seed 42 (epoch 48) | 0.9423 | 0.9423 ✓ | 0.8102 | 0.8125（差 0.23 pp） |
| **fixed_nat seed 42 (epoch 9)** | **0.9402** | **0.9402 ✓（唯一匹配）** | 0.9169 | 0.9179（差 0.10 pp） |
| fixed_nat seed 2026 (epoch 9) | 0.9429 | 0.9429（multi_seed CSV）| 0.9131 | — |
| fixed_nat seed 3407 (epoch 9) | 0.9423 | 0.9423（multi_seed CSV）| 0.9156 | — |

**定论**：`fixed_nat_alpha_sweep.csv`（含论文头条数字 94.02% / 91.79% / AURC 0.9374）的 α=0 指纹 0.9402 **唯一匹配 seed 42**。不一致的根源是两套口径：模型选择按 val acc（seed 2026 最高 94.98%），鲁棒性报告按测试集 α-Sweep（跑的是 seed 42）。**论文统一以 seed 42 为主结果口径**（唯一与 clean/random/sgr 同协议的种子），seed 2026/3407 仅作描述性一致性检查。

**α≠0 处 ≤0.23 pp 的复评偏差解释**：f_α 的归一化因子 m=max|x| 逐张量动态计算，评估 batch 划分不同（复评 500 vs 原始 256）导致 m 不同；α=0 时 f_α 恒等、不受影响，故 α=0 指纹逐位一致。正文表格一律采用原始 CSV 值。

## 3. 多随机种子表述（按任务要求执行）

- Fixed-NAT 三个训练种子（42/2026/3407）**均有完整 11 点 Alpha Sweep**（`results/post_training/multi_seed/`），推翻早期审计备忘"NOT RUN"的记载。
- 三种子结果：Worst 91.79% / 91.36% / 91.60%（均在 α=+0.8，极差 0.43 pp）；AURC 0.9374 / 0.9387 / 0.9386（梯形积分复算核验）。
- **限制**：对比方法（Clean/Random/SGR）各仅 1 个训练种子；Fixed-NAT n=3 且共享同一评估协议种子。**论文与 PPT 均不做显著性检验、不宣称多随机种子统计显著性**（论文 §8.3、图10 图注、PPT P05 均已声明）。

## 4. 注入范围实证（4/21 层）

`scripts/paper/verify_injection_scope.py`：ResNet-18-CIFAR 的 21 个 Conv/Linear 全部被 wrap，但 `NonlinearityController._wrappers` 以子模块局部名为键，21 条登记名塌缩为 4 个唯一键（'0','conv1','conv2','fc'）；`enable_all()` 实际仅启用 **fc、layer4.0.downsample.0、layer4.1.conv1、layer4.1.conv2**。α=+0.8 下 logit 相对 L2 扰动：4 层口径 0.591，真全层口径 0.983（约六成）。

影响评估：训练与评估全程同口径 → 四方法对比公平有效；但结论必须限定为"深层 4 层激活非线性"扰动模型（论文 §3.1/§9.3/§11 与 PPT P05 已限定）。层敏感性排序实验（`layer_sensitivity_ranked.csv`）受同一碰撞影响且仅 128 样本（分辨率 0.78 pp），**不作为逐层敏感性证据**（论文 §6.2）。

## 5. 指标口径核验

- **AURC** = trapz(A, α) / 1.6（`evaluation/robustness_metrics.py`），非算术平均（算术平均将得 0.9262 而非 0.9283）。
- **AURC⁺/AURC⁻**：分别在 [0.1,0.8] / [−0.8,−0.1] 上同式计算（宽度 0.7），严格排除 α=0。
- **ECE**：15 等宽置信度分箱，最末箱闭区间（`scripts/calibration_audit.py`，batch 256）。
- 表1 全部指标与 `fixed_nat_comparison.json` 逐项吻合；表2 与两份 sweep CSV 逐位吻合。

## 6. 校准失准方向更正（重要）

早期理解为"强正 α 下模型保持约 0.99 置信度（过自信）"。对 4 份 `*_alpha_pos_08_bins.csv` 复核 signed gap（conf − acc）：

| 模型（α=+0.8） | mean conf | acc | ECE | 方向 |
|---|---|---|---|---|
| clean | 0.2523 | 0.8125 | 0.560 | 欠自信 |
| random_nat | 0.2520 | 0.8130 | 0.561 | 欠自信 |
| sgr_nat | 0.2693 | 0.8206 | 0.551 | 欠自信 |
| fixed_nat（补算） | 0.4821 | 0.9180 | 0.436 | 欠自信 |

**定论**：强正 α 引发**置信度崩溃型欠自信**（softmax 趋于均匀），与 Guo et al. 常见的过自信方向相反；负 α 侧才是轻度过自信（conf≈0.99、ECE 0.05–0.07）。论文 §10、图8/图9 图注、PPT P10 均按更正后方向表述。

## 7. Fixed-NAT 校准分箱补算（仅评估，不训练）

原 `results/post_training/calibration/` 仅有 clean/random/sgr 的分箱文件。`scripts/paper/compute_fixed_nat_calibration.py` 按同一协议（15 bins、batch 256、seed 42 checkpoint）补算并输出 `fixed_nat_alpha_0_bins.csv`、`fixed_nat_alpha_pos_08_bins.csv`、`fixed_nat_calibration_summary.json`。

**交叉验证**：补算 α=0 → acc 0.9402 / ECE 0.04958 / mean conf 0.98962，与 `fixed_nat_alpha_sweep.csv` 中该行**逐位一致**，证明补算协议与原始评估协议一致；α=+0.8 → acc 0.9180 / ECE 0.4360 / mean conf 0.4821（图9c、论文 §10 引用值）。

## 8. 与早期草稿的差异更正清单

| # | 草稿表述 | 更正后 | 依据 |
|---|---|---|---|
| 1 | AURC 为各点算术平均 | 梯形积分 / 1.6 | robustness_metrics.py |
| 2 | 归一化 m=10 固定 | m=max\|x\| 逐张量动态 | nonlinearity/function.py |
| 3 | Random-NAT α∈±0.8 | α ~ U(−0.5,+0.5) | random_nat.py + configs/random_nat.yaml |
| 4 | 全 21 层注入 | 实际生效深层 4 层 | verify_injection_scope.py |
| 5 | 强正 α 过自信 | 置信度崩溃型欠自信 | calibration/*_bins.csv 复核 |
| 6 | 多种子 sweep "NOT RUN" | 存在完整 sweep，仅作描述性检查 | multi_seed/*.csv |

## 9. 交付物清单

| 交付物 | 路径 |
|---|---|
| 论文 Markdown | `reports/final/ActCIM_Robust_paper_final.md` |
| 论文 DOCX | `reports/final/ActCIM_Robust_paper_final.docx` |
| 论文 PDF（20 页） | `reports/final/ActCIM_Robust_paper_final.pdf` |
| PPT（12 页可编辑） | `reports/final/ActCIM_Robust_slides.pptx` |
| 参考文献 BibTeX | `reports/final/references.bib`（18 条，2 条标"待核验"，无编造 DOI/页码） |
| 图片（10 张 × PNG+PDF，300 DPI） | `results/figures/paper_final/fig01–fig10` |
| 图片清单 | `results/figures/paper_final/figures_manifest.md`（含编号、标题、数据来源、完整图注） |
| checkpoint 复评记录 | `reports/final/checkpoint_reverification.json` |
| 结果审计说明（本文件） | `reports/final/result_audit_statement.md` |
| 生成脚本 | `scripts/paper/{generate_paper_figures,verify_checkpoints_vs_csv,verify_injection_scope,compute_fixed_nat_calibration,build_docx,build_pptx}.py` |

## 10. 复现说明

所有图片与文档可在项目根目录重新生成（python3 + torch CPU + matplotlib + pandas + python-docx + python-pptx + LibreOffice）：

```bash
python3 scripts/paper/generate_paper_figures.py          # 图1–图10 (PNG+PDF, 300 DPI)
python3 scripts/paper/verify_checkpoints_vs_csv.py       # checkpoint 复评
python3 scripts/paper/compute_fixed_nat_calibration.py   # Fixed-NAT 校准补算
python3 scripts/paper/build_docx.py                      # 论文 DOCX
/Applications/LibreOffice.app/Contents/MacOS/soffice --headless \
  --convert-to pdf reports/final/ActCIM_Robust_paper_final.docx \
  --outdir reports/final                                 # 论文 PDF
python3 scripts/paper/build_pptx.py                      # 12 页 PPTX
```

注意：α≠0 的复评结果对评估 batch 划分存在 ≤0.23 pp 的敏感性（m 动态归一化所致），跨环境复现须使用 batch size 256 顺序遍历。
