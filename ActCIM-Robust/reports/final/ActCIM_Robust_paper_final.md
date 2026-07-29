# 面向存算一体激活非线性的鲁棒性建模与非线性感知训练：ActCIM-Robust

> 本文全部实验数值均可由项目内 CSV/JSON/checkpoint 追溯复现；数据来源与复核记录见附录 A 与《结果审计说明》（reports/final/result_audit_statement.md）。

## 摘要

存算一体（Computing-in-Memory, CIM）架构在模数转换、驱动放大与阵列传输环节会给神经网络的激活值引入非线性失真，导致部署精度不可控退化。本文围绕一类可参数化的三次激活非线性 $f_\alpha(x)=m\cdot[\alpha(x/m)^3+(1-\alpha)(x/m)]$（$m=\max|x|$ 为逐张量动态归一化因子，$\alpha\in[-0.8,+0.8]$ 表征非线性强度与方向），在 CIFAR-10 / ResNet-18 上系统研究了该非线性对推理精度与置信度校准的影响，并对比了三种非线性感知训练（Nonlinearity-Aware Training, NAT）策略：固定强度 Fixed-NAT、随机强度 Random-NAT 与灵敏度引导的 SGR-NAT。实验发现：(1) 模型对正 $\alpha$（激活压缩）显著比负 $\alpha$（激活扩张）敏感——干净基线在 $\alpha=+0.8$ 处准确率跌至 81.25%，而 $\alpha=-0.8$ 处仍有 93.66%（方向不对称差 −3.14 pp）；(2) Fixed-NAT（训练 $\alpha=+0.4$）将最差准确率从 81.25% 提升至 91.79%（+10.54 pp），AURC 从 0.9283 提升至 0.9374，同时 Clean Accuracy 保持 94.02%（基线 94.23%，代价仅 0.21 pp）；(3) Random-NAT 与 SGR-NAT 的鲁棒性与基线几乎持平，且 SGR-NAT 与 Random-NAT 表现接近；(4) 强正非线性引发置信度崩溃型欠自信失准（clean 基线 $\alpha=+0.8$ 时 ECE=0.560、平均置信度仅 0.252），Fixed-NAT 将 ECE 降至 0.436 并将平均置信度回升至 0.482。审计还实证确认：受控制器命名碰撞影响，本项目全部训练与评估中的"全层注入"实际仅作用于网络深层 4 个层，该口径在所有方法间保持一致，因此四方法对比结论内部自洽，但结论应表述为"深层非线性扰动模型"下的鲁棒性。除 Fixed-NAT 外各方法仅有单一训练种子，Fixed-NAT 的三个训练种子结果方向与量级一致（最差准确率 91.36%–91.79%），仅作描述性一致性检查，本文不宣称多随机种子统计显著性。

**关键词**：存算一体；模拟计算非理想性；激活非线性；非线性感知训练；鲁棒性评估；模型校准；ResNet；CIFAR-10

## 1 引言

存算一体架构将矩阵-向量乘法下沉到存储阵列内部完成，可数量级地降低数据搬运能耗，被视为神经网络推理加速的重要路线[1][2][3][4][5]。然而，模拟域计算天然伴随非理想性：器件噪声、电导漂移、IR 压降、跨阻放大器与 ADC 的有限线性区等，都会使实际传输特性偏离理想线性关系[6][7]。既有研究大多将这些非理想性建模为权重或输出上的加性/乘性噪声，并通过噪声感知训练缓解[8][9]；相对而言，**激活值经历的确定性非线性失真**——例如驱动/读出链路饱和引起的压缩、或增益校准偏差引起的扩张——所受关注较少，而它同样会随网络深度逐层累积并破坏模型的判别边界与置信度校准[14]。

本文以一类单参数三次非线性 $f_\alpha$ 为载体（§4），把"激活经过非理想传输环节"抽象为在层输入端插入可控失真算子，并给出一套完全可追溯的评估协议：11 点 $\alpha$-Sweep、最差准确率（Worst-Case Accuracy）、准确率-α 曲线归一化面积（AURC）、方向不对称差（Asymmetry Gap）与期望校准误差（ECE）。在该协议下我们回答三个问题：

1. **失真方向是否等价？** 不等价。正 $\alpha$（压缩）造成的精度损失远大于同幅值负 $\alpha$（扩张）：干净基线在 $\alpha=+0.8$ 与 $-0.8$ 处的准确率分别为 81.25% 与 93.66%（§6.1）。
2. **哪种训练策略最有效？** 固定正强度的 Fixed-NAT（$\alpha=+0.4$）以 0.21 pp 的干净精度代价，将最差准确率提升 10.54 pp、AURC 从 0.9283 提升至 0.9374；而随机强度的 Random-NAT 与灵敏度引导的 SGR-NAT 均未取得实质增益，二者表现接近（§8）。
3. **失真如何影响置信度？** 强正非线性使 softmax 置信度整体崩溃（平均置信度 0.25），形成严重欠自信；Fixed-NAT 显著缓解该失准（§10）。

本文的另一项贡献是**结果审计**：我们对项目全部结果文件执行了逐位核对与 checkpoint 复评（附录 A），实证锁定了论文头条数字对应的具体 checkpoint（Fixed-NAT seed 42），修正了技术报告中"推荐模型 seed 2026 / 推荐 checkpoint seed 42"的口径不一致，并确认了非线性注入的真实作用范围（深层 4 层而非全部 21 层），使全部结论建立在与代码实际行为一致的描述之上。

## 2 相关工作

**存算一体加速器与模拟非理想性。** ISAAC[1] 与 PRIME[2] 确立了 ReRAM 交叉阵列上原位模拟乘加的体系结构范式；Yao 等[3]与 Wan 等[4]分别展示了全硬件忆阻器 CNN 与集成 CIM 芯片。模拟域计算的精度受器件与电路非理想性制约：Ambrogio 等[6]在相变存储器上通过混合精度架构达到等效软件精度，Joshi 等[7]系统分析了 PCM 噪声对深度网络推理的影响并提出训练期噪声注入补偿。这些工作主要关注权重侧噪声；本文关注激活传输路径上的确定性非线性，并将其参数化为可扫描的单变量族。

**噪声/失真感知训练。** 在训练期注入目标扰动以换取部署鲁棒性的思想可追溯至 Murray 与 Edwards 对突触权重噪声的分析[8]；近年在模拟加速器背景下发展出多种噪声感知训练与蒸馏方案[9]。本文对比的三种 NAT 策略是该思想在"激活非线性"扰动下的三种实例化：固定强度（Fixed）、随机强度（Random）与灵敏度引导逐层随机（SGR）。SGR-NAT 使用的 KL 一致性正则与知识蒸馏[15]的教师-学生框架同构（干净前向作为教师）。

**穿过非可导算子的梯度。** 当训练期在前向图中插入非线性/量化算子时，直通估计器（STE）[10]及其在二值网络中的应用[11]是标准做法。本文的 $f_\alpha$ 处处可导，训练时直接反传其解析梯度，无需 STE；相关工作在此仅作为方法学参照。

**骨干网络与数据集。** 实验采用 ResNet-18[12] 的 CIFAR-10[13] 适配变体（3×3 首层卷积、无最大池化），是鲁棒性研究的常用基准设置。

**置信度校准。** Guo 等[14]指出现代深度网络普遍存在校准失准并推广了 ECE 度量；ECE 的分箱估计源自 Naeini 等[16]，Brier 分数[17]为经典的概率预报评分。本文将校准分析扩展到激活非线性场景，发现强压缩引发与"过自信"相反的置信度崩溃型欠自信（§10）。

## 3 问题定义

### 3.1 威胁模型

设网络第 $l$ 层的理想计算为 $y_l = W_l * x_l + b_l$。部署在 CIM 阵列时，激活 $x_l$ 在到达乘加阵列前需经过数模驱动、位线传输与读出链路，其整体传输特性记为算子 $T(\cdot)$。我们将 $T$ 参数化为单参数族 $f_\alpha$（§4），得到扰动后计算：

$$\tilde{y}_l = W_l * f_\alpha(x_l) + b_l$$

$\alpha=0$ 时 $f_0=\mathrm{id}$，退化为理想计算；$\alpha>0$ 表示压缩型失真（小信号增益 <1，趋向三次饱和）；$\alpha<0$ 表示扩张型失真（小信号增益 >1）。部署时 $\alpha$ 由工艺与工作点决定、推理期近似恒定，但设计期未知，故鲁棒性目标是在 $\alpha\in[-0.8,+0.8]$ 全区间上维持精度。

**注入范围（实证口径）。** 框架设计意图是对全部 21 个 Conv/Linear 层注入。审计发现（附录 A.3）：控制器以子模块局部名称索引 wrapper，ResNet-18 中 21 条登记名碰撞为 4 个唯一键，导致 `enable_all()` 实际仅激活 **layer4.1.conv1、layer4.1.conv2、layer4.0.downsample.0、fc** 这 4 个深层。该行为贯穿本项目全部训练与评估（各方法完全同口径），因此方法间对比公平有效；但本文所有结论对应的扰动模型应准确表述为"**深层 4 层激活非线性**"，而非全网络注入。实测该口径下 $\alpha=+0.8$ 对 logit 的相对 L2 扰动为 0.59（真正全 21 层注入为 0.98），即本文扰动强度约为设计意图的六成。

### 3.2 评估协议

对每个模型 checkpoint，在 CIFAR-10 全部 10 000 张测试图像上，对 $\alpha\in\{-0.8,-0.6,-0.4,-0.2,-0.1,0,0.1,0.2,0.4,0.6,0.8\}$ 共 11 个点分别完成一次完整评估（全局设定 $\alpha$、启用全部生效层），记录 Top-1 准确率 $A(\alpha)$、交叉熵损失、ECE 与平均置信度。

### 3.3 鲁棒性指标

以下定义与代码实现（`evaluation/robustness_metrics.py`）逐一核对一致：

- **AURC**（Accuracy under Robustness Curve）：准确率-α 曲线的梯形积分除以区间宽度，
$$\mathrm{AURC}=\frac{1}{\alpha_{\max}-\alpha_{\min}}\int_{\alpha_{\min}}^{\alpha_{\max}} A(\alpha)\,d\alpha \approx \frac{\mathrm{trapz}(A,\alpha)}{1.6}$$
  取值越接近 1 越好。注意 AURC **不是**各点准确率的简单算术平均（早期草稿的这一表述有误，按算术平均将得到 0.9262 而非 0.9283）。
- **AURC⁺ / AURC⁻**：分别在 $\alpha\in[0.1,0.8]$ 与 $[-0.8,-0.1]$ 上按同式计算（区间宽度 0.7），严格排除 $\alpha=0$ 点。
- **Worst-Case Accuracy**：$\min_\alpha A(\alpha)$，以及对应的 $\arg\min$。
- **Asymmetry Gap**：$\mathrm{mean}_{\alpha>0}A(\alpha)-\mathrm{mean}_{\alpha<0}A(\alpha)$，负值表示正 α 侧更差。
- **Mean Perturbed Accuracy**：$\alpha\neq 0$ 的 10 个点的准确率均值。
- **ECE**（15 等宽置信度分箱）：$\mathrm{ECE}=\sum_{i=1}^{15}\frac{n_i}{N}\left|\mathrm{acc}(B_i)-\mathrm{conf}(B_i)\right|$，最末箱为闭区间[14][16]。

## 4 非线性数学模型

### 4.1 函数形式

采用逐张量动态归一化的三次多项式（与源码 `nonlinearity/function.py` 完全一致）：

$$f_\alpha(x)=m\cdot\left[\alpha\left(\frac{x}{m}\right)^3+(1-\alpha)\left(\frac{x}{m}\right)\right],\qquad m=\max(|x|)\ \text{（当前张量，钳位下限 }10^{-8}\text{）}$$

三点性质：(1) **端点不动**：$|x|=m$ 处 $f_\alpha(\pm m)=\pm m$，失真集中于中小幅值区；(2) **一阶齐次**：$f_\alpha(kx)=k f_\alpha(x)$，对激活整体尺度不敏感；(3) **处处可导**：小信号增益为 $1-\alpha$（在 $x\to 0$ 处），训练期可直接解析反传。需要强调 $m$ 是**随输入张量动态变化**的（早期草稿误写为固定常数 $m=10$）；该设计带来的副作用是：$\alpha\neq 0$ 时评估结果对 batch 组成存在轻微依赖（附录 A.2 中复评偏差 ≤0.23 pp 即源于此）。

图2(a) 给出 $\alpha\in\{-0.8,-0.4,0,+0.4,+0.8\}$ 的输入-输出曲线；图2(b) 的增益视图直观显示：正 $\alpha$ 压缩中小幅值信号（增益低至 $1-\alpha$），负 $\alpha$ 将其放大（增益高至 $1-\alpha>1$）。

![图2 非线性函数及正负Alpha响应](../../results/figures/paper_final/fig02_nonlinearity.png)

**图2** 非线性传输函数 $f_\alpha(x)=m[\alpha(x/m)^3+(1-\alpha)(x/m)]$（$m=\max|x|$ 逐张量动态归一化，按源码绘制）。(a) 不同 $\alpha$ 下的输入-输出曲线；(b) 增益 $f_\alpha(x)/x$：$\alpha>0$ 时小幅值激活增益低于 1（压缩、趋向三次饱和），$\alpha<0$ 时增益高于 1（扩张）。该图支持的结论：正 $\alpha$ 压缩系统性削弱中小幅值激活所携带的信息，是模型对正 $\alpha$ 更敏感的机理来源。

### 4.2 注入位置与总体流程

非线性作用于层**输入**（$\tilde{y}=W*f_\alpha(x)+b$），权重不受修改；控制器提供全局/逐层的 $\alpha$ 设定与开关。图1 给出从数据、基线训练、非线性注入、三种 NAT 微调到统一评估的总体流程。

![图1 方法总体流程图](../../results/figures/paper_final/fig01_pipeline.png)

**图1** ActCIM-Robust 总体流程。先训练 ResNet-18 基线，再经输入端非线性注入框架模拟 CIM 激活传输非线性（实际生效 4 个深层，见 §3.1），随后以 Fixed/Random/SGR 三种 NAT 策略微调，最后在全测试集上执行 11 点 α-Sweep 并统一计算鲁棒性与校准指标。

## 5 实验设计

**数据与模型。** CIFAR-10[13]：45 000 训练 / 5 000 验证 / 10 000 测试；标准增广（随机裁剪+翻转）与逐通道归一化。骨干为 ResNet-18-CIFAR[12]（11.18 M 参数，21 个可注入 Conv/Linear 层）。

**基线训练。** SGD（momentum 0.9，weight decay 5×10⁻⁴，Nesterov），余弦退火[18]带 warmup，混合精度，deterministic 模式。基线（seed 42）共 50 epoch，最佳验证轮次 48，best val acc 94.84%，用时 23 min 55 s。

**NAT 微调。** 三种策略均从基线 checkpoint 出发，SGD lr=0.01、余弦退火、10–15 epoch（配置详见 §7）：Fixed-NAT 最佳轮次 9（用时约 5 min）；Random-NAT 与 SGR-NAT 训练 11 epoch 触发早停。

**评估。** 统一 α-Sweep 协议见 §3.2，batch size 256。主结果全部来自 **seed 42** 训练协议下的四个 checkpoint（clean / random_nat / sgr_nat / fixed_nat），Fixed-NAT 另有 seed 2026 / 3407 两个完整 sweep 用于描述性一致性检查（§8.3）。

**可追溯性。** 每个数字可回溯到唯一 CSV/JSON 行与 checkpoint；四个 checkpoint 均在本次审计中被重新加载并复评，α=0 处准确率与 CSV 逐位吻合（0.9423/0.9425*/0.9428*/0.9402；带 * 者见附录 A.2 说明）。审计同时解决了"推荐模型 seed 2026 vs 推荐 checkpoint seed 42"的口径不一致：按验证集准确率 Fixed-NAT seed 2026（94.98%）最高，但全部头条鲁棒性数字来自 seed 42 checkpoint（α=0 复评指纹 0.9402 唯一匹配）；本文统一以 seed 42 为主结果口径，seed 2026/3407 仅作一致性检查。

## 6 敏感性分析

### 6.1 方向不对称：压缩远比扩张危险

表1（§8）与图3 显示：干净基线的准确率在负 α 半轴几乎不受影响（$\alpha=-0.8$ 仍有 93.66%），而正半轴在 $\alpha\geq+0.6$ 后陡降（$+0.6$：91.13%，$+0.8$：81.25%）。方向不对称差为 −3.14 pp（负号表示正 α 侧更差）。机理上（图2b），正 α 将中小幅值激活的增益压至 $1-\alpha$，深层特征的判别信息被系统性抹除；负 α 虽然放大激活（在激活幅值上造成的相对 L2 扰动甚至更大，见 §6.3），但保序性更好，对 argmax 决策的破坏较小。Fixed-NAT 训练后不对称差几乎归零（+0.0007），说明该敏感方向可以通过训练期对齐消除。

### 6.2 层敏感性：现有口径下无法给出可靠排序

原计划通过单层注入（$\alpha=\pm0.4$，逐层开启）对 21 层排序。审计发现该实验存在两点退化：(1) 控制器命名碰撞使 21 条登记名实际映射到 4 个唯一生效层，同组名称的数值完全重复；(2) 仅使用单个 128 样本批，精度分辨率 1/128≈0.78 pp，而实测最大准确率变化恰为 ±0.78 pp（1 个样本翻转）。图4 如实呈现原始 CSV，但本文明确**不将其作为逐层敏感性证据**；层间差异的有效证据由 §6.3 的误差累积分析提供。

![图4 层敏感性排序图](../../results/figures/paper_final/fig04_layer_sensitivity.png)

**图4** 单层注入 $\alpha=\pm0.4$ 时相对无注入的准确率下降（基线模型，固定 128 样本批，分辨率 0.78 pp）。由于控制器名称碰撞，21 条登记名映射到 4 个唯一生效层，组内数值重复、最大差异仅 ±0.78 pp。该图支持的结论：在该退化口径与样本量下单层敏感性差异微弱，不足以给出可靠排序（详见审计说明）。

### 6.3 误差沿深度逐层累积

对基线模型全局注入 $\alpha=\pm0.4$，用 activation hook 逐层对比扰动前后的输出激活（128 样本批）。图5 显示：前 17 层（未被启用）误差恒为 0；自 layer4.0.downsample.0 起相对 L2 误差跳升至 0.386，经 layer4.1.conv1（−α:0.531 / +α:0.442）、layer4.1.conv2（0.843/0.588），至 fc 输入处达 −α:0.978 / +α:0.605。两点观察：(1) 误差沿深度**逐级放大**，深层注入足以造成接近整体幅值的激活偏移；(2) 负 α 造成的激活幅值扰动**大于**正 α（fc 处标准差比 1.97 vs 0.41），却对应更小的准确率损失——进一步佐证 §6.1 的结论：决定精度损失的不是激活扰动幅度，而是压缩对判别信息的破坏。此外 fc 输入处正 α 的符号翻转率（10.2% vs 负 α 3.4%）与饱和比差异（13.4% vs 53.4%）与该机理一致。

![图5 误差逐层累积图](../../results/figures/paper_final/fig05_error_accumulation.png)

**图5** 全局注入 $\alpha=\pm0.4$ 时各层输出激活相对干净前向的相对 L2 误差（基线模型，128 样本批）。误差自 layer4.0.downsample.0 起才非零并沿深度快速放大，与"实际仅 4 个深层生效"的实证核验完全一致。该图支持的结论：非线性误差在深层被逐级放大，且负 α 的激活幅值扰动大于正 α，但正 α 因压缩判别信息而导致更大的准确率损失。

## 7 方法：四种训练策略

### 7.1 Clean 基线

标准训练，不注入任何非线性；作为对照与全部 NAT 微调的初始化。

### 7.2 Random-NAT：随机强度注入

每个训练 batch 采样一个全局 $\alpha\sim U(-0.5,+0.5)$（按训练配置与源码 `training/random_nat.py`；早期草稿误写为 ±0.8），对全部生效层统一注入后正常反传交叉熵损失。动机是让模型经历随机的多方向失真以获得区间鲁棒性。最佳验证准确率 94.82%，但出现在**第 0 个微调 epoch**（详见 §9.2）。

### 7.3 SGR-NAT：灵敏度引导逐层随机注入

（Sensitivity-Guided Random NAT，`training/sgr_nat.py`）逐层维护基于灵敏度评分的注入概率（min-max 归一化到 $[p_{\min},p_{\max}]$），每 batch 按概率抽取层子集并从 $(0,\alpha_{\mathrm{global}}]$ 采样各层 α（$\alpha_{\mathrm{global}}$ 由课程调度器随 epoch 递增）；损失为扰动前向交叉熵、干净前向交叉熵（权重 0.25）与二者 logit 间 KL 一致性项（λ=0.5，温度 2.0，干净分支 detach）之和[15]。最佳验证准确率 94.68%，同样出现在第 0 个 epoch。

### 7.4 Fixed-NAT：固定正强度注入

训练全程固定 $\alpha=+0.4$（`configs/fixed_nat.yaml`），即直接把模型放到"最危险方向的中等强度"工作点上微调。最佳验证准确率 94.90%（epoch 9）。该策略隐含假设：正 α 压缩是主导失效方向（§6.1），针对性适应压缩即可覆盖绝大部分风险区间。

## 8 结果

### 8.1 四方法总体对比

表1 汇总核心指标（全部取自 `fixed_nat_comparison.json`，并经 checkpoint 复评核验）；图3 给出完整 Accuracy-α 曲线，图6 与图7 分别给出最差准确率与 Clean-AURC 权衡视图。

**表1** 四种方法的鲁棒性指标（CIFAR-10 测试集，11 点 α-Sweep，seed 42）

| 方法 | AURC | AURC⁺ | AURC⁻ | Worst Acc | worst α | α=0 Acc | Asym Gap | Mean Pert. |
|---|---|---|---|---|---|---|---|---|
| Clean 基线 | 0.9283 | 0.9128 | 0.9399 | 81.25% | +0.8 | 94.23% | −0.0314 | 92.43% |
| Random-NAT | 0.9281 | 0.9116 | 0.9405 | 81.30% | +0.8 | 94.25% | −0.0331 | 92.42% |
| SGR-NAT | 0.9290 | 0.9141 | 0.9400 | 82.06% | +0.8 | 94.28% | −0.0300 | 92.52% |
| **Fixed-NAT (+0.4)** | **0.9374** | **0.9382** | 0.9357 | **91.79%** | +0.8 | 94.02% | **+0.0007** | **93.65%** |

**表2** 各 α 点 Top-1 准确率（%，seed 42）

| α | Clean | Random-NAT | SGR-NAT | Fixed-NAT |
|---|---|---|---|---|
| −0.8 | 93.66 | 93.84 | 93.77 | 93.23 |
| −0.6 | 93.91 | 93.97 | 93.92 | 93.42 |
| −0.4 | 94.03 | 94.05 | 93.99 | 93.59 |
| −0.2 | 94.18 | 94.21 | 94.18 | 93.86 |
| −0.1 | 94.22 | 94.28 | 94.27 | 93.96 |
| 0.0 | 94.23 | 94.25 | 94.28 | 94.02 |
| +0.1 | 94.23 | 94.12 | 94.15 | 94.13 |
| +0.2 | 94.11 | 94.02 | 94.08 | 94.15 |
| +0.4 | 93.59 | 93.46 | 93.56 | **94.27** |
| +0.6 | 91.13 | 90.91 | 91.26 | **94.07** |
| +0.8 | 81.25 | 81.30 | 82.06 | **91.79** |

![图3 四种方法Accuracy-Alpha曲线](../../results/figures/paper_final/fig03_alpha_sweep.png)

**图3** 四种方法在 $\alpha\in[-0.8,+0.8]$ 11 点上的 Top-1 测试准确率（10 000 张全测试集，seed 42 checkpoint，均已复评核验）。Clean/Random-NAT/SGR-NAT 三条曲线几乎重合并在 $\alpha=+0.8$ 处跌至 81.25%/81.30%/82.06%；Fixed-NAT(+0.4) 全程平坦，最差点 91.79%。该图支持的结论：Fixed-NAT 显著抬升最差准确率（+10.54 pp），SGR-NAT 与 Random-NAT 表现接近。

三个要点：

1. **Random-NAT 与 SGR-NAT 未取得实质增益，二者接近。** Random-NAT 的 AURC（0.9281）甚至略低于基线（0.9283），SGR-NAT 略高（0.9290）；最差准确率的提升分别只有 +0.05 pp 与 +0.81 pp。二者与基线的全部指标差异都在千分位量级。
2. **Fixed-NAT 全面占优于正半轴。** AURC⁺ 从 0.9128 升至 0.9382，$\alpha=+0.6$ 处准确率保持 94.07%（基线 91.13%），$\alpha=+0.8$ 处 91.79%（基线 81.25%）。
3. **代价可控且可解释。** Fixed-NAT 在负半轴轻微让步（AURC⁻ 0.9357 vs 0.9399，$\alpha=-0.8$ 处 93.23% vs 93.66%），Clean Accuracy 降 0.21 pp；其 Accuracy-α 曲线峰值移动到 $\alpha=+0.4$（94.27%）——恰为训练工作点，符合"训练-部署工作点对齐"的预期。

![图6 Worst Accuracy对比图](../../results/figures/paper_final/fig06_worst_accuracy.png)

**图6** 四种方法在 11 点 α-Sweep 上的最差准确率（均出现在 $\alpha=+0.8$）。数值取自 fixed_nat_comparison.json 并经 checkpoint 复评核验（单一 seed 42 训练协议，非多种子统计量）。该图支持的结论：Fixed-NAT 将最差准确率提高 10.54 pp，是唯一实质改善最坏情形的方案。

![图7 Clean Accuracy-AURC权衡图](../../results/figures/paper_final/fig07_tradeoff.png)

**图7** 各方法 α=0 准确率（横轴）与 AURC（纵轴）。Fixed-NAT 以 0.21 pp 的 Clean 代价换取 AURC 0.9283→0.9374；Random-NAT/SGR-NAT 几乎停留在基线位置。该图支持的结论：Fixed-NAT 的鲁棒性收益远大于其微小的干净精度损失，位于权衡前沿。

### 8.2 Fixed-NAT 核心结果

以 seed 42 checkpoint（本项目全部头条数字的实证来源，附录 A.2）计：

- 最差准确率：81.25% → **91.79%**（+10.54 pp，$\alpha=+0.8$）
- AURC：0.9283 → **0.9374**；AURC⁺：0.9128 → 0.9382
- Clean Accuracy：**94.02%**（基线 94.23%，−0.21 pp）
- 方向不对称差：−0.0314 → **+0.0007**（正负两侧风险几乎对齐）
- 平均扰动准确率：92.43% → 93.65%

### 8.3 多训练种子的一致性（描述性）

Fixed-NAT 拥有三个训练种子（42/2026/3407）的完整 11 点 sweep（评估协议一致）。三者最差准确率 91.79%/91.36%/91.60%（均在 $\alpha=+0.8$，极差 0.43 pp），AURC 0.9374/0.9387/0.9386，α=0 准确率 94.02%/94.29%/94.23%。方向与量级完全一致（图10）。**需要明确**：对比方法（Clean/Random/SGR）各仅 1 个训练种子、Fixed-NAT 仅 n=3 且共享同一评估种子，本文因此不做显著性检验、不宣称多随机种子统计显著性；上述数字仅构成描述性一致性证据。

![图10 Fixed-NAT多种子一致性](../../results/figures/paper_final/fig10_multiseed.png)

**图10** Fixed-NAT 三个训练种子的完整 α-Sweep。三条曲线在全区间高度一致，最差点极差 0.43 pp。该图仅作描述性一致性检查，不构成统计显著性证据。

## 9 消融与讨论

### 9.1 训练期 α 分布的消融

三种 NAT 仅在"训练期 α 如何取值"上不同（固定正值 / 全局均匀随机 / 逐层灵敏度引导随机），可视为对训练分布的消融：

| 训练 α 分布 | Worst Acc | AURC | 解释 |
|---|---|---|---|
| 无（基线） | 81.25% | 0.9283 | — |
| $U(-0.5,+0.5)$ 全局（Random） | 81.30% | 0.9281 | 单点期望扰动弱：α 期望为 0，多数 batch 处于低失真区，训练信号被稀释 |
| 灵敏度引导逐层 $(0,\alpha_g]$（SGR） | 82.06% | 0.9290 | 正向偏置带来轻微增益，但逐层随机+概率注入进一步稀释单点强度 |
| 恒定 $+0.4$（Fixed） | **91.79%** | **0.9374** | 训练分布质量全部集中在最危险方向的中等强度点 |

结论：在失效方向明确（正 α 压缩）且部署期 α 近似恒定的场景下，**将训练预算集中于该方向的单一工作点**优于摊薄到整个区间。这与"随机噪声注入普遍有效"的直觉相反，根源在于本扰动是确定性的方向性失真而非零均值噪声。

### 9.2 best_epoch=0 现象

Random-NAT 与 SGR-NAT 的最佳验证准确率均出现在第 0 个微调 epoch（即几乎未更新时），此后验证精度不再超过起点并在 11 epoch 触发早停——表明二者的训练信号不足以在保持干净精度的同时改善鲁棒性，其最终 checkpoint 实质上非常接近基线；这与表1 中二者各项指标与基线的千分位差异互相印证。Fixed-NAT 的最佳轮次为 9，说明其确实学到了新的表征。

### 9.3 注入范围偏差的影响

§3.1 的 4/21 层实证意味着：(1) 本文全部结论适用于"深层激活非线性"扰动模型，扰动强度约为全层注入的六成（logit 相对扰动 0.59 vs 0.98）；(2) 由于训练与评估同口径，四方法排序不受影响；(3) 若修复命名碰撞后改为真全层注入，基线在 $\alpha=+0.8$ 的下跌预计更剧烈，Fixed-NAT 的相对优势方向预计保持，但具体数值需重新实验确认（本文不外推）。

## 10 校准分析

图8 给出 ECE-α 曲线，图9 给出三个代表性工作点的可靠性图。核心观察：

1. **负 α 侧：轻度过自信。** 各方法平均置信度维持约 0.99，ECE 在 0.05–0.07（clean，$\alpha=-0.8$ 时 ECE=0.056、mean conf=0.992）。
2. **正 α 强区：置信度崩溃型欠自信。** $\alpha=+0.8$ 时 clean 基线平均置信度崩溃至 0.252，而准确率仍有 81.25%，ECE 高达 0.560；38.07% 的样本落入置信度 <0.20 的两个分箱，其中第 2 箱（0.067–0.133）3 750 个样本的经验准确率却有 59.5%。这与常见的"过自信失准"[14]方向相反：压缩使深层激活与 logit 幅值整体缩小，softmax 分布趋于均匀，模型"变得不敢确定"而非"错误地确定"。
3. **Fixed-NAT 显著缓解但未消除失准。** $\alpha=+0.8$ 时其平均置信度回升至 0.482、准确率 91.79%，ECE 降至 0.436（补算面板，附录 A.4）；$\alpha=+0.4$（训练工作点）处 ECE 仅 0.028，为全表最佳。安全含义：Fixed-NAT 下游若按置信度做拒识/降级决策，其保守方向（欠自信）比基线的负 α 侧过自信更安全，但强失真区的置信度数值仍不可直接当作概率使用，需温度缩放等后校准[14]。

![图8 ECE-Alpha曲线](../../results/figures/paper_final/fig08_ece_alpha.png)

**图8** 四种方法在 α-Sweep 各点的 15-bin ECE。负 α 侧模型保持约 0.99 平均置信度（轻度过自信）；$\alpha=+0.8$ 时 clean/random/sgr 平均置信度崩溃至约 0.25 而准确率仍有 81–82%，形成严重欠自信（ECE 0.55–0.56）；Fixed-NAT 平均置信度 0.48、ECE 0.436。该图支持的结论：强正非线性引起的失准是置信度崩溃型欠自信，Fixed-NAT 显著减轻但未完全消除。

![图9 可靠性图](../../results/figures/paper_final/fig09_reliability.png)

**图9** 15 等宽置信度区间可靠性图。(a) Clean 基线 α=0 接近对角线（ECE=0.033）；(b) $\alpha=+0.8$ 时置信度整体坍缩，柱体位于对角线上方（严重欠自信，ECE=0.560）；(c) Fixed-NAT 在 $\alpha=+0.8$ 时仍偏欠自信但明显更贴近对角线（ECE=0.436）。(c) 面板由 seed 42 checkpoint 按同一 15-bin 协议补算（仅评估）。

## 11 局限性

1. **注入范围与设计意图不符。** 实际生效层为深层 4 层（4/21），所有结论限定于该扰动模型；全层注入下的定量结论需修复控制器命名碰撞后重新实验（§3.1、§9.3）。
2. **统计强度有限。** 除 Fixed-NAT（n=3，描述性）外各方法仅单一训练种子；未做置信区间与显著性检验，头条数字为单次训练协议的结果（§8.3）。
3. **层敏感性分析退化。** 单层敏感性实验受名称碰撞与 128 样本分辨率（0.78 pp）双重限制，无法给出可靠逐层排序（§6.2）；SGR-NAT 的灵敏度先验因此建立在弱证据上，这可能是其表现平庸的原因之一。
4. **评估对 batch 组成轻微敏感。** $m=\max|x|$ 逐张量归一化使 $\alpha\neq0$ 的结果依赖评估 batch 划分（复评偏差 ≤0.23 pp，附录 A.2）；跨环境复现时须固定 batch size=256 与顺序遍历。
5. **场景覆盖。** 仅 CIFAR-10 / ResNet-18、单一非线性函数族、无真实 CIM 硬件在环验证；非线性与器件噪声、量化误差的耦合效应未研究。
6. **校准结论基于单一失真族。** 欠自信崩溃是否在其他非线性形态（如非对称饱和）下同样出现，有待验证。

## 12 结论

本文以可参数化三次非线性为载体，建立了存算一体激活非线性下的可追溯鲁棒性评估协议，并系统对比了三种非线性感知训练策略。主要结论：(1) 失真方向高度不对称——正 α 激活压缩是主导失效方向，其危险性远大于负 α 扩张，且该不对称源于判别信息的破坏而非激活扰动幅度；(2) 在部署期失真近似恒定的设定下，把训练预算集中于最危险方向单一工作点的 Fixed-NAT(+0.4) 以 0.21 pp 干净精度代价将最差准确率提升 10.54 pp（81.25%→91.79%）、AURC 提升至 0.9374，全面优于随机化的 Random-NAT 与灵敏度引导的 SGR-NAT（二者与基线接近）；(3) 强压缩引发置信度崩溃型欠自信，Fixed-NAT 将 ECE 从 0.560 降至 0.436 并使训练工作点处校准最优。后续工作将修复注入范围缺陷并在真全层注入、多种子重复、真实 CIM 硬件闭环下检验上述结论的外推性。

## 参考文献

正文采用统一编号引用；BibTeX 条目见 `references.bib`。凡未能逐字段核实的信息以"（待核验）"标注，未编造任何作者、题目、期刊、DOI 或页码。

[1] A. Shafiee, A. Nag, N. Muralimanohar, R. Balasubramonian, J. P. Strachan, M. Hu, R. S. Williams, V. Srikumar. ISAAC: A Convolutional Neural Network Accelerator with In-Situ Analog Arithmetic in Crossbars. In Proc. 43rd ACM/IEEE International Symposium on Computer Architecture (ISCA), 2016, pp. 14–26.

[2] P. Chi, S. Li, C. Xu, T. Zhang, J. Zhao, Y. Liu, Y. Wang, Y. Xie. PRIME: A Novel Processing-in-Memory Architecture for Neural Network Computation in ReRAM-Based Main Memory. In Proc. 43rd ACM/IEEE International Symposium on Computer Architecture (ISCA), 2016（页码待核验）.

[3] P. Yao, H. Wu, B. Gao, J. Tang, Q. Zhang, W. Zhang, J. J. Yang, H. Qian. Fully hardware-implemented memristor convolutional neural network. Nature, 2020, 577: 641–646.

[4] W. Wan, R. Kubendran, C. Schaefer, S. B. Eryilmaz, W. Zhang, D. Wu, S. Deiss, P. Raina, H. Qian, B. Gao, S. Joshi, H. Wu, H.-S. P. Wong, G. Cauwenberghs. A compute-in-memory chip based on resistive random-access memory. Nature, 2022, 608: 504–512.

[5] V. Sze, Y.-H. Chen, T.-J. Yang, J. S. Emer. Efficient Processing of Deep Neural Networks: A Tutorial and Survey. Proceedings of the IEEE, 2017, 105(12): 2295–2329.

[6] S. Ambrogio, P. Narayanan, H. Tsai, R. M. Shelby, I. Boybat, C. di Nolfo, S. Sidler, M. Giordano, M. Bodini, N. C. P. Farinha, B. Killeen, C. Cheng, Y. Jaoudi, G. W. Burr. Equivalent-accuracy accelerated neural-network training using analogue memory. Nature, 2018, 558: 60–67.

[7] V. Joshi, M. Le Gallo, S. Haefeli, I. Boybat, S. R. Nandakumar, C. Piveteau, M. Dazzi, B. Rajendran, A. Sebastian, E. Eleftheriou. Accurate deep neural network inference using computational phase-change memory. Nature Communications, 2020, 11: 2473.

[8] A. F. Murray, P. J. Edwards. Enhanced MLP performance and fault tolerance resulting from synaptic weight noise during training. IEEE Transactions on Neural Networks, 1994, 5(5): 792–802.

[9] C. Zhou, P. Kadambi, M. Mattina, P. N. Whatmough. Noisy Machines: Understanding Noisy Neural Networks and Enhancing Robustness to Analog Hardware Errors Using Distillation. arXiv:2001.04974, 2020（条目待核验）.

[10] Y. Bengio, N. Léonard, A. Courville. Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation. arXiv:1308.3432, 2013.

[11] M. Courbariaux, Y. Bengio, J.-P. David. BinaryConnect: Training Deep Neural Networks with Binary Weights during Propagations. In Advances in Neural Information Processing Systems (NeurIPS), 2015.

[12] K. He, X. Zhang, S. Ren, J. Sun. Deep Residual Learning for Image Recognition. In Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770–778.

[13] A. Krizhevsky. Learning Multiple Layers of Features from Tiny Images. Technical Report, University of Toronto, 2009.

[14] C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger. On Calibration of Modern Neural Networks. In Proc. 34th International Conference on Machine Learning (ICML), PMLR 70, 2017, pp. 1321–1330.

[15] G. Hinton, O. Vinyals, J. Dean. Distilling the Knowledge in a Neural Network. arXiv:1503.02531, 2015.

[16] M. P. Naeini, G. F. Cooper, M. Hauskrecht. Obtaining Well Calibrated Probabilities Using Bayesian Binning. In Proc. 29th AAAI Conference on Artificial Intelligence (AAAI), 2015, pp. 2901–2907.

[17] G. W. Brier. Verification of Forecasts Expressed in Terms of Probability. Monthly Weather Review, 1950, 78(1): 1–3.

[18] I. Loshchilov, F. Hutter. SGDR: Stochastic Gradient Descent with Warm Restarts. In Proc. International Conference on Learning Representations (ICLR), 2017.

## 附录 A 结果可追溯性与审计摘要

### A.1 头条数字与数据源对照

| 数字 | 来源文件 | 备注 |
|---|---|---|
| Clean 94.23% / Worst 81.25% / AURC 0.9283 | `results/post_training/all_methods_alpha_sweep.csv`、`fixed_nat_comparison.json` | baseline seed 42 |
| Fixed-NAT 94.02% / 91.79% / 0.9374 | `results/post_training/fixed_nat_alpha_sweep.csv`、`fixed_nat_comparison.json` | seed 42（A.2 实证锁定） |
| Random/SGR 全指标 | `all_methods_alpha_sweep.csv` | seed 42 |
| 多种子 sweep | `results/post_training/multi_seed/fixed_nat_seed_{2026,3407}_alpha_sweep.csv` | 完整 11 点 |
| 层敏感性 / 误差累积 | `results/analysis/layer_sensitivity_ranked.csv`、`layer_error_accumulation.csv` | 128 样本批 |
| 校准分箱 | `results/post_training/calibration/*_bins.csv` | Fixed-NAT 两个分箱文件为本次补算（A.4） |
| 训练摘要 | 各 `results/*/seed_*/summary.json` | val acc、best epoch、用时 |

### A.2 checkpoint 复评（seed 归属实证）

脚本 `scripts/paper/verify_checkpoints_vs_csv.py` 重新加载 4 个 checkpoint 在全测试集复评（结果 `reports/final/checkpoint_reverification.json`）。α=0 处与 CSV 逐位吻合且四个指纹互不相同：baseline 0.9423、fixed_nat seed42 **0.9402**、seed2026 0.9429、seed3407 0.9423——实证锁定 `fixed_nat_alpha_sweep.csv`（含 94.02%/91.79%）来自 **seed 42** checkpoint，解决"推荐模型 seed 2026（val acc 94.98% 最高）vs 推荐 checkpoint seed 42"的口径不一致。α=±0.8 处复评偏差 ≤0.23 pp，源于 $m=\max|x|$ 逐张量归一化对评估 batch 划分（256 vs 500）的敏感性。表1/表2 正文数字一律采用原始 CSV 值。

### A.3 注入范围实证

脚本 `scripts/paper/verify_injection_scope.py`：ResNet-18-CIFAR 共 21 个 Conv/Linear，全部被 wrap，但控制器字典仅 4 个唯一键（'0','conv1','conv2','fc'）；`enable_all()` 后实际启用 `fc`、`layer4.0.downsample.0`、`layer4.1.conv1`、`layer4.1.conv2`。α=+0.8 下 logit 相对 L2 扰动：4 层口径 0.591，真全层口径 0.983。

### A.4 Fixed-NAT 校准分箱补算

脚本 `scripts/paper/compute_fixed_nat_calibration.py`（仅评估，不训练；协议与 `scripts/calibration_audit.py` 一致：15 等宽分箱、batch 256）。补算结果 α=0：acc 0.9402 / ECE 0.0496 / mean conf 0.9896，与 `fixed_nat_alpha_sweep.csv` 逐位一致，交叉验证了补算口径的正确性；α=+0.8：acc 0.9180 / ECE 0.4360 / mean conf 0.4821。

### A.5 与早期草稿的差异更正

1. AURC 定义由"算术平均"更正为梯形积分归一化（§3.3）；2. 归一化因子由"固定 m=10"更正为逐张量动态 $\max|x|$（§4.1）；3. Random-NAT 训练分布由"±0.8"更正为 $U(-0.5,+0.5)$（§7.2）；4. "全 21 层注入"更正为"实际生效 4 个深层"（§3.1）；5. 强正 α 失准方向由"过自信"更正为"置信度崩溃型欠自信"（§10）；6. 多种子表述从"NOT RUN"更正为"存在完整 sweep，但仅作描述性检查"（§8.3）。
