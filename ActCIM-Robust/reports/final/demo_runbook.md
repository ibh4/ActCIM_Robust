# ActCIM-Robust 演示操作手册

## 环境准备

### 1. 打开项目目录

```powershell
Set-Location -LiteralPath "I:\比赛项目\存算一体高校挑战赛\ActCIM-Robust"
```

### 2. 检查目录结构

```powershell
Get-ChildItem
```

确认以下目录存在：`src/`、`configs/`、`results/`、`tests/`、`experiments/`

### 3. 激活 Python 环境

```powershell
# 确认 Python 版本（应为 3.12.5）
python --version

# 确认 PyTorch 可用
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

预期输出：
```
PyTorch 2.5.1+cu121
CUDA available: True
```

### 4. 使用项目内置的环境检查

```powershell
python -m actcim_robust.cli check-env
```

预期输出：显示 CUDA 信息、GPU 名称（RTX 4060）、显存（8GB）、PyTorch 版本等。

---

## 演示流程

### 步骤 1：运行单元测试（约 30 秒）

验证代码功能完整性：

```powershell
pytest tests/ -v --tb=short
```

预期结果：所有测试通过（PASSED）。

**可选的快速子集测试**：
```powershell
pytest tests/ -v -k "nonlinearity" --tb=short
```

---

### 步骤 2：查看已有实验结果

```powershell
# 查看 Clean 基线训练摘要
python -c "import json; d=json.load(open('results/baseline/seed_42/summary.json','r',encoding='utf-8')); print(f'val_acc: {d[\"best_val_acc\"]}%, epoch: {d[\"best_epoch\"]}, time: {d[\"total_time_str\"]}')"
```

预期输出：
```
val_acc: 94.84%, epoch: 48, time: 00:23:55
```

```powershell
# 查看 Checkpoint 清单
python -c "import json; data=json.load(open('results/manifests/checkpoint_inventory.json','r',encoding='utf-8')); [print(f'{c[\"method\"]:12s} seed={c[\"seed\"]} epoch={c[\"epoch\"]} val_acc={c[\"validation_accuracy\"]}') for c in data]"
```

预期输出：
```
baseline     seed=42 epoch=48 val_acc=94.84
baseline     seed=42 epoch=49 val_acc=None
sgr_nat      seed=42 epoch=10 val_acc=None
sgr_nat      seed=42 epoch=0  val_acc=94.68
random_nat   seed=42 epoch=10 val_acc=None
random_nat   seed=42 epoch=0  val_acc=94.82
```

---

### 步骤 3：查看 Alpha 扫描对比结果（约 10 秒）

```powershell
# 打印所有方法在各alpha下的准确率对比
python -c "
import json
data = json.load(open('results/post_training/all_methods_alpha_sweep.json', 'r', encoding='utf-8'))
print(f'{\"Alpha\":>8s}  {\"Clean\":>8s}  {\"Random-NAT\":>10s}  {\"SGR-NAT\":>8s}  {\"Fixed-NAT\":>9s}')
# Read from CSV for per-alpha data
import csv
rows = {}
with open('results/post_training/all_methods_alpha_sweep.csv', 'r', encoding='utf-8') as f:
    for r in csv.DictReader(f):
        k = (r['method'], float(r['alpha']))
        rows[k] = float(r['test_accuracy'])
alphas = sorted(set(a for _, a in rows))
for a in alphas:
    print(f'{a:8.1f}  {rows.get((\"clean\",a),0):8.4f}  {rows.get((\"random_nat\",a),0):10.4f}  {rows.get((\"sgr_nat\",a),0):8.4f}  {rows.get((\"fixed_nat\",a),0):9.4f}')
"
```

预期输出（关键行）：
```
   Alpha     Clean  Random-NAT    SGR-NAT   Fixed-NAT
    -0.8    0.9366      0.9384    0.9377     0.9323
     0.0    0.9423      0.9425    0.9428     0.9402
     0.4    0.9359      0.9346    0.9356     0.9427
     0.8    0.8125      0.8130    0.8206     0.9179
```

**关键发现**：观察 alpha=0.8 行：Clean 81.25% → Fixed-NAT 91.79%（+10.54pp）

---

### 步骤 4：查看层敏感度排名（约 5 秒）

```powershell
python -c "
import csv
with open('results/analysis/layer_sensitivity_ranked.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    print(f'{\"Rank\":>5s}  {\"Layer\":>10s}  {\"Neg_04_Drop\":>13s}  {\"Pos_04_Drop\":>13s}  {\"Score\":>8s}')
    for row in list(reader)[:8]:
        print(f'{int(row[\"rank\"]):5d}  {row[\"layer_name\"]:>10s}  {float(row[\"neg_04_accuracy_drop\"]):13.7f}  {float(row[\"pos_04_accuracy_drop\"]):13.7f}  {float(row[\"sensitivity_score\"]):8.7f}')
"
```

---

### 步骤 5：打开已生成的图表

结果图位于 `results/figures/` 目录：

```powershell
# 列出所有图表
Get-ChildItem -Path "results/figures" -Recurse -Filter "*.png" | Select-Object FullName
```

**核心图表速查**：

| 图表 | 路径 | 含义 |
|------|------|------|
| Alpha扫描主图 | `results/figures/post_training/01_accuracy_vs_alpha_all_methods.png` | 四种方法准确率曲线 |
| 最差准确率对比 | `results/figures/post_training/05_worst_case_accuracy_comparison.png` | 柱状图对比 |
| AURC对比 | `results/figures/post_training/06_aurc_all_comparison.png` | AURC柱状图 |
| Clean Acc vs Worst Acc | `results/figures/post_training/08_clean_accuracy_vs_worst_accuracy.png` | 散点图 |
| 不对称间隙 | `results/figures/post_training/10_asymmetry_gap_comparison.png` | 间隙对比 |
| 层敏感度 | `results/figures/layer_sensitivity_bar.png` | 21层敏感度排名 |

双击 `.png` 文件即可在 Windows 图片查看器中打开。

---

### 步骤 6：运行轻量级 Alpha 扫描演示（约 3 分钟）

> **警告**：此操作将运行完整 Alpha 扫描（11 个 alpha × 10,000 样本），仅用于演示实验可复现性。如果已存在结果文件，可跳过。

```powershell
python -m actcim_robust.cli alpha-sweep --checkpoint results/baseline/seed_42/best.pt
```

预期输出：11 个 alpha 值的逐行评估结果，最终显示平均准确率和最差准确率摘要。

**快速演示版本（3 个 alpha，约 45 秒）**：目前 CLI 的 `--alphas` 参数可能默认读取配置文件，如有 `--alphas` 参数可用：

```powershell
# 如果支持自定义alpha列表（需确认CLI接口）
python -m actcim_robust.cli alpha-sweep --checkpoint results/baseline/seed_42/best.pt --config configs/smoke.yaml
```

---

### 步骤 7：验证结果完整性

```powershell
python -m actcim_robust.cli validate-results
```

预期输出：所有检查项 PASSED（CSV非空、无NaN/Inf、checkpoint可加载、alpha=0存在等）。

---

### 步骤 8：生成/更新图表（可选，约 1 分钟）

如果结果 JSON/CSV 有更新，重新生成所有图表：

```powershell
python -m actcim_robust.cli build-figures
```

---

## 常见问题

### Q1: `ModuleNotFoundError: No module named 'actcim_robust'`

**原因**：项目未安装为可导入模块。

**解决**：
```powershell
pip install -e .
```

### Q2: `CUDA out of memory`

**原因**：RTX 4060 8GB 显存不足（通常不会，项目峰值仅 ~500MB）。

**解决**：
```powershell
# 减小batch_size（在config中指定）
python -m actcim_robust.cli alpha-sweep --checkpoint results/baseline/seed_42/best.pt --config configs/smoke.yaml
```

### Q3: Checkpoint 加载失败

**原因**：checkpoint 路径不正确或文件损坏。

**验证**：
```powershell
python -c "
import torch
cp = torch.load('results/baseline/seed_42/best.pt', map_location='cpu', weights_only=False)
print(f'model_state_dict keys: {len(cp[\"model_state_dict\"])}')
print(f'epoch: {cp.get(\"epoch\", \"N/A\")}')
"
```

### Q4: 测试集样本数量不正确

**验证**：所有评估应使用 10,000 个测试样本。

```powershell
python -c "
import csv
with open('results/post_training/all_methods_alpha_sweep.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    samples = set()
    for row in reader:
        samples.add(int(row['sample_count']))
    print(f'All sample counts: {samples}')  # 应只有 {10000}
"
```

### Q5: ECE 计算结果疑问

ECE 使用 15 个等宽分箱计算，公式为 $\sum_{i=1}^{15} \frac{|B_i|}{N} |\text{acc}(B_i) - \text{conf}(B_i)|$。验证：

```powershell
python -c "
import csv
ece = 0
with open('results/post_training/calibration/clean_alpha_0_bins.csv', 'r', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        if row['bin_index'] == 'ECE':
            ece = float(row['bin_lower'])
            break
print(f'Verified ECE at alpha=0: {ece}')  # 应为 ~0.0326
"
```

---

## 演示时间估算

| 步骤 | 操作 | 预计耗时 |
|------|------|---------|
| 1 | 环境检查 | 10s |
| 2 | 运行测试 | 30s |
| 3 | 查看已有结果 | 15s |
| 4 | 层敏感度 | 5s |
| 5 | 打开图表 | 20s（手动） |
| 6 | Alpha扫描演示 | 3min |
| 7 | 验证结果 | 10s |
| **总计** | | **约 4.5 分钟** |

> **建议**：面试/答辩演示时，步骤 6（Alpha扫描）可提前运行好，现场直接展示已有结果文件和图表。重跑实验作为"彩蛋"展示可复现性。

---

## 训练命令参考

```powershell
# Clean 基线训练（~24 min, 不可在演示中运行）
python -m actcim_robust.cli train --config configs/baseline_full.yaml --seed 42

# Fixed-NAT 训练（~5 min）
python -m actcim_robust.cli train --config configs/fixed_nat.yaml --seed 42 --checkpoint results/baseline/seed_42/best.pt --method fixed_nat

# Random-NAT 训练（~5.5 min）
python -m actcim_robust.cli train --config configs/random_nat.yaml --seed 42 --checkpoint results/baseline/seed_42/best.pt --method random_nat

# SGR-NAT 训练（~6.5 min）
python -m actcim_robust.cli train --config configs/sgr_nat.yaml --seed 42 --checkpoint results/baseline/seed_42/best.pt --method sgr_nat
```
