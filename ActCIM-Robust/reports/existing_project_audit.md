# ActCIM-Robust — Existing Project Audit

**Audit Date:** 2026-07-29
**Project Root:** `I:\比赛项目\存算一体高校挑战赛\ActCIM-Robust`

---

## 1. Environment

| Property | Value |
|---|---|
| Python | 3.12.5 (MSC v.1940 64-bit) |
| PyTorch | 2.5.1+cu121 |
| CUDA Available | YES |
| CUDA Version | 12.1 |
| GPU | NVIDIA GeForce RTX 4060 |
| GPU Memory | 8.59 GB (nominal 8.00 GB) |
| CPU | Intel Core i7-10700 @ 2.90GHz |
| RAM | 31.90 GB |
| OS | Windows 11 |
| Hostname | WIN-2SJ7ERJE706 |

---

## 2. Directory Structure Summary

```
ActCIM-Robust/
├── .gitignore
├── .pytest_cache/
├── CHANGELOG.md
├── PROGRESS.md
├── RUN_STATUS.md
├── SUBMISSION_CHECKLIST.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── configs/          (10 .yaml files)
├── data/
│   ├── raw/cifar-10-batches-py/  (5 data_batch + 1 test_batch + meta + readme)
│   └── splits/       (2 .npy files)
├── experiments/
├── reports/
├── results/
│   ├── analysis/     (4 .csv, 3 .json)
│   ├── baseline/seed_42/  (2 .pt, 1 .json, 1 .jsonl, 1 .json env)
│   ├── figures/      (15 .png + 15 .pdf)
│   ├── manifests/
│   ├── random_nat/random_nat/seed_42/  (2 .pt, 1 .json, 1 .jsonl, 1 .json env)
│   └── sgr_nat/sgr_nat/seed_42/  (2 .pt, 1 .json, 1 .jsonl, 1 .json env)
├── scripts/          (8 .ps1 files)
├── src/actcim_robust/
│   ├── analysis/     (5 .py + __init__)
│   ├── data/         (3 .py + __init__)
│   ├── evaluation/   (5 .py + __init__)
│   ├── models/       (3 .py + __init__)
│   ├── nonlinearity/ (5 .py + __init__)
│   ├── training/     (6 .py + __init__)
│   ├── utils/        (4 .py + __init__)
│   └── visualization/(5 .py + __init__)
└── tests/            (8 test .py + __init__)
```

---

## 3. File Counts by Type

| Type | Count | Notes |
|---|---|---|
| `.py` source files | 60 | 51 src + 9 tests (excl. `__pycache__`) |
| `.yaml` configs | 10 | base, baseline_full, baseline_fast, sgr_nat, random_nat, fixed_nat, smoke, error_accumulation, layer_sensitivity, alpha_sweep |
| `.pt` checkpoints | 6 | 2 baseline + 2 sgr_nat + 2 random_nat |
| `.pth` checkpoints | 0 | None found |
| `.csv` result files | 4 | All under `results/analysis/` |
| `.json` result/env files | 10 | 3 training summaries + 3 environments + 3 analysis + 1 batch_stats |
| `.jsonl` metric logs | 3 | 1 per training run |
| `.png` figures | 15 | All under `results/figures/` |
| `.pdf` figures | 15 | Matching set for each .png |
| `.md` reports | 5 | PROGRESS, CHANGELOG, RUN_STATUS, SUBMISSION_CHECKLIST, pytest README |
| `.ps1` scripts | 8 | setup + 7 run scripts |
| `.npy` data splits | 2 | train_indices + val_indices |

---

## 4. Checkpoint Inventory

| Method | File | Size (bytes) | SHA256 (first 16) | Epoch | Val Acc | Params | Seed | Load |
|---|---|---|---|---|---|---|---|---|
| baseline | results/baseline/seed_42/last.pt | 89,489,730 | e450cc6b... | 49 | — | 11,183,582 | 42 | OK |
| baseline | results/baseline/seed_42/best.pt | 89,489,730 | da4a9a01... | 48 | 94.84% | 11,183,582 | 42 | OK |
| sgr_nat | results/sgr_nat/sgr_nat/seed_42/last.pt | 89,489,794 | 0691a1a5... | 10 | — | 11,183,582 | 42 | OK |
| sgr_nat | results/sgr_nat/sgr_nat/seed_42/best.pt | 89,489,794 | 6e29b333... | 0 | 94.68% | 11,183,582 | 42 | OK |
| random_nat | results/random_nat/random_nat/seed_42/last.pt | 89,489,730 | 49cb043c... | 10 | — | 11,183,582 | 42 | OK |
| random_nat | results/random_nat/random_nat/seed_42/best.pt | 89,489,730 | 79ce3062... | 0 | 94.82% | 11,183,582 | 42 | OK |

All checkpoints share the same structure: `epoch`, `model_state_dict`, `metrics`, `optimizer_state_dict`, `scheduler_state_dict`.
SGR/random NAT checkpoints are slightly larger (+64 bytes) due to NAT-specific config/metadata keys in metrics.

### Training Run Summaries

| Method | Best Epoch | Best Val Acc | Total Epochs | Time | Model |
|---|---|---|---|---|---|
| baseline | 48 | 94.84% | 50 | 23m 55s | resnet18_cifar |
| sgr_nat | 0 | 94.68% | 11 (early stop) | 6m 38s | resnet18_cifar |
| random_nat | 0 | 94.82% | 11 (early stop) | 5m 31s | resnet18_cifar |

---

## 5. Data Status

### Splits
| File | Size | Shape | Status |
|---|---|---|---|
| data/splits/cifar10_train_indices.npy | 180,128 bytes | (45000,) | OK |
| data/splits/cifar10_val_indices.npy | 20,128 bytes | (5000,) | OK |
| **Total** | | **50,000** | **Verified** |

### Raw CIFAR-10
| File | Size (bytes) | Status |
|---|---|---|
| data_batch_1 | 31,035,704 | OK |
| data_batch_2 | 31,035,320 | OK |
| data_batch_3 | 31,035,999 | OK |
| data_batch_4 | 31,035,696 | OK |
| data_batch_5 | 31,035,623 | OK |
| test_batch | 31,035,526 | OK |
| batches.meta | 158 | OK |
| readme.html | 88 | OK |

All 5 training batches + test batch present and intact.

### Source Archive
| Property | Value |
|---|---|
| Path | `I:\比赛项目\存算一体高校挑战赛\cifar-10-python.tar.gz` |
| Size | 170,498,071 bytes (162.6 MB) |
| MD5 | `c58f30108f718f92721af3b95e74349a` |

---

## 6. Analysis Results (Existing)

### Layer Sensitivity (21 layers analyzed)
- Baseline clean accuracy: 95.31%
- 20 of 21 layers have sensitivity_score = 0.0078 (0.78% acc drop)
- Final FC layer has sensitivity_score = 0.0 (fully robust at ±0.4 alpha)
- Layers ranked 1–20 show identical sensitivity; FC layer ranked 21

### Alpha Sweep (baseline best.pt, alphas: -0.8 to +0.8)
| Metric | Value |
|---|---|
| Best accuracy | 94.23% (alpha=0.0, 0.1) |
| Worst accuracy | 81.61% (alpha=0.8) |
| Mean accuracy | 92.63% |
| Max accuracy drop | 0.57% (at alpha=-0.8) |
| AURC | 0.9286 |
| Pos/Neg gap | -3.04% |

### Error Accumulation
- 21 layers studied with positive and negative alpha=±0.4
- All conv layers in layer1/layer2/layer3 show zero error (relative_l2=0)
- layer4.0.downsample, layer4.1.conv1, layer4.1.conv2 show significant error
- FC layer: relative_l2 ≈ 0.98 (neg), 0.61 (pos) — highest accumulation

---

## 7. Test Results

```
============================= 26 passed in 4.14s ==============================
```

| Test File | Tests | Status |
|---|---|---|
| test_config.py | 2 | 2 passed |
| test_controller.py | 3 | 3 passed |
| test_data_split.py | 2 | 2 passed |
| test_metrics.py | 3 | 3 passed |
| test_models.py | 4 | 4 passed |
| test_nonlinearity.py | 5 | 5 passed |
| test_smoke_pipeline.py | 1 | 1 passed |
| test_wrapper.py | 6 | 6 passed |
| **Total** | **26** | **26 passed, 0 failed** |

---

## 8. Issues & Anomalies

1. **SGR NAT best_epoch = 0**: The best checkpoint was at epoch 0, suggesting the fine-tuned model did not improve over the initial weights. Training continued to epoch 10 (early stop triggered). This may indicate the sensitivity-guided NAT requires hyperparameter tuning.
2. **Random NAT best_epoch = 0**: Same pattern as SGR NAT — best result at epoch 0.
3. **SGR checkpoint size discrepancy**: SGR NAT checkpoints are 64 bytes larger than baseline/random NAT, likely due to additional NAT config keys in the checkpoint metadata.
4. **No `.pth` checkpoints**: All checkpoints use `.pt` extension only.
5. **Data directory name encoding**: The path contains Chinese characters in `比赛项目` and `存算一体高校挑战赛`, which may cause issues with some tools. All tests and operations completed successfully regardless.

---

## 9. Summary

| Metric | Value |
|---|---|
| Total checkpoints | 6 |
| Total CSV result files | 4 |
| Total JSON result files | 10 |
| Total JSONL metric files | 3 |
| Total figures (PNG) | 15 |
| Total figures (PDF) | 15 |
| Total YAML configs | 10 |
| Total Python source files | 60 |
| Total PowerShell scripts | 8 |
| CUDA available | YES (RTX 4060, 8GB) |
| Dataset CIFAR-10 | Complete (train=45000 + val=5000 split verified) |
| Dataset archive | Present (MD5 verified) |
| All tests passing | 26/26 PASSED |
| Missing files | None detected |
| Damaged files | None detected |
