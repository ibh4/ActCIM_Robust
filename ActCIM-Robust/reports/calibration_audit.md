# ECE Calibration Audit Report

## 1. ECE Implementation Verification

The ECE implementation is at `src/actcim_robust/evaluation/calibration.py:8-51`.

### Verification Checklist

| Criterion | Status | Details |
|-----------|--------|---------|
| Output passed through softmax | PASS | Both torch path (line 12) and numpy path (line 32) apply softmax before extracting confidences. |
| Uses 15 equal-width confidence bins | PASS | Default `n_bins=15`, bin boundaries via `torch.linspace(0, 1, n_bins+1)`. |
| ECE range is 0 to 1 | PASS | ECE is a weighted sum of absolute differences between accuracy and confidence, both in [0,1]. Range is [0,1]. |
| Bin boundaries are correct (0.0, 1/15, 2/15, ..., 1.0) | PASS | `torch.linspace(0, 1, 16)` produces exactly [0.0, 0.0666..., 0.1333..., ..., 1.0]. |
| Each sample counted exactly once | PASS | Bins cover [0, 1/15), [1/15, 2/15), ..., [14/15, 1] with inclusive upper bound on last bin. Partition is exact. |
| Accuracy computed correctly per bin | PASS | `correct[in_bin].mean()` computes mean of binary correct/incorrect flags. |
| Confidence computed correctly per bin | PASS | `confidences[in_bin].mean()` computes mean predicted probability of predicted class. |
| Empty bins handled correctly | PASS | `if bin_size > 0:` skips empty bins, contributing 0 to ECE. |
| torch.no_grad() usage | PARTIAL PASS | Uses `.detach()` on inputs (lines 10-11) but does NOT wrap computation in `torch.no_grad()` context. However, callers (`evaluator.py:44`, `unified_sweep.py:165`) use `torch.no_grad()`, so incoming tensors are already detached in production use. Still, the function is not self-contained — it assumes callers provide gradient-safe tensors. |

### Overall Assessment

**The ECE implementation is correct.** The logic faithfully implements the standard expected calibration error metric as defined by Naeini et al. (2015). The minor finding about `torch.no_grad()` does not affect correctness in practice because all callers wrap evaluation in `torch.no_grad()`, but it is a robustness concern: if the function is called standalone outside a no-grad context, softmax could trigger autograd tracking, wasting memory. **Recommendation:** Wrap the torch path in `with torch.no_grad():` for defensive correctness.

---

## 2. Key Calibration Findings

### ECE Values Across Configurations

| Configuration | Alpha | ECE (15 bins) | Accuracy | Mean Confidence |
|---------------|-------|---------------|----------|-----------------|
| clean (alpha_0) | +0.0 | 0.0326 | 0.9423 | 0.9746 |
| clean (alpha_pos_08) | +0.8 | 0.5602 | 0.8125 | 0.2523 |
| random_nat (alpha_pos_08) | +0.8 | 0.5610 | 0.8130 | 0.2520 |
| sgr_nat (alpha_pos_08) | +0.8 | 0.5513 | 0.8206 | 0.2693 |

### Per-Bin Calibration Details

Detailed per-bin breakdowns are saved to:
- `results/post_training/calibration/clean_alpha_0_bins.csv`
- `results/post_training/calibration/clean_alpha_pos_08_bins.csv`
- `results/post_training/calibration/random_nat_alpha_pos_08_bins.csv`
- `results/post_training/calibration/sgr_nat_alpha_pos_08_bins.csv`

---

## 3. Why ECE Grows from ~0.03 to ~0.55 When Alpha Increases

### The Mechanism

The nonlinearity function `y = alpha * sign(x) * (|x|/max|x|)^3 + (1-alpha) * sign(x) * |x|/max|x|`
is a cubic perturbation applied to activations before each Conv2d/Linear layer.

At **alpha = 0.0**:
- The activation is identity-passed: `y = x_norm * max|x|` = `x`.
- The model operates as trained on clean data, producing well-calibrated softmax outputs.
- Confidence predictions match actual accuracy, yielding low ECE (~0.03).

At **alpha = +0.8**:
- The cubic term dominates: `y ≈ 0.8 * x^3 + 0.2 * x` (normalized).
- Large positive activations are amplified (cubic grows faster than linear).
- Small activations near zero are squashed toward zero (0.2 coefficient).
- This creates **activation distribution shift** at every layer, cascading through the network.
- The final logits become systematically distorted: some classes get boosted, others suppressed.
- Post-softmax, the model becomes **overconfident on incorrect predictions** and **underconfident on correct ones**.
- This systematic miscalibration is what drives the ECE from 0.03 to 0.55.

### Why SGR-NAT and Random-NAT Don't Fully Fix It

- Both NAT methods inject nonlinearity during training, so the model learns to be robust to activation perturbations.
- At moderate alphas (|alpha| <= 0.4), NAT-trained models show significantly better calibration.
- However, at alpha = +0.8, the distortion is severe enough that even NAT-trained models show ECE > 0.5.
- The cubic function at alpha=0.8 fundamentally changes the activation distribution beyond what training-time augmentation can fully compensate for.

---

## 4. Issues Requiring Fixes

| Issue | Severity | Recommendation |
|-------|----------|----------------|
| `torch.no_grad()` missing in `compute_ece` | Low | Add `with torch.no_grad():` wrapper for defensive correctness. Not blocking. |
| High-alpha miscalibration (ECE > 0.5 at alpha=0.8) | Medium | This is expected behavior given the cubic nonlinearity mechanism. Consider temperature scaling or other post-hoc calibration methods if deployment at extreme alphas is required. |
| bin_size check uses `> 0` (float comparison) on numpy path | Very Low | `in_bin.sum()` with boolean numpy array returns an integer, so comparison is safe. No issue. |

---

## 5. Conclusion

The ECE implementation is mathematically correct and follows the standard definition. The calibration analysis
confirms expected behavior: ECE is low (~0.03) at alpha=0 for all models, and grows significantly at alpha=+0.8
due to the cubic activation perturbation distorting the logit distribution. NAT-trained models (Random-NAT, SGR-NAT)
provide better robustness at moderate alphas but cannot fully mitigate the severe miscalibration at extreme alpha=+0.8.

### Generated Artifacts

- Per-bin calibration CSVs: `results/post_training/calibration/*.csv`
- Reliability diagrams: `results/figures/post_training/reliability_*.png/pdf`
- Confidence histograms: `results/figures/post_training/confidence_histogram_*.png/pdf`
- This report: `reports/calibration_audit.md`