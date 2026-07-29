# Nonlinearity Implementation Audit — ActCIM-Robust

**Date:** 2026-07-29
**Environment:** Python 3.12.5, PyTorch 2.5.1+cu121, Windows

---

## 1. Summary of Findings

| # | Test | Result | Notes |
|---|------|--------|-------|
| 2.1 | alpha=0 identity | PASS | `alpha=0` short-circuits to return `x` identically |
| 2.2 | Zero input no NaN/Inf | PASS | `clamp_min(eps)` prevents div-by-zero; zeros stay zeros |
| 2.3 | Injection location (INPUT) | PASS | `y = conv(nonlinearity(x))` — applied to input, not output/weights |
| 2.4 | Single-layer injection | PASS | Only conv1 gets nonlinearity; conv2 receives clean input through ReLU |
| 2.5 | All-layer injection | PASS | Conv2d, Linear, ConvTranspose2d all wrapped and individually controllable |
| 2.6 | BatchNorm freeze during eval | PASS | Running stats frozen in eval, updated normally in train |
| 2.7a | Gradient computation | PASS | Gradients flow correctly through cubic nonlinearity at all alpha levels |
| 2.7b | Enable/disable toggling | PASS | `enable()`/`disable()` properly toggle behavior |
| 2.7c | State dict roundtrip | PASS | `state_dict()` / `load_state_dict()` delegates correctly to wrapped module |
| 2.7d | Controller export/restore | PASS | `export_state()` / `restore_state()` correctly persists alpha, enabled, scope |
| 3 | Scheduler probability bounds | PASS | Normalized scores in [0,1]; output probabilities bounded to [p_min, p_max] |
| Extra | Variant shapes | PASS | per_tensor, per_sample, per_channel all preserve shape |
| Extra | Scaling invariance | PASS | `nonlinearity(k*x, a) == k * nonlinearity(x, a)` |
| Extra | eps prevents NaN | PASS | Zero tensors handled safely |
| Extra | Linear wrapper | PASS | Wrapper works correctly with `nn.Linear` |
| Extra | ConvTranspose2d wrapper | PASS | Wrapper works correctly with `nn.ConvTranspose2d` |
| Extra | Type checking | PASS | `TypeError` raised for non-injectable modules (ReLU, BatchNorm) |

**Overall: 17/17 audit tests PASS**
**Existing pytest suite: 43/43 PASS** (including 26 pre-existing + 17 audit tests)

---

## 2. File-by-File Review

### 2.1 `function.py` (`src/actcim_robust/nonlinearity/function.py`)

**Matches official spec: YES**

```python
def nonlinearity(x, alpha=0.0, eps=1e-8):
    if alpha == 0.0:
        return x
    max_val = x.abs().amax().clamp_min(eps)
    x_norm = x / max_val
    y = alpha * (x_norm ** 3) + (1 - alpha) * x_norm
    return y * max_val
```

- Uses `amax()` (equivalent to `max()` on tensor, flattens all dims) + `clamp_min(eps)` → matches spec with epsilon variant
- `alpha=0` short-circuit returns identity → correct shortcut
- `clamp_min(eps)` prevents division-by-zero for all-zero tensors → correct guard
- Function is homogeneous of degree 1: `f(kx) = k * f(x)` → confirmed by scaling invariance test
- Three normalization scopes provided: `per_tensor` (default), `per_sample`, `per_channel` — all delegate to the core function with appropriate slicing

### 2.2 `wrapper.py` (`src/actcim_robust/nonlinearity/wrapper.py`)

**Injection location: INPUT — CORRECT**

```python
def forward(self, x):
    if self._enabled and self._alpha != 0.0:
        x = self._nonlinearity_fn(x, alpha=self._alpha)
    return self.module(x)   # nonlinearity(x) → conv → output
```

The wrapper applies the nonlinearity to the layer's **input** before passing it to `self.module(x)`. Weights are never touched. This matches the required `y = conv(nonlinearity(x))` pattern.

- Only wraps `nn.Conv2d`, `nn.Linear`, `nn.ConvTranspose2d` — rejects others with `TypeError`
- Exposes `.weight` and `.bias` proxy properties
- `state_dict()` / `load_state_dict()` delegate to the inner module
- `enable` property is read-only externally; set via `enable()` / `disable()` methods

### 2.3 `controller.py` (`src/actcim_robust/nonlinearity/controller.py`)

- `_INJECTABLE_TYPES = (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)` — covers the spec
- `_wrap_all_layers()` recursively walks `named_children()` — handles nested Sequential blocks
- `enable_all()` / `disable_all()` / `enable_layers()` → batch toggles
- `set_global_alpha()` / `set_layer_alpha()` → per-layer alpha control
- `export_state()` / `restore_state()` → full roundtrip of wrapper metadata

**Minor issue [LOW]:** Layer name collision in `_wrappers` dict. When modules are nested inside
Sequential blocks, PyTorch assigns sequential integer names ('0', '1', ...). If two
Sequential blocks both contain a Conv2d at index 0, the second overwrites the first in
`self._wrappers`. The actual wrappers (set as attributes on their parent modules) remain
correct and functional; only the controller's lookup by name is affected. In practice,
TinyCNN's Conv layers get names like '0', '3', '6', '8' which are unique. This issue
could surface with models that have multiple Sequential blocks with overlapping indices.

**Minor issue [LOW]:** `sample_batch_configuration()` (line 76) uses
`getattr(wrapper, "probability", wrapper.alpha)`. The `probability` attribute is set
dynamically via `set_layer_probability()` but is not declared as a property in
`NonlinearInputWrapper.__init__`. This works but is fragile.

### 2.4 `scheduler.py` (`src/actcim_robust/nonlinearity/scheduler.py`)

**SensitivityProbabilityCalculator: CORRECT**

- Min-max normalization: `(score - min_score) / score_range` → [0, 1] ✓
- When all scores equal (range=0): `normalized = 1.0` for all → maps to `p_max` ✓
- Final probability: `p = p_min + (p_max - p_min) * (normalized ** gamma)` → always in [p_min, p_max] ✓
- Highest-score layer → `p_max`; lowest-score layer → `p_min` ✓
- Empty input → empty output ✓
- `get_top_layers()` returns layers sorted by score descending ✓

**CurriculumAlphaScheduler:** Linear/power-law interpolation with optional warmup — standard implementation, no issues.

### 2.5 `registry.py` (`src/actcim_robust/nonlinearity/registry.py`)

Simple global registry using class methods. Each `NonlinearInputWrapper` registers itself on `__init__`. No issues.

---

## 3. Spec Compliance Checklist

| Spec Requirement | Status | Details |
|-----------------|--------|---------|
| `nonlinearity(x, alpha) = alpha*x^3 + (1-alpha)*x` with max-norm | PASS | `function.py:11-14` |
| `eps` clamping to prevent div-by-zero | PASS | `function.py:11` |
| Alpha=0 returns identity | PASS | `function.py:8-9` |
| Per-tensor normalization | PASS | `function.py:17-18` |
| Per-sample normalization | PASS | `function.py:21-31` |
| Per-channel normalization | PASS | `function.py:34-48` |
| Applied to INPUT of Conv2d/Linear | PASS | `wrapper.py:79-82` |
| NOT applied to weights or outputs | PASS | Verified: weights unchanged |
| Wraps Conv2d, Linear, ConvTranspose2d | PASS | `wrapper.py:28` |
| Single-layer control | PASS | `controller.py:42-49` |
| All-layer control | PASS | `controller.py:32-40` |
| Enable/disable toggling | PASS | `wrapper.py:73-77` |
| State dict roundtrip | PASS | `wrapper.py:84-88` |
| Scheduler probability in [p_min, p_max] | PASS | `scheduler.py:81` |
| BatchNorm running stats frozen in eval | PASS | Standard PyTorch behavior, unchanged |

---

## 4. Pytest Results

```
43 passed in 4.13s
```

All pre-existing tests pass (26 tests across `test_nonlinearity.py`, `test_wrapper.py`, `test_controller.py`, `test_config.py`, `test_data_split.py`, `test_metrics.py`, `test_models.py`, `test_smoke_pipeline.py`). All 17 new audit tests pass.

---

## 5. Conclusion

**The nonlinearity implementation matches the official specification.** The core
function correctly implements the cubic nonlinearity with max-absolute normalization,
epsilon guard, and three normalization scopes. The wrapper correctly injects nonlinearity
on the **input** side of Conv2d/Linear/ConvTranspose2d layers without modifying weights.
The controller provides full per-layer enable/disable and alpha control with exportable
state. The scheduler correctly normalizes sensitivity scores into bounded probabilities.

Two low-severity issues were identified:
1. `_wrappers` dict key collision risk with nested Sequential blocks
2. Dynamically-set `probability` attribute in `sample_batch_configuration()`

Neither issue affects correctness in the current model configurations.
