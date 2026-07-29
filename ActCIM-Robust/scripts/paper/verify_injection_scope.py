"""Empirically verify the actual nonlinearity injection scope of NonlinearityController."""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import torch
from actcim_robust.models import create_model
from actcim_robust.nonlinearity import NonlinearityController
from actcim_robust.nonlinearity.wrapper import NonlinearInputWrapper

model = create_model("resnet18_cifar", num_classes=10)
n_inject_candidates = sum(1 for m in model.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)))
ctrl = NonlinearityController(model)

wrappers_dict = ctrl.get_wrappers()
layer_names = ctrl.get_layer_names()

# all wrappers actually present in the module tree
tree_wrappers = {name: m for name, m in model.named_modules() if isinstance(m, NonlinearInputWrapper)}

ctrl.set_global_alpha(0.4)
ctrl.enable_all()
enabled = sorted([n for n, w in tree_wrappers.items() if w.enabled])
alpha_set = sorted([n for n, w in tree_wrappers.items() if w.alpha != 0.0])

out = {
    "conv_linear_modules_in_resnet18_cifar": n_inject_candidates,
    "wrappers_in_module_tree": len(tree_wrappers),
    "controller_layer_names_len": len(layer_names),
    "controller_layer_names_unique": sorted(set(layer_names)),
    "controller_wrappers_dict_keys": sorted(wrappers_dict.keys()),
    "enabled_after_enable_all": enabled,
    "n_enabled_after_enable_all": len(enabled),
    "alpha_nonzero_after_set_global_alpha": alpha_set,
}

# functional check: does output change when only "enable_all" is used?
x = torch.randn(4, 3, 32, 32)
model.eval()
with torch.no_grad():
    y_pert = model(x)
ctrl.disable_all(); ctrl.set_global_alpha(0.0)
with torch.no_grad():
    y_clean = model(x)
out["logit_relative_l2_enable_all_vs_clean"] = float((y_pert - y_clean).norm() / y_clean.norm())

# manual full injection: enable every wrapper in the tree
for w in tree_wrappers.values():
    w.alpha = 0.4
    w.enable()
with torch.no_grad():
    y_full = model(x)
out["logit_relative_l2_true_all_layers_vs_clean"] = float((y_full - y_clean).norm() / y_clean.norm())
print(json.dumps(out, indent=2, ensure_ascii=False))
