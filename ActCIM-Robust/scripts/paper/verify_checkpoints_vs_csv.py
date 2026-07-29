"""Verify which checkpoint produced fixed_nat_alpha_sweep.csv by re-evaluating
all Fixed-NAT checkpoints (and the clean baseline) at alpha=0 and alpha=+0.8
on the full CIFAR-10 test set, using the project's own pipeline (controller.enable_all).
"""
import sys, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch

from actcim_robust.models import create_model
from actcim_robust.nonlinearity import NonlinearityController

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cifar_loader import load_cifar10_test, iterate_batches

test_x, test_y = load_cifar10_test(ROOT / "data" / "raw")

CHECKPOINTS = {
    "baseline_seed_42": ROOT / "results/baseline/seed_42/best.pt",
    "fixed_nat_seed_42": ROOT / "results/fixed_nat/fixed_nat/seed_42/best.pt",
    "fixed_nat_seed_2026": ROOT / "results/fixed_nat/fixed_nat/seed_2026/best.pt",
    "fixed_nat_seed_3407": ROOT / "results/fixed_nat/fixed_nat/seed_3407/best.pt",
}

EXPECTED = {  # from CSVs in results/post_training
    "baseline_seed_42": {"0.0": 0.9423, "0.8": 0.8125},
    "fixed_nat_seed_42": {"0.0": None, "0.8": None},  # candidate for fixed_nat_alpha_sweep.csv: 0.9402 / 0.9179
    "fixed_nat_seed_2026": {"0.0": 0.9429, "0.8": 0.9136},
    "fixed_nat_seed_3407": {"0.0": 0.9423, "0.8": 0.9160},
}

def evaluate(model, controller, alpha):
    controller.set_global_alpha(alpha)
    controller.enable_all()
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in iterate_batches(test_x, test_y, 500):
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.numel()
    return correct / total

results = {}
for name, path in CHECKPOINTS.items():
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model = create_model("resnet18_cifar", num_classes=10)
    model.load_state_dict(state)
    controller = NonlinearityController(model)
    entry = {"ckpt_epoch": ckpt.get("epoch"), "ckpt_val_acc": ckpt.get("val_acc") or ckpt.get("best_val_acc")}
    for alpha in (0.0, 0.8):
        t0 = time.time()
        acc = evaluate(model, controller, alpha)
        entry[f"acc_alpha_{alpha}"] = acc
        entry[f"sec_alpha_{alpha}"] = round(time.time() - t0, 1)
        print(f"{name} alpha={alpha}: acc={acc:.4f} ({entry[f'sec_alpha_{alpha}']}s)", flush=True)
    results[name] = entry

out_path = ROOT / "reports" / "final" / "checkpoint_reverification.json"
out_path.write_text(json.dumps({"expected_from_csv": EXPECTED, "reproduced": results}, indent=2))
print(json.dumps(results, indent=2))
