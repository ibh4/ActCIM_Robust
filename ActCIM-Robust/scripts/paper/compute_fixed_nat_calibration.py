"""Compute reliability-diagram bins for the Fixed-NAT (seed 42) checkpoint at
alpha=0 and alpha=+0.8, matching the protocol of scripts/calibration_audit.py
(15 equal-width bins, batch_size=256, softmax confidence). Evaluation only,
no training. Output CSVs mirror results/post_training/calibration/*.csv.
"""
import sys, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn.functional as F

from actcim_robust.models import create_model
from actcim_robust.nonlinearity import NonlinearityController
from cifar_loader import load_cifar10_test, iterate_batches

N_BINS = 15
BATCH_SIZE = 256  # match scripts/calibration_audit.py

test_x, test_y = load_cifar10_test(ROOT / "data" / "raw")

ckpt = torch.load(ROOT / "results/fixed_nat/fixed_nat/seed_42/best.pt",
                  map_location="cpu", weights_only=False)
state = ckpt.get("model_state_dict", ckpt)
model = create_model("resnet18_cifar", num_classes=10)
model.load_state_dict(state)
controller = NonlinearityController(model)
model.eval()

out_dir = ROOT / "results" / "post_training" / "calibration"
summary = {}

for alpha, tag in ((0.0, "fixed_nat_alpha_0"), (0.8, "fixed_nat_alpha_pos_08")):
    controller.set_global_alpha(alpha)
    controller.enable_all()
    confs, correct = [], []
    t0 = time.time()
    with torch.no_grad():
        for x, y in iterate_batches(test_x, test_y, BATCH_SIZE):
            probs = F.softmax(model(x), dim=1)
            c, pred = probs.max(dim=1)
            confs.append(c)
            correct.append((pred == y).float())
    confs = torch.cat(confs).numpy()
    correct = torch.cat(correct).numpy()
    n = len(confs)

    edges = np.linspace(0.0, 1.0, N_BINS + 1, dtype=np.float32)
    rows, ece = [], 0.0
    for i in range(N_BINS):
        lo, hi = edges[i], edges[i + 1]
        if i == N_BINS - 1:
            mask = (confs >= lo) & (confs <= hi)
        else:
            mask = (confs >= lo) & (confs < hi)
        cnt = int(mask.sum())
        if cnt > 0:
            mc = float(confs[mask].mean())
            ba = float(correct[mask].mean())
            gap = ba - mc
            wgap = abs(gap) * cnt / n
            ece += wgap
        else:
            mc = ba = gap = wgap = 0.0
        rows.append((i, float(lo), float(hi), cnt, mc, ba, gap, wgap))

    csv_path = out_dir / f"{tag}_bins.csv"
    with open(csv_path, "w") as f:
        f.write("bin_index,bin_lower,bin_upper,sample_count,mean_confidence,"
                "bin_accuracy,calibration_gap,weighted_gap\n")
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")

    acc = float(correct.mean())
    summary[tag] = {"alpha": alpha, "accuracy": acc, "ece_15_bins": float(ece),
                    "mean_confidence": float(confs.mean()),
                    "elapsed_seconds": round(time.time() - t0, 1)}
    print(f"{tag}: acc={acc:.4f} ece={ece:.5f} conf={confs.mean():.5f} "
          f"({summary[tag]['elapsed_seconds']}s)", flush=True)

(out_dir / "fixed_nat_calibration_summary.json").write_text(json.dumps(summary, indent=2))
print("done")
