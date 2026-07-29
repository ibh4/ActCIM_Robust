from __future__ import annotations

import torch
from pathlib import Path

from ..constants import DEFAULT_ALPHAS
from ..models import create_model
from ..nonlinearity import NonlinearityController
from ..evaluation import Evaluator
from ..utils import save_json, ensure_dir, get_logger


def run_alpha_sweep(
    checkpoint_path,
    model_name,
    config,
    device,
    data_dir,
    results_dir,
    alphas=None,
):
    logger = get_logger("actcim.analysis")
    results_dir = Path(results_dir)
    analysis_dir = ensure_dir(results_dir / "analysis")

    if alphas is None:
        alphas = DEFAULT_ALPHAS[:]
    alphas = sorted(alphas)

    model = create_model(model_name, num_classes=getattr(config, "num_classes", 10))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    controller = NonlinearityController(model)

    from ..data import get_cifar10_loaders
    _, _, test_loader = get_cifar10_loaders(
        batch_size=getattr(config, "batch_size", 128),
        num_workers=getattr(config, "num_workers", 4),
        data_dir=data_dir,
    )

    evaluator = Evaluator(model, test_loader, device, controller=controller)

    sweep_results = []
    for alpha in alphas:
        logger.info(f"Alpha sweep: alpha={alpha:+.2f}")
        controller.set_global_alpha(alpha)
        controller.enable_all()
        metrics = evaluator.evaluate(compute_calibration=True)
        metrics["alpha"] = alpha
        sweep_results.append(metrics)

    controller.set_global_alpha(0.0)
    controller.enable_all()
    clean_metrics = evaluator.evaluate(compute_calibration=True)
    clean_metrics["alpha"] = 0.0
    clean_acc = clean_metrics["accuracy"]

    summary = {
        "checkpoint": str(checkpoint_path),
        "model_name": model_name,
        "alphas": alphas,
        "clean_accuracy": clean_acc,
        "results": sweep_results,
    }

    accuracies = [r["accuracy"] for r in sweep_results]
    summary["mean_accuracy"] = float(sum(accuracies) / len(accuracies))
    summary["worst_case_accuracy"] = float(min(accuracies))
    summary["best_case_accuracy"] = float(max(accuracies))

    from ..evaluation.robustness_metrics import compute_aurc, compute_positive_negative_gap, compute_accuracy_drop
    summary["aurc"] = compute_aurc(accuracies, alphas)

    pos_alphas = [a for a in alphas if a > 0]
    neg_alphas = [a for a in alphas if a < 0]
    pos_accs = []
    neg_accs = []
    for r in sweep_results:
        if r["alpha"] > 0:
            pos_accs.append(r["accuracy"])
        elif r["alpha"] < 0:
            neg_accs.append(r["accuracy"])

    if pos_accs and neg_accs:
        summary["positive_negative_gap"] = compute_positive_negative_gap(pos_accs, neg_accs)

    max_abs_alpha_idx = max(range(len(alphas)), key=lambda i: abs(alphas[i]))
    max_drop_alpha = alphas[max_abs_alpha_idx]
    max_drop_acc = accuracies[max_abs_alpha_idx]
    summary["max_accuracy_drop"] = compute_accuracy_drop(clean_acc, max_drop_acc)
    summary["max_accuracy_drop_alpha"] = max_drop_alpha

    save_json(summary, analysis_dir / "alpha_sweep_summary.json")

    csv_path = analysis_dir / "alpha_sweep.csv"
    import csv
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["alpha", "accuracy", "loss", "ece", "mean_confidence", "nll", "brier"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in sweep_results:
            writer.writerow(r)

    logger.info(f"Alpha sweep complete. Results saved to {analysis_dir}")
    return summary
