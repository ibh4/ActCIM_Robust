from __future__ import annotations

import csv
from pathlib import Path

import torch
import torch.nn as nn

from ..constants import SUPPORTED_LAYER_TYPES
from ..models import create_model
from ..nonlinearity import NonlinearityController, compute_activation_stats
from ..utils import save_json, ensure_dir, get_logger


def run_layer_sensitivity(
    checkpoint_path,
    model_name,
    config,
    device,
    data_dir,
    results_dir,
):
    logger = get_logger("actcim.analysis")
    results_dir = Path(results_dir)
    analysis_dir = ensure_dir(results_dir / "analysis")

    model = create_model(model_name, num_classes=getattr(config, "num_classes", 10))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    from ..data import get_cifar10_loaders
    _, _, test_loader = get_cifar10_loaders(
        batch_size=getattr(config, "batch_size", 128),
        num_workers=getattr(config, "num_workers", 4),
        data_dir=data_dir,
    )

    fixed_batch = next(iter(test_loader))
    if isinstance(fixed_batch, (list, tuple)):
        fixed_inputs, fixed_targets = fixed_batch[:2]
    else:
        fixed_inputs = fixed_batch
        fixed_targets = torch.zeros(fixed_inputs.size(0), dtype=torch.long)
    fixed_inputs = fixed_inputs.to(device)
    fixed_targets = fixed_targets.to(device)

    controller = NonlinearityController(model)
    controller.disable_all()
    controller.set_global_alpha(0.0)

    with torch.no_grad():
        clean_outputs = model(fixed_inputs)
    clean_acc = (clean_outputs.argmax(dim=1) == fixed_targets).float().mean().item()

    sensitivity_rows = []

    for layer_name in controller.get_layer_names():
        wrapper = controller.get_wrappers()[layer_name]
        layer_module = wrapper.module
        layer_type = type(layer_module).__name__
        param_count = sum(p.numel() for p in layer_module.parameters())

        if not isinstance(layer_module, (nn.Conv2d, nn.Linear)):
            continue

        controller.disable_all()
        controller.set_layer_alpha(layer_name, 0.0)
        controller.enable_layers([layer_name])

        row = {
            "layer_index": len(sensitivity_rows),
            "layer_name": layer_name,
            "layer_type": layer_type,
            "param_count": param_count,
        }

        for alpha_val in [-0.4, 0.4]:
            controller.set_layer_alpha(layer_name, alpha_val)
            controller.enable_layers([layer_name])

            with torch.no_grad():
                perturbed_outputs = model(fixed_inputs)
            acc = (perturbed_outputs.argmax(dim=1) == fixed_targets).float().mean().item()
            accuracy_drop = clean_acc - acc

            col_prefix = f"neg_04_" if alpha_val < 0 else "pos_04_"
            row[f"{col_prefix}accuracy"] = acc
            row[f"{col_prefix}accuracy_drop"] = accuracy_drop

            from ..evaluation.calibration import compute_ece, compute_brier_score, compute_nll
            from ..evaluation.classification_metrics import compute_accuracy
            row[f"{col_prefix}ece"] = compute_ece(perturbed_outputs, fixed_targets)
            row[f"{col_prefix}brier"] = compute_brier_score(perturbed_outputs, fixed_targets)
            row[f"{col_prefix}nll"] = compute_nll(perturbed_outputs, fixed_targets)

        row["sensitivity_score"] = max(
            abs(row.get("neg_04_accuracy_drop", 0)),
            abs(row.get("pos_04_accuracy_drop", 0)),
        )

        sensitivity_rows.append(row)
        logger.info(
            f"Layer {layer_name} ({layer_type}): "
            f"acc_drop(-0.4)={row['neg_04_accuracy_drop']:.4f}, "
            f"acc_drop(+0.4)={row['pos_04_accuracy_drop']:.4f}"
        )

    controller.disable_all()
    controller.set_global_alpha(0.0)

    csv_path = analysis_dir / "layer_sensitivity.csv"
    if sensitivity_rows:
        fieldnames = list(sensitivity_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sensitivity_rows)

    ranked = sorted(sensitivity_rows, key=lambda r: r["sensitivity_score"], reverse=True)
    for i, r in enumerate(ranked):
        r["rank"] = i + 1

    ranked_csv_path = analysis_dir / "layer_sensitivity_ranked.csv"
    if ranked:
        fieldnames = list(ranked[0].keys())
        with open(ranked_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(ranked)

    save_json({"clean_accuracy": clean_acc, "n_layers": len(sensitivity_rows), "results": sensitivity_rows},
              analysis_dir / "layer_sensitivity.json")

    logger.info(f"Layer sensitivity complete. {len(sensitivity_rows)} layers analyzed.")
    return sensitivity_rows
