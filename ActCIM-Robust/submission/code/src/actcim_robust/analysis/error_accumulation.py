from __future__ import annotations

import csv
from pathlib import Path

import torch
import torch.nn as nn

from ..constants import DEFAULT_ALPHAS
from ..models import create_model
from ..nonlinearity import NonlinearityController, compute_activation_stats
from ..utils import save_json, ensure_dir, get_logger


def run_error_accumulation(
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

    controller = NonlinearityController(model)

    def get_activations_for_mode(alpha):
        controller.set_global_alpha(alpha)
        if alpha == 0:
            controller.disable_all()
        else:
            controller.enable_all()

        activation_outputs = {}

        def hook_fn(name):
            def fn(module, inp, out):
                if isinstance(out, torch.Tensor):
                    activation_outputs[name] = out.detach().clone()
            return fn

        handles = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                h = module.register_forward_hook(hook_fn(name))
                handles.append(h)

        with torch.no_grad():
            _ = model(fixed_inputs)

        for h in handles:
            h.remove()

        return activation_outputs

    clean_activations = get_activations_for_mode(0.0)
    neg_activations = get_activations_for_mode(-0.4)
    pos_activations = get_activations_for_mode(0.4)

    controller.disable_all()

    error_rows = []
    layer_names = sorted(clean_activations.keys())

    for layer_name in layer_names:
        clean_act = clean_activations[layer_name]

        for label, act_dict in [("neg_04", neg_activations), ("pos_04", pos_activations)]:
            if layer_name not in act_dict:
                continue
            perturbed_act = act_dict[layer_name]

            stats = compute_activation_stats(clean_act, perturbed_act)

            row = {
                "layer_name": layer_name,
                "alpha_sign": label,
                "clean_mean": clean_act.float().mean().item(),
                "clean_std": clean_act.float().std().item(),
                "perturbed_mean": perturbed_act.float().mean().item(),
                "perturbed_std": perturbed_act.float().std().item(),
                "relative_l2": stats["relative_l2"],
                "cosine_similarity": stats["cosine_similarity"],
                "mean_shift": stats["mean_shift"],
                "std_ratio": stats["std_ratio"],
                "max_abs_error": stats["max_abs_error"],
                "p95_error": stats["p95_error"],
                "sign_flip_ratio": stats["sign_flip_ratio"],
                "zero_fraction_change": stats["zero_fraction_change"],
                "saturation_ratio": stats["saturation_ratio"],
            }
            error_rows.append(row)

    csv_path = analysis_dir / "layer_error_accumulation.csv"
    if error_rows:
        fieldnames = list(error_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(error_rows)

    save_json({"n_layers": len(layer_names), "results": error_rows},
              analysis_dir / "layer_error_accumulation.json")

    logger.info(f"Error accumulation complete. {len(error_rows)} entries saved.")
    return error_rows
