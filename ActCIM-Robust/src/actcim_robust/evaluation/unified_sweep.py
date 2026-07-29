"""
Unified robustness evaluation for multiple models.
Evaluates all methods under identical test conditions across all alpha values.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ..constants import DEFAULT_ALPHAS, CIFAR10_MEAN, CIFAR10_STD, PROJECT_ROOT
from ..models import create_model
from ..nonlinearity import NonlinearityController
from ..evaluation.calibration import (
    compute_ece,
    compute_brier_score,
    compute_nll,
    compute_mean_confidence,
)
from ..evaluation.classification_metrics import compute_accuracy
from ..evaluation.robustness_metrics import (
    compute_aurc,
    compute_positive_negative_gap,
    compute_relative_improvement,
    compute_accuracy_drop,
    compute_worst_case_accuracy,
    compute_mean_accuracy_across_alpha,
)
from ..evaluation.performance import get_peak_gpu_memory
from ..utils import ensure_dir, save_json, get_logger

ALPHAS = [-0.8, -0.6, -0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4, 0.6, 0.8]
BATCH_SIZE = 256
NUM_WORKERS = 4


def compute_adaptive_ece(outputs, targets, n_bins=15):
    probs = F.softmax(outputs, dim=1)
    confidences, predictions = probs.max(dim=1)
    correct = predictions.eq(targets).float()

    sorted_conf, indices = confidences.sort()
    sorted_correct = correct[indices]

    n = len(sorted_conf)
    bin_size = n // n_bins

    ece = torch.tensor(0.0, device=outputs.device)
    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size if i < n_bins - 1 else n
        if start >= end:
            continue
        bin_count = end - start
        bin_conf = sorted_conf[start:end].mean()
        bin_acc = sorted_correct[start:end].mean()
        ece += (bin_count / n) * torch.abs(bin_acc - bin_conf)

    return ece.item()


def _get_test_loader(data_dir, batch_size, num_workers, seed=42):
    raw_dir = Path(data_dir) / "raw"

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])

    test_dataset = datasets.CIFAR10(
        root=str(raw_dir), train=False, download=False, transform=test_transform,
    )

    generator = torch.Generator()
    if num_workers > 0 and os.name == "nt":
        loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )
        try:
            _ = next(iter(loader))
        except (RuntimeError, OSError):
            num_workers = 0

    loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return loader


def _resolve_checkpoint_path(path_str, results_dir):
    path = Path(path_str)
    if not path.is_absolute():
        path = Path(results_dir) / path_str
    if path.exists():
        return str(path)

    method = None
    parts = Path(path_str).parts
    if parts[0] in ("results", "baseline"):
        if len(parts) >= 3 and parts[1] != "baseline":
            method = parts[1]
        elif parts[0] == "baseline":
            method = "baseline"
            path = Path(results_dir) / "baseline" / "seed_42" / "best.pt"
            if path.exists():
                return str(path)
            return None

    if method and method != "baseline":
        nested = Path(results_dir) / method / method / "seed_42" / "best.pt"
        if nested.exists():
            return str(nested)

    flat = Path(results_dir) / method / "seed_42" / "best.pt" if method else None
    if flat and flat.exists():
        return str(flat)

    return None


def _try_load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def evaluate_alpha(model, test_loader, device, alpha, controller):
    if alpha != 0.0:
        controller.set_global_alpha(alpha)
        controller.enable_all()
    else:
        controller.set_global_alpha(0.0)
        controller.enable_all()

    model.eval()
    model.to(device)

    all_outputs = []
    all_targets = []
    total_loss = 0.0
    total_samples = 0

    criterion = torch.nn.CrossEntropyLoss()

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    start_time = time.perf_counter()

    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[:2]
            else:
                inputs = batch
                targets = torch.zeros(inputs.size(0), dtype=torch.long)

            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_outputs.append(outputs.cpu())
            all_targets.append(targets.cpu())

    elapsed = time.perf_counter() - start_time

    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    all_outputs_device = all_outputs.to(device)
    all_targets_device = all_targets.to(device)

    accuracy = compute_accuracy(all_outputs_device, all_targets_device)
    loss = total_loss / total_samples if total_samples > 0 else 0.0
    nll = compute_nll(all_outputs_device, all_targets_device)
    brier = compute_brier_score(all_outputs_device, all_targets_device)
    ece = compute_ece(all_outputs_device, all_targets_device, n_bins=15)
    adaptive_ece = compute_adaptive_ece(all_outputs_device, all_targets_device, n_bins=15)
    mean_conf = compute_mean_confidence(all_outputs_device)

    confidence_accuracy_gap = mean_conf - accuracy

    throughput = total_samples / elapsed if elapsed > 0 else 0.0

    peak_gpu_memory_mb = get_peak_gpu_memory()

    return {
        "test_accuracy": float(accuracy),
        "test_loss": float(loss),
        "nll": float(nll),
        "brier_score": float(brier),
        "ece_15_bins": float(ece),
        "adaptive_ece": float(adaptive_ece),
        "mean_confidence": float(mean_conf),
        "confidence_accuracy_gap": float(confidence_accuracy_gap),
        "sample_count": total_samples,
        "elapsed_seconds": float(elapsed),
        "throughput_samples_per_second": float(throughput),
        "peak_gpu_memory_mb": float(peak_gpu_memory_mb),
        "alpha": float(alpha),
    }


def run_unified_sweep(checkpoints, output_dir, data_dir, device, model_name="resnet18_cifar", seed=42):
    logger = get_logger("actcim.unified")
    output_dir = ensure_dir(Path(output_dir))

    alphas = sorted(ALPHAS)

    test_loader = _get_test_loader(data_dir, BATCH_SIZE, NUM_WORKERS, seed=seed)

    all_rows = []
    clean_baseline_acc = None
    method_alpha0_acc = {}

    for cp_info in checkpoints:
        method = cp_info["method"]
        cp_path = cp_info["path"]
        resolved = _resolve_checkpoint_path(cp_path, str(PROJECT_ROOT))
        if resolved is None:
            logger.warning(f"Checkpoint not found: {cp_path}, skipping {method}")
            continue

        logger.info(f"Evaluating {method}: {resolved}")

        model = create_model(model_name, num_classes=10)
        state_dict = _try_load_checkpoint(resolved)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)

        controller = NonlinearityController(model)

        method_rows = []
        for alpha in alphas:
            logger.info(f"  {method} alpha={alpha:+.2f}")
            row = evaluate_alpha(model, test_loader, device, alpha, controller)
            row["method"] = method
            row["checkpoint_path"] = resolved
            row["seed"] = seed
            row["accuracy_drop_from_own_alpha0"] = None
            row["accuracy_drop_from_clean_reference"] = None
            row["relative_accuracy_drop"] = None
            method_rows.append(row)
            all_rows.append(row)

        if method_rows:
            own_alpha0_row = next((r for r in method_rows if r["alpha"] == 0.0), None)
            if own_alpha0_row:
                own_alpha0_acc = own_alpha0_row["test_accuracy"]
                method_alpha0_acc[method] = own_alpha0_acc
                for row in method_rows:
                    row["accuracy_drop_from_own_alpha0"] = own_alpha0_acc - row["test_accuracy"]

        torch.cuda.empty_cache()

    clean_alpha0 = method_alpha0_acc.get("clean", None)
    if clean_alpha0 is None:
        clean_rows = [r for r in all_rows if r["method"] == "clean" and r["alpha"] == 0.0]
        if clean_rows:
            clean_alpha0 = clean_rows[0]["test_accuracy"]
            method_alpha0_acc["clean"] = clean_alpha0

    if clean_alpha0 is not None:
        for row in all_rows:
            row["accuracy_drop_from_clean_reference"] = clean_alpha0 - row["test_accuracy"]
            if clean_alpha0 > 0:
                row["relative_accuracy_drop"] = (clean_alpha0 - row["test_accuracy"]) / clean_alpha0

    per_method = {}
    for row in all_rows:
        method = row["method"]
        if method not in per_method:
            per_method[method] = []
        per_method[method].append(row)

    for method, rows in per_method.items():
        csv_path = output_dir / f"{method}_alpha_sweep.csv"
        fieldnames = [
            "method", "checkpoint_path", "seed", "alpha",
            "test_accuracy", "test_loss", "nll", "brier_score",
            "ece_15_bins", "adaptive_ece", "mean_confidence",
            "confidence_accuracy_gap", "accuracy_drop_from_own_alpha0",
            "accuracy_drop_from_clean_reference", "relative_accuracy_drop",
            "sample_count", "elapsed_seconds", "throughput_samples_per_second",
            "peak_gpu_memory_mb",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    combined_csv_path = output_dir / "all_methods_alpha_sweep.csv"
    fieldnames = [
        "method", "checkpoint_path", "seed", "alpha",
        "test_accuracy", "test_loss", "nll", "brier_score",
        "ece_15_bins", "adaptive_ece", "mean_confidence",
        "confidence_accuracy_gap", "accuracy_drop_from_own_alpha0",
        "accuracy_drop_from_clean_reference", "relative_accuracy_drop",
        "sample_count", "elapsed_seconds", "throughput_samples_per_second",
        "peak_gpu_memory_mb",
    ]
    with open(combined_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    aggregate = _compute_aggregates(all_rows, per_method, alphas, clean_alpha0, method_alpha0_acc)

    combined_json_path = output_dir / "all_methods_alpha_sweep.json"
    save_json(aggregate, combined_json_path)

    manifest = {
        "description": "Unified alpha sweep evaluation for Clean, Random-NAT, and SGR-NAT models",
        "output_dir": str(output_dir),
        "alphas": alphas,
        "methods": list(per_method.keys()),
        "files": {
            "per_method_csv": {m: f"{m}_alpha_sweep.csv" for m in per_method},
            "combined_csv": "all_methods_alpha_sweep.csv",
            "combined_json": "all_methods_alpha_sweep.json",
        },
        "aggregate_summary": {
            "aurc_all": aggregate.get("aurc_all"),
            "aurc_positive": aggregate.get("aurc_positive"),
            "aurc_negative": aggregate.get("aurc_negative"),
            "clean_baseline_acc": clean_alpha0,
        },
    }
    manifest_path = output_dir / "all_methods_alpha_sweep_manifest.json"
    save_json(manifest, manifest_path)

    logger.info(f"Unified sweep complete. Results saved to {output_dir}")
    return aggregate


def _compute_aggregates(all_rows, per_method, alphas, clean_alpha0, method_alpha0_acc):
    aggregate = {
        "alphas": alphas,
        "clean_baseline_accuracy": clean_alpha0,
        "methods": {},
    }

    for method, rows in per_method.items():
        accuracies = [r["test_accuracy"] for r in rows]
        method_alphas_list = [r["alpha"] for r in rows]
        sorted_pairs = sorted(zip(method_alphas_list, accuracies), key=lambda x: x[0])
        sorted_alphas = [p[0] for p in sorted_pairs]
        sorted_accs = [p[1] for p in sorted_pairs]

        pos_alphas = [a for a in sorted_alphas if a > 0]
        neg_alphas = [a for a in sorted_alphas if a < 0]
        pos_accs = [acc for a, acc in sorted_pairs if a > 0]
        neg_accs = [acc for a, acc in sorted_pairs if a < 0]

        aurc_all = compute_aurc(sorted_accs, sorted_alphas)
        aurc_pos = compute_aurc(pos_accs, pos_alphas) if len(pos_alphas) >= 2 else float(np.mean(pos_accs)) if pos_accs else 0.0
        aurc_neg = compute_aurc(neg_accs, neg_alphas) if len(neg_alphas) >= 2 else float(np.mean(neg_accs)) if neg_accs else 0.0

        worst_acc = compute_worst_case_accuracy(accuracies)
        worst_idx = np.argmin(accuracies)
        worst_alpha_val = method_alphas_list[worst_idx]

        non_zero_accs = [acc for a, acc in zip(method_alphas_list, accuracies) if a != 0.0]
        mean_pert = float(np.mean(non_zero_accs)) if non_zero_accs else 0.0

        pos_neg_gap = compute_positive_negative_gap(pos_accs, neg_accs) if pos_accs and neg_accs else None

        method_info = {
            "aurc_all": aurc_all,
            "aurc_positive": aurc_pos,
            "aurc_negative": aurc_neg,
            "worst_case_accuracy": worst_acc,
            "worst_alpha": worst_alpha_val,
            "mean_perturbed_accuracy": mean_pert,
            "alpha0_accuracy": next((acc for a, acc in sorted_pairs if a == 0.0), None),
            "asymmetry_gap_positive_negative": pos_neg_gap,
        }

        if clean_alpha0 is not None and method != "clean":
            mean_clean_nonzero = [acc for a, acc in sorted_pairs if a != 0.0 and "clean" in per_method]
            method_info["relative_improvement_vs_clean"] = compute_relative_improvement(
                clean_alpha0, mean_pert
            )

        aggregate["methods"][method] = method_info
        aggregate[f"aurc_{method}"] = aurc_all

    if len(per_method) >= 2 and "clean" in per_method:
        clean_method = aggregate["methods"].get("clean", {})
        clean_aurc = clean_method.get("aurc_all", 0)
        for method in per_method:
            if method == "clean":
                continue
            method_aurc = aggregate["methods"][method].get("aurc_all", 0)
            aggregate[f"aurc_improvement_{method}_vs_clean"] = clean_aurc - method_aurc
            if clean_alpha0 and method in method_alpha0_acc:
                rel_imp = compute_relative_improvement(clean_alpha0, method_alpha0_acc[method])
                aggregate["methods"][method]["relative_alpha0_improvement"] = rel_imp

    return aggregate
