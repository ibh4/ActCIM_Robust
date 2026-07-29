from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch

from .constants import PROJECT_ROOT, DEFAULT_ALPHAS


def main():
    parser = argparse.ArgumentParser(prog="actcim_robust")
    subparsers = parser.add_subparsers(dest="command")

    p = subparsers.add_parser("check-env", help="Check environment setup")

    p = subparsers.add_parser("test", help="Run tests")
    p.add_argument("--k", type=str, default=None, help="Test keyword expression")

    p = subparsers.add_parser("train", help="Train a model")
    p.add_argument("--config", required=True, help="Path to YAML config file")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--checkpoint", default=None, help="Checkpoint path for fine-tuning")
    p.add_argument("--profile", default="fast", choices=["smoke", "fast", "full"], help="Training profile")
    p.add_argument("--method", default=None, choices=["baseline", "fixed_nat", "random_nat", "sgr_nat"],
                   help="Training method")

    p = subparsers.add_parser("alpha-sweep", help="Run alpha sweep analysis")
    p.add_argument("--config", default=None, help="Path to YAML config file")
    p.add_argument("--checkpoint", required=True, help="Model checkpoint path")

    p = subparsers.add_parser("layer-sensitivity", help="Run layer sensitivity analysis")
    p.add_argument("--config", default=None, help="Path to YAML config file")
    p.add_argument("--checkpoint", required=True, help="Model checkpoint path")

    p = subparsers.add_parser("error-accumulation", help="Run error accumulation analysis")
    p.add_argument("--config", default=None, help="Path to YAML config file")
    p.add_argument("--checkpoint", required=True, help="Model checkpoint path")

    p = subparsers.add_parser("build-figures", help="Build all figures from analysis results")

    p = subparsers.add_parser("build-report", help="Build final report")

    p = subparsers.add_parser("validate-results", help="Validate result files for completeness")

    p = subparsers.add_parser("unified-sweep", help="Run unified alpha sweep on all models")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    if args.command == "check-env":
        check_env()
    elif args.command == "test":
        run_tests(args.k)
    elif args.command == "train":
        run_train(args)
    elif args.command == "alpha-sweep":
        run_alpha_sweep_cmd(args)
    elif args.command == "layer-sensitivity":
        run_layer_sensitivity_cmd(args)
    elif args.command == "error-accumulation":
        run_error_accumulation_cmd(args)
    elif args.command == "build-figures":
        build_figures()
    elif args.command == "validate-results":
        validate_results()
    elif args.command == "unified-sweep":
        run_unified_sweep_cmd()
    elif args.command == "build-report":
        build_report()


def check_env():
    from .environment import check_cuda, get_system_info

    print("=== ActCIM-Robust Environment Check ===")
    info = get_system_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    print()
    cuda_ok = check_cuda()

    try:
        import numpy
        print(f"  numpy: {numpy.__version__}")
    except ImportError:
        print("  numpy: NOT INSTALLED")

    try:
        import matplotlib
        print(f"  matplotlib: {matplotlib.__version__}")
    except ImportError:
        print("  matplotlib: NOT INSTALLED")

    try:
        import yaml
        print(f"  PyYAML: available")
    except ImportError:
        print("  PyYAML: NOT INSTALLED")

    try:
        from torchvision import __version__ as tv_version
        print(f"  torchvision: {tv_version}")
    except ImportError:
        print("  torchvision: NOT INSTALLED")

    print()
    if cuda_ok:
        print("Environment ready (CUDA available).")
    else:
        print("Environment ready (CPU only).")


def run_tests(k):
    import pytest

    test_dir = PROJECT_ROOT / "tests"
    args = ["-v"]
    if k:
        args.extend(["-k", k])
    args.append(str(test_dir))
    sys.exit(pytest.main(args))


def run_train(args):
    from .config import load_config
    from .reproducibility import set_seed
    from .constants import PROJECT_ROOT
    from .utils import get_logger

    set_seed(args.seed)
    config = load_config(args.config)

    data_root = "data"
    try:
        data_root = config.data.root
    except AttributeError:
        pass
    data_dir = str(PROJECT_ROOT / data_root)
    results_dir = str(PROJECT_ROOT / "results")

    logger = get_logger("actcim.train")
    logger.info(f"Starting training with method={args.method}, seed={args.seed}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Profile: {args.profile}")

    if args.method == "baseline":
        from .training.baseline import train_baseline
        summary = train_baseline(
            config=config,
            seed=args.seed,
            profile=args.profile,
            data_dir=data_dir,
            results_dir=results_dir,
        )
    elif args.method == "random_nat":
        from .training.random_nat import train_random_nat
        summary = train_random_nat(
            config=config,
            checkpoint_path=args.checkpoint,
            seed=args.seed,
            profile=args.profile,
            data_dir=data_dir,
            results_dir=results_dir,
        )
    elif args.method == "sgr_nat":
        from .training.sgr_nat import train_sgr_nat
        summary = train_sgr_nat(
            config=config,
            checkpoint_path=args.checkpoint,
            seed=args.seed,
            profile=args.profile,
            data_dir=data_dir,
            results_dir=results_dir,
        )
    elif args.method == "fixed_nat":
        from .training.fixed_nat import train_fixed_nat
        summary = train_fixed_nat(
            config=config,
            checkpoint_path=args.checkpoint,
            seed=args.seed,
            profile=args.profile,
            data_dir=data_dir,
            results_dir=results_dir,
        )
    else:
        print(f"Unknown method: {args.method}")
        return

    print(f"Training complete. Best val_acc: {summary.get('best_val_acc', 'N/A')}")


def run_alpha_sweep_cmd(args):
    from .config import load_config
    from .analysis.alpha_sweep import run_alpha_sweep
    from .constants import DATA_DIR, RESULTS_DIR
    from .utils import get_logger

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_config(args.config) if args.config else type("Config", (), {"num_classes": 10, "batch_size": 128})()

    logger = get_logger("actcim.cli")
    logger.info(f"Running alpha sweep on {args.checkpoint}")

    model_name = getattr(config, "model_name", "resnet18_cifar")
    result = run_alpha_sweep(
        checkpoint_path=args.checkpoint,
        model_name=model_name,
        config=config,
        device=device,
        data_dir=str(DATA_DIR),
        results_dir=str(RESULTS_DIR),
    )
    print(f"Alpha sweep complete. AURC={result.get('aurc', 'N/A'):.4f}" if isinstance(result, dict) else f"Done. {len(result)} alphas evaluated.")


def run_layer_sensitivity_cmd(args):
    from .config import load_config
    from .analysis.layer_sensitivity import run_layer_sensitivity
    from .constants import DATA_DIR, RESULTS_DIR
    from .utils import get_logger

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_config(args.config) if args.config else type("Config", (), {"num_classes": 10, "batch_size": 128})()

    logger = get_logger("actcim.cli")
    logger.info(f"Running layer sensitivity on {args.checkpoint}")

    model_name = getattr(config, "model_name", "resnet18_cifar")
    rows = run_layer_sensitivity(
        checkpoint_path=args.checkpoint,
        model_name=model_name,
        config=config,
        device=device,
        data_dir=str(DATA_DIR),
        results_dir=str(RESULTS_DIR),
    )
    print(f"Layer sensitivity complete. {len(rows)} layers analyzed.")


def run_error_accumulation_cmd(args):
    from .config import load_config
    from .analysis.error_accumulation import run_error_accumulation
    from .constants import DATA_DIR, RESULTS_DIR
    from .utils import get_logger

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_config(args.config) if args.config else type("Config", (), {"num_classes": 10, "batch_size": 128})()

    logger = get_logger("actcim.cli")
    logger.info(f"Running error accumulation on {args.checkpoint}")

    model_name = getattr(config, "model_name", "resnet18_cifar")
    rows = run_error_accumulation(
        checkpoint_path=args.checkpoint,
        model_name=model_name,
        config=config,
        device=device,
        data_dir=str(DATA_DIR),
        results_dir=str(RESULTS_DIR),
    )
    print(f"Error accumulation complete. {len(rows)} entries saved.")


def build_figures():
    from .constants import RESULTS_DIR
    from .utils import get_logger, load_json, ensure_dir
    from .visualization import setup_plot_style

    setup_plot_style()
    logger = get_logger("actcim.figures")
    figures_dir = ensure_dir(RESULTS_DIR / "figures")
    analysis_dir = RESULTS_DIR / "analysis"

    sweep_path = analysis_dir / "alpha_sweep_summary.json"
    if sweep_path.exists():
        from .visualization.robustness_plots import plot_accuracy_vs_alpha, plot_accuracy_drop_vs_alpha, plot_ece_vs_alpha, plot_loss_vs_alpha, plot_confidence_vs_alpha
        sweep_data = load_json(sweep_path)
        plot_accuracy_vs_alpha(sweep_data, figures_dir)
        plot_accuracy_drop_vs_alpha(sweep_data, figures_dir)
        plot_ece_vs_alpha(sweep_data, figures_dir)
        plot_loss_vs_alpha(sweep_data, figures_dir)
        plot_confidence_vs_alpha(sweep_data, figures_dir)
        logger.info("Robustness plots generated.")

    sensitivity_path = analysis_dir / "layer_sensitivity.json"
    if sensitivity_path.exists():
        from .visualization.layer_plots import (
            plot_layer_sensitivity_bar, plot_layer_sensitivity_heatmap,
            plot_sensitivity_vs_depth, plot_sensitivity_vs_param_count,
            plot_positive_negative_layer_gap,
        )
        sensitivity_data = load_json(sensitivity_path)
        rows = sensitivity_data.get("results", sensitivity_data)
        if rows:
            plot_layer_sensitivity_bar(rows, figures_dir)
            plot_layer_sensitivity_heatmap(rows, figures_dir)
            plot_sensitivity_vs_depth(rows, figures_dir)
            plot_sensitivity_vs_param_count(rows, figures_dir)
            plot_positive_negative_layer_gap(rows, figures_dir)
            logger.info("Layer sensitivity plots generated.")

    error_path = analysis_dir / "layer_error_accumulation.json"
    if error_path.exists():
        from .visualization.layer_plots import (
            plot_layer_error_accumulation, plot_layer_cosine_similarity,
            plot_layer_mean_std_shift, plot_activation_distribution_shift,
        )
        error_data = load_json(error_path)
        rows = error_data.get("results", error_data)
        if rows:
            plot_layer_error_accumulation(rows, "neg_04", figures_dir)
            plot_layer_error_accumulation(rows, "pos_04", figures_dir)
            plot_layer_cosine_similarity(rows, figures_dir)
            plot_layer_mean_std_shift(rows, figures_dir)
            plot_activation_distribution_shift(rows, figures_dir)
            logger.info("Error accumulation plots generated.")

    print(f"Figures built in {figures_dir}")


def build_report():
    from .constants import RESULTS_DIR
    from .utils import get_logger, ensure_dir

    logger = get_logger("actcim.report")
    report_dir = ensure_dir(RESULTS_DIR / "report")
    logger.info("Building report...")
    print(f"Report building handler called. Analysis data should be in {RESULTS_DIR / 'analysis'}")
    print(f"Report output to: {report_dir}")


def validate_results():
    from pathlib import Path
    from .constants import RESULTS_DIR
    from .utils import get_logger

    logger = get_logger("actcim.validate")
    analysis_dir = RESULTS_DIR / "analysis"
    required_files = [
        "alpha_sweep_summary.json",
        "alpha_sweep.csv",
        "layer_sensitivity.csv",
        "layer_sensitivity_ranked.csv",
        "layer_error_accumulation.csv",
    ]

    print("=== Result Validation ===")
    for fname in required_files:
        fpath = analysis_dir / fname
        status = "OK" if fpath.exists() else "MISSING"
        print(f"  {fname}: {status}")
        if fpath.exists():
            size = fpath.stat().st_size
            print(f"    Size: {size} bytes")

    print("Validation complete.")


def run_unified_sweep_cmd():
    import torch
    from .constants import PROJECT_ROOT, DATA_DIR
    from .evaluation.unified_sweep import run_unified_sweep
    from .utils import get_logger, ensure_dir

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = get_logger("actcim.cli")

    results_dir = PROJECT_ROOT / "results"
    output_dir = ensure_dir(results_dir / "post_training")

    checkpoints = [
        {"method": "clean", "path": "results/baseline/seed_42/best.pt"},
        {"method": "random_nat", "path": "results/random_nat/random_nat/seed_42/best.pt"},
        {"method": "sgr_nat", "path": "results/sgr_nat/sgr_nat/seed_42/best.pt"},
    ]

    logger.info("Starting unified sweep across Clean, Random-NAT, and SGR-NAT models")
    result = run_unified_sweep(
        checkpoints=checkpoints,
        output_dir=str(output_dir),
        data_dir=str(DATA_DIR),
        device=device,
        model_name="resnet18_cifar",
    )

    print(f"Unified sweep complete. Output: {output_dir}")
    if isinstance(result, dict):
        for method, info in result.get("methods", {}).items():
            aurc = info.get("aurc_all", "N/A")
            print(f"  {method}: AURC={aurc:.4f}" if isinstance(aurc, float) else f"  {method}: AURC={aurc}")


if __name__ == "__main__":
    main()
