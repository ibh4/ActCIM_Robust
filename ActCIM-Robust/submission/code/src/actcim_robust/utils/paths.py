import shutil
from pathlib import Path


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_exp_dir(results_dir, exp_name, seed):
    return ensure_dir(Path(results_dir) / exp_name / f"seed_{seed}")


def find_latest_checkpoint(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    for name in ["best.pt", "best.pth", "last.pt", "last.pth"]:
        candidate = checkpoint_dir / name
        if candidate.exists():
            return candidate

    checkpoints = sorted(checkpoint_dir.glob("*.pt")) + sorted(checkpoint_dir.glob("*.pth"))
    return checkpoints[-1] if checkpoints else None


def clean_results_dir(path):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
