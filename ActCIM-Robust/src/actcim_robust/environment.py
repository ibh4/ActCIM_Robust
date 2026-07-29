import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch


def get_system_info():
    info = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "hostname": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "timestamp": datetime.now().isoformat(),
    }

    info["cuda_version"] = torch.version.cuda if torch.version.cuda else "N/A"

    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["gpu_memory"] = f"{props.total_memory / (1024 ** 3):.2f} GB"
    else:
        info["gpu_name"] = "N/A"
        info["gpu_memory"] = "N/A"

    try:
        import cpuinfo
        info["cpu_name"] = cpuinfo.get_cpu_info()["brand_raw"]
    except ImportError:
        info["cpu_name"] = platform.processor() or "Unknown"

    try:
        import psutil
        info["total_ram"] = f"{psutil.virtual_memory().total / (1024 ** 3):.2f} GB"
    except ImportError:
        info["total_ram"] = "Unknown"

    return info


def save_environment_info(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    info = get_system_info()

    env_path = output_dir / "environment.json"
    with open(env_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True
        )
        freeze_path = output_dir / "pip_freeze.txt"
        with open(freeze_path, "w", encoding="utf-8") as f:
            f.write(result.stdout)
    except Exception:
        pass


def check_cuda():
    available = torch.cuda.is_available()
    if available:
        print(f"CUDA available: {available}")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available")
    return available
