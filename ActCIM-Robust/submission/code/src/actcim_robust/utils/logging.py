import logging
import sys
from pathlib import Path


def setup_logging(log_dir, name="actcim"):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def get_logger(name="actcim"):
    return logging.getLogger(name)


class LogHelper:
    def __init__(self, logger):
        self.logger = logger

    def log_epoch(self, epoch, train_loss, val_loss, metrics=None):
        msg = f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        if metrics:
            for k, v in metrics.items():
                msg += f" | {k}: {v:.4f}"
        self.logger.info(msg)

    def log_metric(self, name, value, step=None):
        msg = f"{name}: {value:.4f}"
        if step is not None:
            msg = f"Step {step} | {msg}"
        self.logger.info(msg)

    def log_stage(self, stage_name):
        self.logger.info(f"{'=' * 50}")
        self.logger.info(f"  STAGE: {stage_name}")
        self.logger.info(f"{'=' * 50}")
