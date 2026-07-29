from pathlib import Path
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
CONFIG_DIR = PROJECT_ROOT / "config"

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

DEFAULT_SEEDS = [42, 3407, 2026]
DEFAULT_ALPHAS = [-0.8, -0.6, -0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4, 0.6, 0.8]

NUM_CLASSES = 10

SUPPORTED_LAYER_TYPES = (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)

TRAIN_SIZE = 45000
VAL_SIZE = 5000
TEST_SIZE = 10000

EPSILON = 1e-8
