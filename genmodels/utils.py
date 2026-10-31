"""Shared helpers: device, MNIST loading and saving image grids."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torchvision import datasets, transforms  # noqa: E402
from torchvision.utils import make_grid  # noqa: E402

DATA_DIR = "data/raw"
OUTPUT_DIR = Path("outputs")
CHECKPOINT_DIR = Path("checkpoints")


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def mnist(train: bool = True) -> datasets.MNIST:
    """MNIST as [0, 1] tensors of shape (1, 28, 28)."""
    return datasets.MNIST(DATA_DIR, train=train, download=True, transform=transforms.ToTensor())


def mnist_arrays(train: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """MNIST as flat float32 arrays in [0, 1] (N, 784) and integer labels (N,)."""
    ds = mnist(train)
    x = ds.data.numpy().reshape(-1, 28 * 28).astype(np.float32) / 255.0
    return x, ds.targets.numpy()


def save_grid(images: torch.Tensor, path: Path, nrow: int | None = None, title: str | None = None) -> Path:
    """Save a grid of (N, 1, 28, 28) or (N, 784) images in [0, 1]."""
    images = images.detach().cpu().float()
    if images.dim() == 2:
        images = images.view(-1, 1, 28, 28)
    nrow = nrow or int(np.ceil(np.sqrt(len(images))))
    grid = make_grid(images.clamp(0, 1), nrow=nrow, pad_value=1.0)
    fig = plt.figure(figsize=(nrow * 0.6 + 1, nrow * 0.6 + 1))
    plt.imshow(grid.permute(1, 2, 0), vmin=0, vmax=1)
    plt.axis("off")
    if title:
        plt.title(title)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
