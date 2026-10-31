"""Fully connected GANs on MNIST: minimax vs non-saturating loss, and a conditional GAN.

Architecture (from the notebooks): one hidden layer of 128 ReLU units in both
networks, sigmoid outputs, Xavier-style initialisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .utils import CHECKPOINT_DIR, OUTPUT_DIR, get_device, mnist_arrays, save_grid

X_DIM = 28 * 28
N_CLASSES = 10


def _xavier_linear(in_dim: int, out_dim: int) -> nn.Linear:
    layer = nn.Linear(in_dim, out_dim)
    nn.init.normal_(layer.weight, std=1.0 / np.sqrt(in_dim / 2.0))
    nn.init.zeros_(layer.bias)
    return layer


class Generator(nn.Module):
    def __init__(self, z_dim: int = 100, h_dim: int = 128, n_classes: int = 0):
        super().__init__()
        self.net = nn.Sequential(
            _xavier_linear(z_dim + n_classes, h_dim), nn.ReLU(), _xavier_linear(h_dim, X_DIM), nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        return self.net(z if c is None else torch.cat([z, c], dim=1))


class Discriminator(nn.Module):
    def __init__(self, h_dim: int = 128, n_classes: int = 0):
        super().__init__()
        self.net = nn.Sequential(_xavier_linear(X_DIM + n_classes, h_dim), nn.ReLU(), _xavier_linear(h_dim, 1))

    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        """Returns logits; D(x) = sigmoid(logits)."""
        return self.net(x if c is None else torch.cat([x, c], dim=1))


@dataclass
class GANConfig:
    loss: str = "non-saturating"  # or "minimax"
    conditional: bool = False
    iterations: int = 100_000
    batch_size: int = 64
    lr: float = 1e-3
    z_dim: int = 100
    noise: str = "normal"  # or "uniform" in [-1, 1]
    sample_every: int = 1000


# Settings used in the notebooks.
PRESETS = {
    # 2.1 (TensorFlow): uniform noise, batch 128, lr 1e-4, 30k iterations.
    "minimax": GANConfig(loss="minimax", iterations=30_000, batch_size=128, lr=1e-4, noise="uniform"),
    # 2.2 (PyTorch): normal noise, batch 64, lr 1e-3, 100k iterations.
    "non-saturating": GANConfig(loss="non-saturating"),
    # 2.3: same as 2.2 with the one-hot label fed to both networks.
    "conditional": GANConfig(loss="non-saturating", conditional=True),
}


def sample_noise(n: int, cfg: GANConfig, device) -> torch.Tensor:
    if cfg.noise == "uniform":
        return torch.empty(n, cfg.z_dim, device=device).uniform_(-1.0, 1.0)
    return torch.randn(n, cfg.z_dim, device=device)


def discriminator_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """-[log D(x) + log(1 - D(G(z)))], the same for both GAN variants."""
    return F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits)) + F.binary_cross_entropy_with_logits(
        fake_logits, torch.zeros_like(fake_logits)
    )


def generator_loss(fake_logits: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "minimax":
        # Original objective: minimise log(1 - D(G(z))). Its gradient vanishes
        # when D confidently rejects the fakes, i.e. early in training.
        return -F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    if kind == "non-saturating":
        # Maximise log D(G(z)) instead: strong gradients when the generator is poor.
        return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
    raise ValueError(f"unknown generator loss: {kind}")


def train(cfg: GANConfig, name: str | None = None) -> tuple[Generator, Discriminator]:
    device = get_device()
    name = name or ("conditional" if cfg.conditional else cfg.loss)
    out_dir = OUTPUT_DIR / f"gan_{name}"
    x_train, y_train = mnist_arrays(train=True)
    x_train = torch.from_numpy(x_train)
    y_onehot = F.one_hot(torch.from_numpy(y_train).long(), N_CLASSES).float()

    n_classes = N_CLASSES if cfg.conditional else 0
    G = Generator(cfg.z_dim, n_classes=n_classes).to(device)
    D = Discriminator(n_classes=n_classes).to(device)
    g_opt = torch.optim.Adam(G.parameters(), lr=cfg.lr)
    d_opt = torch.optim.Adam(D.parameters(), lr=cfg.lr)

    print(f"Training GAN '{name}': {cfg}")
    for it in range(cfg.iterations):
        idx = torch.randint(0, len(x_train), (cfg.batch_size,))
        x = x_train[idx].to(device)
        c = y_onehot[idx].to(device) if cfg.conditional else None

        # Discriminator step.
        fake = G(sample_noise(cfg.batch_size, cfg, device), c)
        d_loss = discriminator_loss(D(x, c), D(fake.detach(), c))
        d_opt.zero_grad()
        d_loss.backward()
        d_opt.step()

        # Generator step.
        fake = G(sample_noise(cfg.batch_size, cfg, device), c)
        g_loss = generator_loss(D(fake, c), cfg.loss)
        g_opt.zero_grad()
        g_loss.backward()
        g_opt.step()

        if it % cfg.sample_every == 0 or it == cfg.iterations - 1:
            print(f"Iter-{it}; D_loss: {d_loss.item():.4f}; G_loss: {g_loss.item():.4f}")
            save_samples(G, cfg, out_dir / f"{it:06d}.png", device)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f"gan_{name}.pt"
    torch.save({"config": cfg.__dict__, "generator": G.state_dict(), "discriminator": D.state_dict()}, path)
    print(f"Samples in {out_dir}/, model saved to {path}")
    return G, D


@torch.no_grad()
def save_samples(G: Generator, cfg: GANConfig, path: Path, device) -> Path:
    G.eval()
    if cfg.conditional:
        # One row per digit 0-9.
        labels = torch.arange(N_CLASSES, device=device).repeat_interleave(8)
        c = F.one_hot(labels, N_CLASSES).float()
        images = G(sample_noise(len(labels), cfg, device), c)
        out = save_grid(images.view(N_CLASSES, 8, 784).reshape(-1, 784), path, nrow=8)
    else:
        out = save_grid(G(sample_noise(16, cfg, device)), path, nrow=4)
    G.train()
    return out
