"""Score-based diffusion on MNIST.

- 1D forward/reverse diffusion demos
- Variance-exploding SDE with sigma = 25
- Three time-conditioned U-Nets: concatenation skips (UNet), additive skips
  (UNetRes), and additive skips + cross-attention to a digit label (UNetTransformer)
- Denoising score-matching loss and an Euler-Maruyama sampler
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402
from torch.optim.lr_scheduler import LambdaLR  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from .utils import CHECKPOINT_DIR, OUTPUT_DIR, get_device, mnist, save_grid  # noqa: E402

SIGMA = 25.0

# ------------------------------------------------------------------ 1D demos


def forward_diffusion_1d(x0, noise_strength_fn, t0, nsteps, dt):
    x = np.zeros(nsteps + 1)
    x[0] = x0
    t = t0 + np.arange(nsteps + 1) * dt
    for i in range(nsteps):
        x[i + 1] = x[i] + np.random.randn() * noise_strength_fn(t[i])
    return x, t


def reverse_diffusion_1d(x0, noise_strength_fn, score_fn, T, nsteps, dt):
    x = np.zeros(nsteps + 1)
    x[0] = x0
    t = np.arange(nsteps + 1) * dt
    for i in range(nsteps):
        g = noise_strength_fn(T - t[i])
        score = score_fn(x[i], 0, g, T - t[i])
        x[i + 1] = x[i] + score * g**2 * dt + g * np.random.randn() * np.sqrt(dt)
    return x, t


def score_simple(x, x0, noise_strength, t):
    return -(x - x0) / (noise_strength**2 * t)


def plot_1d_demos(num_tries: int = 5) -> list[Path]:
    paths = []
    const = lambda t: 1  # noqa: E731

    fig = plt.figure(figsize=(15, 3))
    for i in range(num_tries):
        x, t = forward_diffusion_1d(0, const, 0, 100, 0.1)
        plt.plot(t, x, label=f"Trial {i + 1}")
    plt.xlabel("Time")
    plt.ylabel("Sample value (x)")
    plt.title("Forward diffusion")
    plt.legend()
    paths.append(_save(fig, "diffusion_1d_forward"))

    T = 11
    fig = plt.figure(figsize=(15, 3))
    for i in range(num_tries):
        x, t = reverse_diffusion_1d(np.random.normal(0, T), const, score_simple, T, 100, 0.1)
        plt.plot(t, x, label=f"Trial {i + 1}")
    plt.xlabel("Time")
    plt.ylabel("Sample value (x)")
    plt.title("Reverse diffusion")
    plt.legend()
    paths.append(_save(fig, "diffusion_1d_reverse"))
    return paths


def _save(fig, name: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")
    return path


# ------------------------------------------------------------------ SDE


def marginal_prob_std(t: torch.Tensor, sigma: float = SIGMA) -> torch.Tensor:
    """Std of the perturbation kernel p_0t(x(t) | x(0)) for dx = sigma^t dw."""
    return torch.sqrt((sigma ** (2 * t) - 1.0) / 2.0 / math.log(sigma))


def diffusion_coeff(t: torch.Tensor, sigma: float = SIGMA) -> torch.Tensor:
    return sigma**t


# ------------------------------------------------------------------ networks


class GaussianFourierProjection(nn.Module):
    """Fixed random Fourier features for encoding the time step."""

    def __init__(self, embed_dim: int, scale: float = 30.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """Linear layer whose output is broadcast over the spatial dimensions."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense(x)[..., None, None]


class _UNetBase(nn.Module):
    """Shared encoder/decoder layers. `concat_skips` selects concatenation (UNet)
    or addition (UNetRes / UNetTransformer) for the skip connections."""

    concat_skips = False

    def __init__(self, channels=(32, 64, 128, 256), embed_dim: int = 256, sigma: float = SIGMA):
        super().__init__()
        self.sigma = sigma
        k = 2 if self.concat_skips else 1
        self.time_embed = nn.Sequential(GaussianFourierProjection(embed_dim), nn.Linear(embed_dim, embed_dim))

        self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
        self.tconv3 = nn.ConvTranspose2d(k * channels[2], channels[1], 3, stride=2, bias=False, output_padding=1)
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        self.tconv2 = nn.ConvTranspose2d(k * channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
        self.tconv1 = nn.ConvTranspose2d(k * channels[0], 1, 3, stride=1)
        self.act = nn.SiLU()  # swish

    def _skip(self, h, skip):
        return torch.cat([h, skip], dim=1) if self.concat_skips else h + skip

    def _attend(self, h, level, y_embed):  # overridden by the transformer variant
        return h

    def _condition(self, y):
        return None

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        embed = self.act(self.time_embed(t))
        y_embed = self._condition(y)

        h1 = self.act(self.gnorm1(self.conv1(x) + self.dense1(embed)))
        h2 = self.act(self.gnorm2(self.conv2(h1) + self.dense2(embed)))
        h3 = self._attend(self.act(self.gnorm3(self.conv3(h2) + self.dense3(embed))), 3, y_embed)
        h4 = self._attend(self.act(self.gnorm4(self.conv4(h3) + self.dense4(embed))), 4, y_embed)

        h = self.act(self.tgnorm4(self.tconv4(h4) + self.dense5(embed)))
        h = self.act(self.tgnorm3(self.tconv3(self._skip(h, h3)) + self.dense6(embed)))
        h = self.act(self.tgnorm2(self.tconv2(self._skip(h, h2)) + self.dense7(embed)))
        h = self.tconv1(self._skip(h, h1))
        return h / marginal_prob_std(t, self.sigma)[:, None, None, None]


class UNet(_UNetBase):
    """Time-conditioned U-Net with concatenation skip connections."""

    concat_skips = True


class UNetRes(_UNetBase):
    """Time-conditioned U-Net with additive (residual) skip connections."""

    concat_skips = False


class CrossAttention(nn.Module):
    """Single-head attention; self-attention when context_dim is None."""

    def __init__(self, embed_dim: int, hidden_dim: int, context_dim: int | None = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.self_attn = context_dim is None
        kv_dim = hidden_dim if self.self_attn else context_dim
        self.query = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.key = nn.Linear(kv_dim, embed_dim, bias=False)
        self.value = nn.Linear(kv_dim, hidden_dim, bias=False)

    def forward(self, tokens: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        source = tokens if self.self_attn else context
        Q, K, V = self.query(tokens), self.key(source), self.value(source)
        attn = F.softmax(torch.einsum("BTH,BSH->BTS", Q, K) / math.sqrt(self.embed_dim), dim=-1)
        return torch.einsum("BTS,BSH->BTH", attn, V)


class TransformerBlock(nn.Module):
    """Self-attention, cross-attention and a feed-forward net, each pre-norm with a residual."""

    def __init__(self, hidden_dim: int, context_dim: int):
        super().__init__()
        self.attn_self = CrossAttention(hidden_dim, hidden_dim)
        self.attn_cross = CrossAttention(hidden_dim, hidden_dim, context_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, 3 * hidden_dim), nn.GELU(), nn.Linear(3 * hidden_dim, hidden_dim))

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        x = self.attn_self(self.norm1(x)) + x
        x = self.attn_cross(self.norm2(x), context=context) + x
        return self.ffn(self.norm3(x)) + x


class SpatialTransformer(nn.Module):
    """Runs a TransformerBlock over the H*W positions of a feature map."""

    def __init__(self, hidden_dim: int, context_dim: int):
        super().__init__()
        self.transformer = TransformerBlock(hidden_dim, context_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # b (h w) c
        tokens = self.transformer(tokens, context)
        return tokens.transpose(1, 2).reshape(b, c, h, w) + x


class UNetTransformer(_UNetBase):
    """UNetRes plus cross-attention to an embedding of the digit label."""

    concat_skips = False

    def __init__(self, channels=(32, 64, 128, 256), embed_dim: int = 256, text_dim: int = 256, n_classes: int = 10,
                 sigma: float = SIGMA):
        super().__init__(channels, embed_dim, sigma)
        self.attn3 = SpatialTransformer(channels[2], text_dim)
        self.attn4 = SpatialTransformer(channels[3], text_dim)
        self.cond_embed = nn.Embedding(n_classes, text_dim)

    def _condition(self, y):
        return self.cond_embed(y).unsqueeze(1)

    def _attend(self, h, level, y_embed):
        return (self.attn3 if level == 3 else self.attn4)(h, y_embed)


# ------------------------------------------------------------------ training and sampling


def score_matching_loss(model, x, y=None, eps: float = 1e-5, conditional_t_range: bool = False) -> torch.Tensor:
    """Denoising score-matching loss.

    t ~ U(eps, 1 - eps) for the unconditional models and U(eps, 1) for the
    conditional one, as in the notebook.
    """
    scale = (1.0 - eps) if conditional_t_range else (1.0 - 2 * eps)
    random_t = torch.rand(x.shape[0], device=x.device) * scale + eps
    std = marginal_prob_std(random_t, model.sigma)[:, None, None, None]
    z = torch.randn_like(x)
    score = model(x + z * std, random_t, y=y)
    return torch.mean(torch.sum((score * std + z) ** 2, dim=(1, 2, 3)))


@torch.no_grad()
def euler_maruyama_sampler(model, batch_size: int = 64, num_steps: int = 500, device=None, eps: float = 1e-3,
                           y: torch.Tensor | None = None) -> torch.Tensor:
    device = device or next(model.parameters()).device
    model.eval()
    t = torch.ones(batch_size, device=device)
    x = torch.randn(batch_size, 1, 28, 28, device=device) * marginal_prob_std(t, model.sigma)[:, None, None, None]
    time_steps = torch.linspace(1.0, eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    mean_x = x
    for time_step in time_steps:
        batch_t = torch.ones(batch_size, device=device) * time_step
        g = diffusion_coeff(batch_t, model.sigma)[:, None, None, None]
        mean_x = x + g**2 * model(x, batch_t, y=y) * step_size
        x = mean_x + torch.sqrt(step_size) * g * torch.randn_like(x)
    return mean_x.clamp(0.0, 1.0)  # no noise in the last step


MODELS = {"unet": UNet, "unet_res": UNetRes, "transformer": UNetTransformer}


@dataclass
class DiffusionConfig:
    model: str = "unet"
    epochs: int = 50
    batch_size: int = 2048
    lr: float = 5e-4
    lr_decay: bool = False  # lr *= max(0.2, 0.98^epoch)
    num_workers: int = 2


PRESETS = {
    "unet": DiffusionConfig("unet", epochs=50, batch_size=2048, lr=5e-4, lr_decay=False),
    "unet_res": DiffusionConfig("unet_res", epochs=75, batch_size=1024, lr=1e-3, lr_decay=True),
    "transformer": DiffusionConfig("transformer", epochs=100, batch_size=1024, lr=1e-3, lr_decay=True),
}


def checkpoint_path(model_name: str) -> Path:
    return CHECKPOINT_DIR / f"diffusion_{model_name}.pt"


def train(cfg: DiffusionConfig, limit_batches: int | None = None):
    device = get_device()
    conditional = cfg.model == "transformer"
    model = MODELS[cfg.model]().to(device)
    print(f"Training {cfg.model}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters, {cfg}")

    loader = DataLoader(mnist(train=True), batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = LambdaLR(optimizer, lambda epoch: max(0.2, 0.98**epoch)) if cfg.lr_decay else None
    path = checkpoint_path(cfg.model)
    path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.epochs):
        model.train()
        total, n = 0.0, 0
        for b, (x, y) in enumerate(loader):
            if limit_batches is not None and b >= limit_batches:
                break
            x, y = x.to(device), y.to(device)
            loss = score_matching_loss(model, x, y if conditional else None, conditional_t_range=conditional)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x)
            n += len(x)
        if scheduler:
            scheduler.step()
        print(f"Epoch {epoch}: average loss {total / max(n, 1):.4f}, lr {optimizer.param_groups[0]['lr']:.1e}")
        torch.save({"model": cfg.model, "state_dict": model.state_dict()}, path)
    print(f"Saved model to {path}")
    return model


def load(model_name: str, device=None, path: Path | None = None):
    device = device or get_device()
    ckpt = torch.load(path or checkpoint_path(model_name), map_location=device)
    model = MODELS[ckpt["model"]]().to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model


def sample(model_name: str, n: int = 64, num_steps: int | None = None, digit: int | None = None,
           path: Path | None = None) -> Path:
    device = get_device()
    model = load(model_name, device, path)
    y = None
    if model_name == "transformer":
        digit = 7 if digit is None else digit
        y = torch.full((n,), digit, dtype=torch.long, device=device)
    steps = num_steps or (250 if model_name == "transformer" else 500)
    samples = euler_maruyama_sampler(model, n, steps, device, y=y)
    name = f"diffusion_{model_name}_samples" + (f"_digit{digit}" if y is not None else "")
    out = save_grid(samples, OUTPUT_DIR / f"{name}.png", nrow=int(np.sqrt(n)))
    print(f"Saved {out}")
    return out
