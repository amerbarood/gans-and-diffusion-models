"""Adversarial examples: train a CNN on MNIST, then fool it with a targeted
fast-gradient-sign perturbation (ported from the TensorFlow notebook 2.4)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from .utils import CHECKPOINT_DIR, OUTPUT_DIR, get_device, mnist  # noqa: E402


class CNN(nn.Module):
    """conv5x5(32) - pool - conv5x5(64) - pool - fc(1024) - dropout(0.5) - fc(10)."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(nn.Linear(64 * 4 * 4, 1024), nn.ReLU(), nn.Dropout(0.5), nn.Linear(1024, 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x.view(-1, 1, 28, 28)))


def train_classifier(epochs: int = 4, batch_size: int = 100, lr: float = 1e-3, device=None) -> CNN:
    device = device or get_device()
    model = CNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(mnist(train=True), batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        model.train()
        correct = total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
        print(f"Epoch {epoch}, loss: {loss.item():.4f}, train accuracy: {correct / total:.4f}")
    return model


@torch.no_grad()
def test_accuracy(model: CNN, device) -> float:
    model.eval()
    loader = DataLoader(mnist(train=False), batch_size=1000)
    correct = sum((model(x.to(device)).argmax(1).cpu() == y).sum().item() for x, y in loader)
    acc = correct / len(loader.dataset)
    print(f"Test accuracy: {acc:.4f}")
    return acc


def targeted_fgsm(model: CNN, x: torch.Tensor, target: int, epsilon: float = 0.2, n_steps: int = 1):
    """Push `x` towards class `target`: x <- clip(x - eps * sign(dL/dx)), with L the
    cross-entropy against the target label. Returns (adversarial image, last gradient)."""
    model.eval()
    x_adv = x.clone()
    target_t = torch.tensor([target], device=x.device)
    for _ in range(n_steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), target_t)
        grad, = torch.autograd.grad(loss, x_adv)
        x_adv = (x_adv.detach() - torch.sign(grad) * (epsilon / n_steps)).clamp(0, 1)
    return x_adv.detach(), grad.detach()


@torch.no_grad()
def predict(model: CNN, x: torch.Tensor) -> tuple[int, float]:
    prob = F.softmax(model(x), dim=1)[0]
    return int(prob.argmax()), float(prob.max()) * 100


def plot_attack(model, original, gradient, adversarial, path: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.4))
    label, pct = predict(model, original)
    adv_label, adv_pct = predict(model, adversarial)
    panels = [
        (original, f"Label: {label}\nCertainty: {pct:.1f}%"),
        (gradient, "Adversarial\nPartial Derivatives"),
        (adversarial, f"Label: {adv_label}\nCertainty: {adv_pct:.1f}%"),
    ]
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img.detach().cpu().view(28, 28).numpy())
        ax.set_title(title)
        ax.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")
    return path


def run(source_digit: int = 4, target_digit: int = 9, epsilon: float = 0.2, epochs: int = 4, seed: int | None = None):
    device = get_device()
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    model = train_classifier(epochs=epochs, device=device)
    test_accuracy(model, device)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_DIR / "mnist_cnn.pt")

    test = mnist(train=False)
    candidates = (test.targets == source_digit).nonzero().flatten().tolist()
    np.random.shuffle(candidates)
    # Pick a correctly classified example, as in the notebook figure.
    for i in candidates:
        x = test[i][0].unsqueeze(0).to(device)
        if predict(model, x)[0] == source_digit:
            break
    x_adv, grad = targeted_fgsm(model, x, target_digit, epsilon)
    before, after = predict(model, x), predict(model, x_adv)
    print(f"Original: predicted {before[0]} ({before[1]:.1f}%)  ->  adversarial: predicted {after[0]} ({after[1]:.1f}%)")
    plot_attack(model, x, grad, x_adv, OUTPUT_DIR / f"adversarial_{source_digit}_to_{target_digit}.png")
    return before, after
