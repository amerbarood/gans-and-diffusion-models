"""Fast shape and loss checks on random tensors; no downloads."""

import torch

from genmodels import adversarial, diffusion, gan


def test_gan_shapes_and_losses():
    G, D = gan.Generator(n_classes=10), gan.Discriminator(n_classes=10)
    z, c = torch.randn(4, 100), torch.eye(10)[:4]
    fake = G(z, c)
    assert fake.shape == (4, 784) and fake.min() >= 0 and fake.max() <= 1
    logits = D(fake, c)
    assert logits.shape == (4, 1)
    for kind in ("minimax", "non-saturating"):
        assert torch.isfinite(gan.generator_loss(logits, kind))
    # Minimax generator loss is log(1 - D): always <= 0.
    assert gan.generator_loss(logits, "minimax") <= 0


def test_cnn_and_fgsm():
    model = adversarial.CNN()
    x = torch.rand(1, 1, 28, 28)
    assert model(x).shape == (1, 10)
    x_adv, grad = adversarial.targeted_fgsm(model, x, target=9, epsilon=0.2)
    assert x_adv.shape == x.shape and (x_adv - x).abs().max() <= 0.2 + 1e-6


def test_unets_and_sampler():
    x, t = torch.rand(2, 1, 28, 28), torch.rand(2) * 0.9 + 0.05
    for name, cls in diffusion.MODELS.items():
        model = cls()
        y = torch.tensor([3, 7]) if name == "transformer" else None
        assert model(x, t, y=y).shape == x.shape
        loss = diffusion.score_matching_loss(model, x, y, conditional_t_range=y is not None)
        assert torch.isfinite(loss)
    samples = diffusion.euler_maruyama_sampler(diffusion.UNetRes(), batch_size=2, num_steps=3)
    assert samples.shape == (2, 1, 28, 28)
