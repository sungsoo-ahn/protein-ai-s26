---
layout: post
title: "Code Walkthrough: nano-polymer-diffusion"
date: 2026-03-23
description: "Build a denoising diffusion model (DDPM) for 2D bead-spring polymers — noise schedule, MLP denoiser with timestep embedding, and side-by-side comparison with the VAE."
course: "2026-spring-protein-ai"
course_title: "Protein & Artificial Intelligence"
course_semester: "Spring 2026"
lecture_number: 14
preliminary: false
toc:
  sidebar: left
related_posts: false
---

<div style="background: var(--global-code-bg-color); border-left: 4px solid var(--global-theme-color); padding: 1em 1.2em; margin-bottom: 2em; border-radius: 4px;">
<strong>Code:</strong> Self-contained notebook / script<br>
<strong>Parameters:</strong> ~250K &nbsp;|&nbsp; <strong>Data:</strong> ~10K bead-spring polymer conformations from <a href="https://dmol.pub/dl/VAE.html">dmol.pub</a>
</div>

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Companion to <a href="{{ '/lectures/17-diffusion/' | relative_url }}">Lecture 4: Diffusion Models for Protein Generation</a>. The lecture develops the DDPM framework — forward process, noise prediction, reverse denoising. This page applies every concept to the same bead-spring polymer from <a href="{{ '/lectures/18-nano-polymer-vae/' | relative_url }}">nano-polymer-vae</a>, enabling a direct comparison between two generative frameworks on identical data.</em>
</p>

The [nano-polymer-vae]({{ '/lectures/18-nano-polymer-vae/' | relative_url }}) walkthrough trained a VAE on 2D bead-spring polymers — 12 beads, 24 coordinates, generation via a single decoder pass. Here we train a DDPM on the exact same preprocessed data. Same polymer, same preprocessing, different generative model. By the end, we can put VAE and DDPM samples side by side and see what each framework does well.

## 1. Same Data, Different Generator

We reuse the preprocessed polymer data from the VAE walkthrough: center-of-mass subtracted, PCA-aligned, flattened to 24D vectors.

```python
import numpy as np
import urllib.request
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt

# --- Data loading (identical to nano-polymer-vae) ---
urllib.request.urlretrieve(
    "https://github.com/whitead/dmol-book/raw/main/data/long_paths.npz",
    "long_paths.npz",
)
paths = np.load("long_paths.npz")["arr"]


def center_com(paths):
    return paths - np.mean(paths, axis=-2, keepdims=True)


def find_principal_axis(points):
    inertia = points.T @ points
    evals, evecs = np.linalg.eigh(inertia)
    return evecs[:, np.argmax(evals)]


def make_2d_rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def align_principal(paths):
    aligned = np.zeros_like(paths)
    for i, p in enumerate(paths):
        axis = find_principal_axis(p)
        angle = -np.arctan2(axis[1], axis[0])
        aligned[i] = p @ make_2d_rotation(angle).T
    return aligned


data = align_principal(center_com(paths))
flat_data = data.reshape(-1, 24).astype(np.float32)

np.random.seed(42)
idx = np.random.permutation(len(flat_data))
flat_data = flat_data[idx]

split = int(0.9 * len(flat_data))
train_data = torch.from_numpy(flat_data[:split])
test_data = torch.from_numpy(flat_data[split:])

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=256, shuffle=True
)
print(f"Train: {len(train_data)}, Test: {len(test_data)}")
```

The data is the same 24-dimensional vectors. The only difference is the model that will learn to generate them.

## 2. Noise Schedule

The forward process gradually corrupts clean polymer coordinates with Gaussian noise over $$T = 200$$ timesteps. We use a linear schedule for $$\beta_t$$ from $$10^{-4}$$ to $$0.02$$:

```python
class NoiseSchedule:
    """Precompute all noise schedule quantities for efficient training/sampling."""

    def __init__(self, T=200, beta_start=1e-4, beta_end=0.02):
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

    def add_noise(self, x0, t, noise=None):
        """q(x_t | x_0) — jump directly to any timestep."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alpha_bars[t].unsqueeze(-1)
        sqrt_1m_ab = self.sqrt_one_minus_alpha_bars[t].unsqueeze(-1)
        return sqrt_ab * x0 + sqrt_1m_ab * noise

schedule = NoiseSchedule(T=200)
```

> **Why $$T = 200$$ instead of 1000?** Our data is 24-dimensional — far simpler than images. Fewer steps suffice because each denoising step removes noise from a low-dimensional space. This keeps training fast (~2 min) and sampling quick (~0.5s for a batch).

Visualize the forward process on a single polymer:

```python
example = train_data[0]
fig, axes = plt.subplots(1, 6, figsize=(18, 3))
timesteps = [0, 20, 50, 100, 150, 199]
for ax, t in zip(axes, timesteps):
    if t == 0:
        noisy = example
    else:
        noisy = schedule.add_noise(
            example.unsqueeze(0),
            torch.tensor([t]),
        ).squeeze(0)
    coords = noisy.numpy().reshape(12, 2)
    ax.plot(coords[:, 0], coords[:, 1], "o-", markersize=4)
    ax.set_title(f"t = {t}")
    ax.set_aspect("equal")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.axis("off")
plt.suptitle("Forward process: polymer → noise")
plt.tight_layout()
plt.show()
```

## 3. Denoiser Architecture

The denoiser is an MLP that takes a noisy 24D coordinate vector and a 64D sinusoidal timestep embedding as input, and predicts the noise $$\epsilon$$ that was added. Three hidden layers of 256 units — roughly the same capacity as the VAE.

**Sinusoidal timestep embedding** (identical to the lecture):

```python
class SinusoidalEmbedding(nn.Module):
    """Encode scalar timestep t into a d-dimensional vector."""

    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
```

**Denoiser MLP:**

```python
class Denoiser(nn.Module):
    """MLP noise predictor: (x_t, t) → predicted noise ε."""

    def __init__(self, data_dim=24, time_dim=64, hidden_dim=256):
        super().__init__()
        self.time_embed = SinusoidalEmbedding(time_dim)
        self.net = nn.Sequential(
            nn.Linear(data_dim + time_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, x_t, t):
        t_emb = self.time_embed(t)
        return self.net(torch.cat([x_t, t_emb], dim=-1))

denoiser = Denoiser()
total_params = sum(p.numel() for p in denoiser.parameters())
print(f"Total parameters: {total_params:,}")  # ~250K
```

> **Why concatenate instead of add?** For low-dimensional data (24D), concatenation with the 64D timestep embedding is simple and effective. For high-dimensional data (images), adding the timestep embedding to intermediate feature maps is standard because it avoids increasing the input dimension of every layer.

## 4. Training

The training objective is noise-prediction MSE: sample a random timestep, corrupt the data, predict the noise, minimize the squared error.

```python
def train_step(model, x0, schedule, optimizer):
    """One training step: noise prediction loss."""
    batch_size = x0.size(0)
    t = torch.randint(0, schedule.T, (batch_size,))
    noise = torch.randn_like(x0)
    x_t = schedule.add_noise(x0, t, noise)
    noise_pred = model(x_t, t)
    loss = nn.functional.mse_loss(noise_pred, noise)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

Full training loop:

```python
optimizer = torch.optim.Adam(denoiser.parameters(), lr=1e-3)
epochs = 200

for epoch in range(epochs):
    denoiser.train()
    total_loss = 0
    for batch in train_loader:
        total_loss += train_step(denoiser, batch, schedule, optimizer)

    if (epoch + 1) % 40 == 0:
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1:3d} | loss={avg_loss:.6f}")
```

Training converges in about 200 epochs. The loss should drop to roughly 0.3–0.5 (the model can't perfectly predict the noise, but captures the data distribution well).

## 5. Sampling

Generation reverses the forward process: start from pure noise $$x_T \sim \mathcal{N}(0, I)$$, apply the learned denoising step 200 times:

```python
@torch.no_grad()
def ddpm_sample(model, schedule, n_samples=16):
    """Generate samples via DDPM reverse process."""
    x = torch.randn(n_samples, 24)

    for t in reversed(range(schedule.T)):
        t_batch = torch.full((n_samples,), t, dtype=torch.long)
        noise_pred = model(x, t_batch)

        alpha = schedule.alphas[t]
        alpha_bar = schedule.alpha_bars[t]
        beta = schedule.betas[t]

        mean = (1.0 / torch.sqrt(alpha)) * (
            x - (beta / torch.sqrt(1.0 - alpha_bar)) * noise_pred
        )

        if t > 0:
            x = mean + torch.sqrt(beta) * torch.randn_like(x)
        else:
            x = mean

    return x
```

Generate and visualize:

```python
denoiser.eval()
samples = ddpm_sample(denoiser, schedule, n_samples=8).numpy().reshape(-1, 12, 2)

fig, axes = plt.subplots(1, 8, figsize=(20, 3))
for i, ax in enumerate(axes):
    ax.plot(samples[i, :, 0], samples[i, :, 1], "o-", markersize=4)
    ax.set_aspect("equal")
    ax.axis("off")
plt.suptitle("DDPM-generated polymer conformations")
plt.tight_layout()
plt.show()
```

Visualize intermediate denoising steps to see the polymer emerge from noise:

```python
@torch.no_grad()
def ddpm_sample_trajectory(model, schedule):
    """Sample one polymer, recording intermediate states."""
    x = torch.randn(1, 24)
    trajectory = [x.numpy().reshape(12, 2)]

    for t in reversed(range(schedule.T)):
        t_batch = torch.full((1,), t, dtype=torch.long)
        noise_pred = model(x, t_batch)
        alpha = schedule.alphas[t]
        alpha_bar = schedule.alpha_bars[t]
        beta = schedule.betas[t]
        mean = (1.0 / torch.sqrt(alpha)) * (
            x - (beta / torch.sqrt(1.0 - alpha_bar)) * noise_pred
        )
        if t > 0:
            x = mean + torch.sqrt(beta) * torch.randn_like(x)
        else:
            x = mean
        if t % 33 == 0 or t == 0:
            trajectory.append(x.numpy().reshape(12, 2))

    return trajectory

traj = ddpm_sample_trajectory(denoiser, schedule)
fig, axes = plt.subplots(1, len(traj), figsize=(3 * len(traj), 3))
for i, (ax, coords) in enumerate(zip(axes, traj)):
    ax.plot(coords[:, 0], coords[:, 1], "o-", markersize=4)
    ax.set_aspect("equal")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.axis("off")
axes[0].set_title("t = T (noise)")
axes[-1].set_title("t = 0 (clean)")
plt.suptitle("Reverse process: noise → polymer")
plt.tight_layout()
plt.show()
```

## 6. VAE vs. DDPM

Both models were trained on the same preprocessed polymer data with similar parameter counts (~250K). Here we compare them directly.

### Generation speed

```python
import time

# Time VAE generation (if you have the VAE model from nano-polymer-vae)
# vae_start = time.time()
# with torch.no_grad():
#     z = torch.randn(1000, 2)
#     vae_samples = vae_model.decoder(z)
# vae_time = time.time() - vae_start

# Time DDPM generation
ddpm_start = time.time()
ddpm_samples = ddpm_sample(denoiser, schedule, n_samples=1000)
ddpm_time = time.time() - ddpm_start
print(f"DDPM: {ddpm_time:.2f}s for 1000 samples")
# VAE would be ~0.01s — orders of magnitude faster
```

### Sample quality

Compare generated distributions against the training data by looking at the radius of gyration:

```python
def radius_of_gyration(coords):
    """Rg for each conformation. coords: (batch, 12, 2)."""
    com = coords.mean(axis=1, keepdims=True)
    return np.sqrt(np.mean(np.sum((coords - com) ** 2, axis=-1), axis=-1))

# Generate DDPM samples
denoiser.eval()
ddpm_gen = ddpm_sample(denoiser, schedule, n_samples=2000).numpy().reshape(-1, 12, 2)
rg_ddpm = radius_of_gyration(ddpm_gen)
rg_train = radius_of_gyration(data[idx[:split]])

plt.figure(figsize=(8, 4))
plt.hist(rg_train, bins=50, alpha=0.5, density=True, label="Training data")
plt.hist(rg_ddpm, bins=50, alpha=0.5, density=True, label="DDPM samples")
plt.xlabel("Radius of gyration ($R_g$)")
plt.ylabel("Density")
plt.legend()
plt.title("$R_g$ distribution: training data vs. DDPM")
plt.tight_layout()
plt.show()
```

### Summary

| Aspect | VAE | DDPM |
|--------|-----|------|
| **Generation** | One decoder pass (~0.01s / 1000 samples) | 200 denoising steps (~2s / 1000 samples) |
| **Latent space** | 2D, interpretable ($$R_g$$ gradient visible) | No explicit latent space |
| **Property optimization** | Gradient descent in latent space | Requires classifier guidance |
| **Sample quality** | Good; slightly blurry | Sharp; captures fine details |
| **Parameters** | ~250K | ~250K |
| **Training** | 150 epochs | 200 epochs |

The VAE excels at **interpretability and speed** — the 2D latent space gives you a map of polymer shapes, and generation is a single forward pass. The DDPM excels at **sample quality** — each denoising step refines the output, producing sharper conformations. For this toy system, both work well. For real proteins, the choice depends on whether you need fast screening (VAE) or high-fidelity generation (diffusion).

## Key Takeaways

- The same polymer data that the VAE models in a single encoder-decoder pass, the **DDPM decomposes into 200 small denoising steps** — a fundamentally different generative strategy.
- A **sinusoidal timestep embedding** tells the denoiser how much noise to expect at each step, reusing the same positional encoding idea from transformers.
- **Fewer diffusion steps** (200 vs. 1000) suffice for low-dimensional data — always match the schedule to your data complexity.
- **Direct comparison on identical data** reveals the trade-off: VAEs trade sample sharpness for speed and interpretability; DDPMs trade speed for quality.
