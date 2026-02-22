---
layout: post
title: "Code Walkthrough: nano-polymer-vae"
date: 2026-03-23
description: "Build a variational autoencoder for 2D bead-spring polymers — data preprocessing, encoder-decoder MLP, latent-space visualization, and property optimization."
course: "2026-spring-protein-ai"
course_title: "Protein & Artificial Intelligence"
course_semester: "Spring 2026"
lecture_number: 13
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
<em>Companion to <a href="{{ '/lectures/06-generative-models/' | relative_url }}">Lecture 3: Variational Autoencoders for Proteins</a>. The lecture develops the VAE framework — encoder, decoder, ELBO, reparameterization trick. This page applies every concept to a concrete toy system: a 2D bead-spring polymer with 12 beads and 24 coordinates. Code translated from JAX (<a href="https://dmol.pub/dl/VAE.html">dmol.pub</a>) to PyTorch.</em>
</p>

Real proteins live in high-dimensional spaces — thousands of atoms in 3D, 20 amino acid types, complex energy landscapes. That complexity makes it hard to *see* what a VAE is doing. A bead-spring polymer strips away everything except the geometry: 12 point masses connected by springs in 2D, producing a 24-dimensional coordinate vector. The data is small enough to train on a laptop in minutes, and the 2D latent space can be visualized directly.

## 1. The Bead-Spring Polymer

A bead-spring polymer is a coarse-grained model where each "bead" represents a monomer unit connected to its neighbors by harmonic springs. The conformations in our dataset come from a molecular dynamics simulation: the polymer wriggles, stretches, and coils, sampling a distribution of shapes governed by thermal fluctuations.

Each conformation is 12 beads × 2 coordinates = 24 numbers. The dataset contains roughly 10,000 such snapshots.

```python
import numpy as np
import urllib.request
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

urllib.request.urlretrieve(
    "https://github.com/whitead/dmol-book/raw/main/data/long_paths.npz",
    "long_paths.npz",
)
paths = np.load("long_paths.npz")["arr"]
print(f"Shape: {paths.shape}")  # (N, 12, 2)
```

Visualize a few conformations to build intuition:

```python
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i, ax in enumerate(axes):
    ax.plot(paths[i * 1000, :, 0], paths[i * 1000, :, 1], "o-", markersize=4)
    ax.set_aspect("equal")
    ax.set_title(f"Frame {i * 1000}")
    ax.axis("off")
plt.tight_layout()
plt.show()
```

Each polymer looks like a short, wiggly chain. Some are compact (coiled), others are extended (stretched). This variation is exactly what the VAE will learn to capture.

## 2. Preprocessing

Raw coordinates contain three kinds of nuisance variation that have nothing to do with shape: **translation** (where the polymer sits in the plane), **rotation** (which direction it points), and **reflection**. We remove translation and rotation before training.

**Center of mass subtraction** removes translation:

```python
def center_com(paths):
    """Subtract the center of mass from each conformation."""
    coms = np.mean(paths, axis=-2, keepdims=True)
    return paths - coms
```

**PCA alignment** removes rotation. For each conformation, we find the principal axis (the direction of maximum spread) and rotate the polymer so this axis aligns with the x-axis:

```python
def find_principal_axis(points):
    """Find the direction of maximum variance (first principal component)."""
    inertia = points.T @ points
    evals, evecs = np.linalg.eigh(inertia)
    return evecs[:, np.argmax(evals)]


def make_2d_rotation(angle):
    """2D rotation matrix for a given angle."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def align_principal(paths):
    """Rotate each conformation so its principal axis aligns with x-axis."""
    aligned = np.zeros_like(paths)
    for i, p in enumerate(paths):
        axis = find_principal_axis(p)
        angle = -np.arctan2(axis[1], axis[0])
        aligned[i] = p @ make_2d_rotation(angle).T
    return aligned
```

Apply both transformations and flatten to 24-dimensional vectors:

```python
data = align_principal(center_com(paths))
flat_data = data.reshape(-1, 24).astype(np.float32)
print(f"Preprocessed data: {flat_data.shape}")  # (N, 24)
```

> **Why flatten?** Our VAE uses fully connected layers (MLPs), which expect 1D input vectors. Flattening 12×2 to 24 is lossless — the network just sees 24 numbers. For larger systems (real proteins), you would use architectures that respect the spatial structure (GNNs, equivariant networks).

**Train/test split:**

```python
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

## 3. VAE Architecture

The architecture mirrors the dmol.pub example, translated to PyTorch. Both encoder and decoder are 3-hidden-layer MLPs with 256 units per layer.

**Encoder:** maps 24D coordinates → 2D latent parameters (mean $$\mu$$ and standard deviation $$\sigma$$). The standard deviation is parameterized via softplus to guarantee positivity:

```python
class Encoder(nn.Module):
    def __init__(self, input_dim=24, hidden_dim=256, latent_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_sigma = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.net(x)
        mu = self.fc_mu(h)
        sigma = nn.functional.softplus(self.fc_sigma(h))  # sigma > 0
        return mu, sigma
```

> **softplus vs. log-variance.** Lecture 3 parameterizes the encoder with log-variance ($$\log \sigma^2$$) and exponentiates to get $$\sigma$$. The dmol.pub source uses softplus ($$\log(1 + e^x)$$) to directly output $$\sigma$$. Both guarantee positive standard deviation — softplus is smoother near zero, log-variance is more common in the literature.

**Decoder:** maps 2D latent → 24D reconstructed coordinates:

```python
class Decoder(nn.Module):
    def __init__(self, latent_dim=2, hidden_dim=256, output_dim=24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z):
        return self.net(z)
```

**Full VAE:** combines encoder, reparameterization trick, and decoder:

```python
class PolymerVAE(nn.Module):
    def __init__(self, input_dim=24, hidden_dim=256, latent_dim=2):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def reparameterize(self, mu, sigma):
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

    def forward(self, x):
        mu, sigma = self.encoder(x)
        z = self.reparameterize(mu, sigma)
        x_recon = self.decoder(z)
        return x_recon, mu, sigma

model = PolymerVAE()
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")  # ~250K
```

## 4. Training

The loss is the $$\beta$$-weighted ELBO: MSE reconstruction plus $$\beta \cdot$$ KL divergence. Setting $$\beta = 0.01$$ emphasizes reconstruction — the polymer coordinates must be faithfully reconstructed, with only gentle pressure to keep the latent space organized.

```python
def vae_loss(x, x_recon, mu, sigma, beta=0.01):
    """Beta-VAE loss = MSE reconstruction + beta * KL divergence."""
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction="mean")
    # KL(N(mu, sigma^2) || N(0, I))
    kl_loss = -0.5 * torch.mean(
        1 + 2 * torch.log(sigma + 1e-8) - mu.pow(2) - sigma.pow(2)
    )
    return recon_loss + beta * kl_loss, recon_loss, kl_loss
```

> **Why $$\beta = 0.01$$?** With $$\beta = 1$$ (standard VAE), the KL term dominates early in training: the model collapses the latent space to $$\mathcal{N}(0, I)$$ before learning to reconstruct. For continuous coordinate data with MSE loss, the reconstruction and KL losses live on different scales. A small $$\beta$$ lets the model first learn good reconstructions, then gradually organize the latent space. This is the $$\beta$$-VAE trick (Higgins et al., 2017).

Training loop:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 150

for epoch in range(epochs):
    model.train()
    total_loss, total_recon, total_kl = 0, 0, 0
    for batch in train_loader:
        optimizer.zero_grad()
        x_recon, mu, sigma = model(batch)
        loss, recon, kl = vae_loss(batch, x_recon, mu, sigma)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()

    n = len(train_loader)
    if (epoch + 1) % 25 == 0:
        print(
            f"Epoch {epoch+1:3d} | "
            f"loss={total_loss/n:.4f}  "
            f"recon={total_recon/n:.4f}  "
            f"kl={total_kl/n:.4f}"
        )
```

After 150 epochs, reconstruction loss should drop to roughly 0.01–0.05 (coordinates are well-recovered) while KL stays moderate (the latent space is not fully collapsed).

## 5. Generation and Latent Space

### Sampling new polymers

Generation is a single decoder pass: sample $$z \sim \mathcal{N}(0, I)$$, decode to 24D, reshape to 12×2:

```python
model.eval()
with torch.no_grad():
    z_sample = torch.randn(8, 2)
    generated = model.decoder(z_sample).numpy().reshape(-1, 12, 2)

fig, axes = plt.subplots(1, 8, figsize=(20, 3))
for i, ax in enumerate(axes):
    ax.plot(generated[i, :, 0], generated[i, :, 1], "o-", markersize=4)
    ax.set_aspect("equal")
    ax.axis("off")
plt.suptitle("Generated polymer conformations")
plt.tight_layout()
plt.show()
```

### Latent space visualization

With a 2D latent space, we can directly plot every training conformation colored by a physical property. The **radius of gyration** $$R_g$$ measures how compact or extended a polymer is:

$$R_g = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \|\mathbf{r}_i - \mathbf{r}_{\mathrm{com}}\|^2}$$

```python
def radius_of_gyration(coords):
    """Compute Rg for each conformation. coords: (batch, 12, 2)."""
    com = coords.mean(axis=1, keepdims=True)
    return np.sqrt(np.mean(np.sum((coords - com) ** 2, axis=-1), axis=-1))


# Encode all training data
model.eval()
with torch.no_grad():
    mu_all, _ = model.encoder(train_data)
    mu_all = mu_all.numpy()

# Compute Rg for coloring
rg_values = radius_of_gyration(data[idx[:split]])

plt.figure(figsize=(8, 6))
sc = plt.scatter(mu_all[:, 0], mu_all[:, 1], c=rg_values, cmap="viridis",
                 s=1, alpha=0.5)
plt.colorbar(sc, label="Radius of gyration ($R_g$)")
plt.xlabel("$z_1$")
plt.ylabel("$z_2$")
plt.title("Latent space colored by $R_g$")
plt.tight_layout()
plt.show()
```

The scatter plot reveals that the latent space is **physically meaningful**: compact polymers (low $$R_g$$) cluster in one region, extended polymers (high $$R_g$$) in another. The VAE discovered this organization on its own — no $$R_g$$ labels were used during training.

### Latent space interpolation

Walk between two points in latent space to smoothly morph one polymer conformation into another:

```python
model.eval()
with torch.no_grad():
    # Pick two training conformations
    z1, _ = model.encoder(train_data[0:1])
    z2, _ = model.encoder(train_data[100:1])

    # Interpolate
    alphas = torch.linspace(0, 1, 8).unsqueeze(1)
    z_interp = (1 - alphas) * z1 + alphas * z2
    interp_coords = model.decoder(z_interp).numpy().reshape(-1, 12, 2)

fig, axes = plt.subplots(1, 8, figsize=(20, 3))
for i, ax in enumerate(axes):
    ax.plot(interp_coords[i, :, 0], interp_coords[i, :, 1], "o-", markersize=4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"α={i/7:.1f}")
plt.suptitle("Latent space interpolation")
plt.tight_layout()
plt.show()
```

Smooth interpolation is a hallmark of a well-trained VAE: every point along the path decodes to a physically plausible polymer.

## 6. Property Optimization

The structured latent space enables **property optimization**: find polymer conformations with extreme properties by gradient descent *in the latent space*.

Target: minimize the radius of gyration (find the most compact polymer the model can generate).

```python
# Start from a random latent point
z = torch.randn(2, requires_grad=True)
optimizer_z = torch.optim.Adam([z], lr=0.01)

model.eval()
rg_history = []

for step in range(200):
    optimizer_z.zero_grad()
    coords = model.decoder(z).reshape(12, 2)
    rg = torch.sqrt(torch.mean(torch.sum(coords ** 2, dim=-1)))
    rg.backward()
    optimizer_z.step()
    rg_history.append(rg.item())

# Plot optimization trajectory
plt.figure(figsize=(6, 4))
plt.plot(rg_history)
plt.xlabel("Optimization step")
plt.ylabel("$R_g$")
plt.title("Radius of gyration optimization in latent space")
plt.tight_layout()
plt.show()
```

Compare the optimized structure with the most compact training example:

```python
with torch.no_grad():
    optimized = model.decoder(z).numpy().reshape(12, 2)

# Most compact in training set
rg_train = radius_of_gyration(data[idx[:split]])
most_compact_idx = np.argmin(rg_train)
most_compact = data[idx[most_compact_idx]]

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].plot(most_compact[:, 0], most_compact[:, 1], "o-", markersize=6)
axes[0].set_title(f"Training (Rg={rg_train[most_compact_idx]:.2f})")
axes[0].set_aspect("equal")

axes[1].plot(optimized[:, 0], optimized[:, 1], "o-", markersize=6, color="C1")
axes[1].set_title(f"Optimized (Rg={rg_history[-1]:.2f})")
axes[1].set_aspect("equal")

plt.tight_layout()
plt.show()
```

The optimized polymer is more compact than anything in the training set — the VAE has *extrapolated* beyond the data. This is the same principle behind protein design: encode known proteins, then navigate the latent space toward desired properties (binding affinity, stability, solubility).

## Key Takeaways

- A **bead-spring polymer** (12 beads, 24 coordinates) is a minimal testbed for generative models — small enough to train in minutes, rich enough to illustrate every VAE concept.
- **Preprocessing matters:** center-of-mass subtraction and PCA alignment remove translation/rotation, letting the model focus on shape.
- The **$$\beta$$-VAE** with $$\beta = 0.01$$ balances reconstruction against latent-space organization for continuous coordinate data.
- A **2D latent space** reveals physically meaningful structure ($$R_g$$ gradient) without any property labels during training.
- **Latent-space optimization** finds novel conformations with extreme properties — the same principle scales to real proteins.

The companion page [nano-polymer-diffusion]({{ '/lectures/19-nano-polymer-diffusion/' | relative_url }}) applies a denoising diffusion model to the same data, enabling a direct comparison between the two generative frameworks.
