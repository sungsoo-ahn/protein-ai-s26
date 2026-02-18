---
layout: post
title: "Generative Models: VAEs and Diffusion for Proteins"
date: 2026-03-23
description: "Variational autoencoders and denoising diffusion models—two frameworks for generating novel proteins, from the ELBO derivation to the denoising score-matching objective."
course: "2026-spring-protein-ai"
course_title: "Protein & Artificial Intelligence"
course_semester: "Spring 2026"
lecture_number: 2
preliminary: false
toc:
  sidebar: left
related_posts: false
collapse_code: true
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;"><em>This is Lecture 2 of the Protein & Artificial Intelligence course (Spring 2026), co-taught by Prof. Sungsoo Ahn and Prof. Homin Kim at KAIST. The course covers core machine learning techniques for protein science, from representation learning to generative design. In this lecture we shift from discriminative models—which predict properties of existing proteins—to generative models that can imagine entirely new ones.</em></p>

## Introduction: Dreaming Up New Proteins

Generative models have transformed machine learning.  In computer vision, they synthesize photorealistic faces, fill in missing image regions, and transfer artistic styles.  In NLP, large language models generate coherent text, translate between languages, and write code.  The same generative paradigm is now reshaping protein science.

Evolution has spent roughly four billion years crafting the proteins we observe today.
These molecules are the result of relentless natural selection—optimized for the particular environments and challenges their host organisms faced.
Yet the proteins that exist in nature represent only a vanishing sliver of what is possible.
A polypeptide chain of length 100 can be assembled from 20 standard amino acids in $$20^{100} \approx 10^{130}$$ distinct ways, a number that dwarfs the roughly $$10^{80}$$ atoms in the observable universe.

What lies in that unexplored space?
Enzymes that degrade plastic pollutants.
Binders that neutralize pandemic viruses.
Molecular machines that catalyze reactions no natural enzyme has ever performed.
Generative models give us a systematic way to explore this vast, uncharted territory.
Instead of waiting for evolution to stumble upon useful proteins through random mutation, we can train machine learning models to internalize the statistical patterns that make proteins work—and then sample novel sequences and structures from the learned distribution.

In Lecture 1 we studied **transformers** and **graph neural networks**, architectures that process protein sequences and structures.
Those were discriminative tools: given a protein, predict a property.
This lecture flips the direction.
We ask: *given a desired property or no constraint at all, can we generate a protein that satisfies it?*

We will study two foundational frameworks for generative modeling—**variational autoencoders (VAEs)** and **denoising diffusion probabilistic models (DDPMs)**—and see how each has been applied to design proteins that have never existed in nature.

### Roadmap

| Section | Topic | Why It Is Needed |
|---------|-------|------------------|
| 1 | The Generation Problem | Sets up the core challenge: training a noise-to-protein decoder |
| 2 | The Variational Autoencoder | Introduces encoder, decoder, and KL regularization as one coherent idea |
| 3 | The ELBO: Formalizing the Training Objective | Derives the principled training objective from maximum likelihood |
| 4 | The Reparameterization Trick | Solves the practical problem of backpropagating through sampling |
| 5 | Diffusion Models: Controlled Destruction | Introduces the forward noising process |
| 6 | The Reverse Process and Training Objective | Shows how the network learns to denoise |
| 7 | The Denoising Loop and Architecture | Covers generation and timestep conditioning |
| 8 | VAEs vs. Diffusion | Compares strengths, weaknesses, and computational trade-offs |
| 9 | Conditional Generation | Classifier guidance and classifier-free guidance for steering generation |

---

## 1. The Generation Problem

The goal is concrete: build a machine that takes random noise as input and outputs a novel, realistic protein sequence.

Suppose you have a database of 50,000 serine protease sequences.
Despite their diversity—some share less than 30% sequence identity—they all fold into similar structures, catalyze the same reaction, and place a conserved catalytic triad (Ser, His, Asp) in nearly identical spatial positions.
You want a neural network that can generate *new* serine proteases—sequences not in the database, but statistically indistinguishable from those that are.

The architecture is simple: a **decoder** network $$f_\theta$$ that maps a random vector $$z$$ to a protein sequence $$x$$.
At inference time, we sample $$z \sim \mathcal{N}(0, I)$$, feed it through the decoder, and read off the generated sequence.
Different draws of $$z$$ produce different proteins; the distribution of outputs should match the distribution of real proteins.

The problem is training.
We have 50,000 real serine proteases, but we do not know which noise vector $$z$$ should map to which protein.
The decoder expects an input $$z$$ and must produce the corresponding protein $$x$$—but the "corresponding" noise for each training protein is unknown.
We cannot simply pair random noise vectors with training sequences, because there is no reason a random $$z$$ should have anything to do with a particular protein.

This is the central challenge: **how do you train a noise-to-data decoder when you only have data and no corresponding noise inputs?**

[^latent]: The latent code is also called the *latent representation*, *latent variable*, or *embedding*, depending on the community.

---

## 2. The Variational Autoencoder

The **variational autoencoder** (VAE), introduced by Kingma and Welling (2014), solves the training problem from Section 1 with three interlocking ideas.

### Idea 1: The Decoder

The decoder $$p_\theta(x \mid z)$$ is the generator we ultimately want.
It is a neural network with parameters $$\theta$$ that takes a latent[^latent] vector $$z \in \mathbb{R}^J$$ and outputs a probability distribution over protein sequences.
At inference time, we sample $$z \sim \mathcal{N}(0, I)$$ and decode it into a protein.

```python
class ProteinDecoder(nn.Module):
    """Decodes a latent vector into amino-acid logits for each position."""

    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor):
        h = torch.relu(self.fc1(z))
        return self.fc2(h)  # raw logits; softmax is applied in the loss
```

### Idea 2: The Encoder as a Training Trick

In supervised learning, training data comes as (input, label) pairs — the pairing is given.  Generative models face a harder problem: there are no pre-assigned latent codes for each training example.  The encoder manufactures these pairings.

To train the decoder, we need noise inputs paired with real proteins.
The **encoder** $$q_\phi(z \mid x)$$ provides them.
Given a training protein $$x$$, the encoder infers a distribution over latent vectors that could plausibly map to $$x$$:

$$z \sim q_\phi(z \mid x) = \mathcal{N}\!\bigl(\mu_\phi(x),\; \sigma^2_\phi(x) I\bigr)$$

Here $$\phi$$ denotes the learnable parameters of the encoder, $$\mu_\phi(x) \in \mathbb{R}^J$$ and $$\sigma^2_\phi(x) \in \mathbb{R}^J$$ are the per-dimension mean and variance, and $$I$$ is the $$J \times J$$ identity matrix.
The distribution $$q_\phi(z \mid x)$$ is called the **approximate posterior**[^posterior] because it approximates the true (intractable[^intractable]) posterior $$p(z \mid x)$$.

[^intractable]: A computation is **intractable** when it is mathematically well-defined but too expensive to carry out exactly --- typically because it requires summing or integrating over an astronomically large space.  **Tractable** is the opposite: the computation has a closed-form solution or an efficient algorithm.

Training works as follows: for each training protein $$x$$, the encoder proposes a distribution over noise inputs $$z$$; we sample a $$z$$ from that distribution and ask the decoder to reconstruct $$x$$ from it.
The reconstruction loss trains both networks jointly—the encoder to propose useful noise inputs, the decoder to recover proteins from them.

```python
import torch
import torch.nn as nn

class ProteinEncoder(nn.Module):
    """Encodes a protein sequence into Gaussian parameters (mu, log-variance)."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)      # mean of q(z|x)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)   # log-variance of q(z|x)

    def forward(self, x: torch.Tensor):
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)  # outputting log(sigma^2) keeps sigma > 0
        return mu, logvar
```

We output $$\log \sigma^2$$ rather than $$\sigma^2$$ directly.
Exponentiating a real number always yields a positive result, so this parameterization guarantees positive variance without needing explicit constraints.

[^posterior]: In Bayesian terminology, the *posterior* is the distribution over latent variables given observed data.  The word "approximate" reminds us that $$q_\phi$$ is a parametric family (here, diagonal Gaussians) that may not perfectly match the true posterior.

### Idea 3: KL Regularization

Without regularization, a face VAE might memorize each training face in a unique corner of latent space — moving between corners produces garbage rather than smooth interpolation.  The KL term prevents this collapse by keeping the latent distribution close to a standard Gaussian.

Reconstruction alone is not enough.
If we only minimize reconstruction error, the encoder can map each protein to a tiny, isolated region of latent space—some arbitrary corner far from the origin.
Reconstruction would be perfect: the decoder memorizes which corner corresponds to which protein.
But at inference we sample $$z \sim \mathcal{N}(0, I)$$, and those arbitrary corners are nowhere near the standard normal.
The decoder has never seen noise drawn from the regions it will encounter at test time.

The **KL divergence** term fixes this mismatch:

$$D_{\mathrm{KL}}\!\bigl(q_\phi(z \mid x) \,\|\, \mathcal{N}(0, I)\bigr)$$

This penalty forces the encoder's output distribution for each training protein to stay close to the standard normal.
The latent vectors the decoder sees during training then overlap with the distribution it will sample from at inference.
The entire latent space becomes populated: sampling $$z \sim \mathcal{N}(0, I)$$ and decoding it produces a valid protein, because the decoder has been trained on noise drawn from exactly this region.

<div class="col-sm-8 mt-3 mb-3 mx-auto">
    <img class="img-fluid rounded" src="{{ '/assets/img/teaching/udl/VAEArch.png' | relative_url }}" alt="VAE architecture">
    <div class="caption mt-1"><strong>Variational autoencoder architecture.</strong> The encoder \(\mathbf{g}[\mathbf{x}, \boldsymbol{\theta}]\) maps input data \(\mathbf{x}\) to the mean \(\boldsymbol{\mu}\) and covariance \(\boldsymbol{\Sigma}\) of a variational distribution \(q(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta})\). A latent code \(\mathbf{z}^*\) is sampled from this distribution and passed to the decoder \(\mathbf{f}[\mathbf{z}^*, \boldsymbol{\phi}]\), which outputs the reconstruction probability \(Pr(\mathbf{x}|\mathbf{z}^*, \boldsymbol{\phi})\). The ELBO loss (top) combines two terms: the reconstruction log-probability \(\log Pr(\mathbf{x}|\mathbf{z}^*, \boldsymbol{\phi})\) (data should have high probability) and the KL divergence \(D_{KL}[q(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}) \| Pr(\mathbf{z})]\) (variational distribution should be close to the prior). <em>Note: this figure uses \(\boldsymbol{\theta}\) for the encoder and \(\boldsymbol{\phi}\) for the decoder; our text uses the opposite convention (\(\phi\) for encoder, \(\theta\) for decoder).</em> Source: Prince, <em>Understanding Deep Learning</em>, CC BY-NC-ND. Used without modification.</div>
</div>

To summarize the two loss terms intuitively:
- **Reconstruction loss**: the decoder should recover $$x$$ from the proposed $$z$$.
- **KL loss**: the proposed $$z$$ should look like standard normal noise.

---

## 3. The ELBO: Formalizing the Training Objective

### Motivation

Section 2 motivated the two loss terms intuitively: reconstruct the data, and keep the encoder's output close to the prior[^prior].
Here we derive these terms from a single principled objective.

Our generative model defines the probability of a protein $$x$$ by integrating over all possible noise inputs:

$$p_\theta(x) = \int p_\theta(x \mid z)\, p(z)\, dz$$

We want to maximize this **marginal likelihood**[^likelihood]—the probability that our decoder, fed with random noise from the prior, produces the training proteins.
This is exactly the right objective for a generative model: if $$p_\theta(x)$$ is high for all training proteins, then sampling $$z \sim \mathcal{N}(0, I)$$ and decoding is likely to produce realistic outputs.

[^prior]: In probability, the **prior** $$p(z)$$ is the distribution we assume over latent variables *before* observing any data.  Here the prior is $$\mathcal{N}(0, I)$$—we assume latent codes are standard-normal random vectors.  The three Bayesian terms form a chain: the **prior** $$p(z)$$ encodes our initial belief, the **likelihood** $$p(x \mid z)$$ says how probable the data is given a particular $$z$$, and the **posterior** $$p(z \mid x)$$ updates the belief after observing data.

[^likelihood]: The **likelihood** $$p_\theta(x)$$ measures the probability the model assigns to observed data.  It is called *marginal* likelihood because we integrate (marginalize) over the latent variable $$z$$: $$p_\theta(x) = \int p_\theta(x \mid z)\, p(z)\, dz$$.  Maximizing it trains the model to consider the training data plausible.

The integral is intractable—it sums the decoder's output over every conceivable $$z$$.
The encoder $$q_\phi(z \mid x)$$ from Section 2 provides the way forward: rather than integrating over all $$z$$, we focus on the values of $$z$$ that the encoder considers plausible for each $$x$$.

### Deriving the Evidence Lower Bound

Variational inference sidesteps the intractable integral by deriving a tractable **lower bound** on $$\log p_\theta(x)$$.
We introduce the encoder distribution $$q_\phi(z \mid x)$$ by multiplying and dividing inside the integral:

$$\log p_\theta(x) = \log \int \frac{p_\theta(x \mid z)\, p(z)}{q_\phi(z \mid x)}\; q_\phi(z \mid x)\, dz$$

Recognizing the right-hand side as an expectation[^expectation] under $$q_\phi(z \mid x)$$, we apply **Jensen's inequality**[^jensen].
Because $$\log$$ is a concave function[^concave], we have $$\log \mathbb{E}[Y] \geq \mathbb{E}[\log Y]$$ for any random variable $$Y > 0$$:

[^expectation]: The **expectation** $$\mathbb{E}[Y]$$ of a random variable $$Y$$ is its average value, weighted by probability.  For a continuous variable with density $$p$$, it is $$\mathbb{E}[Y] = \int y \, p(y)\, dy$$.  Think of it as the "center of mass" of the distribution.

[^concave]: A function $$f$$ is **concave** if its graph curves downward --- formally, $$f(\lambda a + (1-\lambda) b) \geq \lambda f(a) + (1-\lambda) f(b)$$ for any $$\lambda \in [0, 1]$$.  **Convex** is the opposite (curves upward, inequality flipped).  The logarithm is concave because its slope decreases as its input grows.

$$\log p_\theta(x) \geq \int q_\phi(z \mid x) \log \frac{p_\theta(x \mid z)\, p(z)}{q_\phi(z \mid x)}\, dz$$

[^jensen]: Jensen's inequality states that for a concave function $$f$$, we have $$f(\mathbb{E}[X]) \geq \mathbb{E}[f(X)]$$.  Applying it with $$f = \log$$ yields the ELBO.

Splitting the logarithm of the fraction yields two terms:

$$\underbrace{\int q_\phi(z \mid x) \log p_\theta(x \mid z)\, dz}_{\text{reconstruction term}} \;+\; \underbrace{\int q_\phi(z \mid x) \log \frac{p(z)}{q_\phi(z \mid x)}\, dz}_{\text{negative KL divergence}}$$

Recognizing the second integral as the negative **Kullback-Leibler (KL) divergence**[^kl], we arrive at the **Evidence Lower Bound** (ELBO):

$$\text{ELBO}(\phi, \theta; x) = \mathbb{E}_{q_\phi(z \mid x)}\!\bigl[\log p_\theta(x \mid z)\bigr] - D_{\mathrm{KL}}\!\bigl(q_\phi(z \mid x) \,\|\, p(z)\bigr)$$

[^kl]: The KL divergence $$D_{\mathrm{KL}}(q \,\|\, p) = \int q(z) \log \frac{q(z)}{p(z)}\, dz$$ measures how different two distributions are.  It is always non-negative and equals zero only when $$q = p$$.

The relationship to the marginal log-likelihood is:

$$\log p_\theta(x) \geq \text{ELBO}$$

Maximizing the ELBO pushes up on the true log-likelihood from below.

<div class="col-sm-8 mt-3 mb-3 mx-auto">
    <img class="img-fluid rounded" src="{{ '/assets/img/teaching/udl/VAEELBO.png' | relative_url }}" alt="The evidence lower bound (ELBO)">
    <div class="caption mt-1"><strong>The evidence lower bound (ELBO).</strong> The dark curve is the log marginal likelihood \(\log Pr(\mathbf{x}|\boldsymbol{\phi})\); the light curve is the ELBO, which is always below it. (a) Fixing the decoder parameters at \(\boldsymbol{\phi}^{[0]}\) and optimizing the encoder from \(\boldsymbol{\theta}^{[0]}\) to \(\boldsymbol{\theta}^{[1]}\) raises the ELBO (tightens the bound). (b) Then optimizing the decoder from \(\boldsymbol{\phi}^{[0]}\) to \(\boldsymbol{\phi}^{[1]}\) raises both the ELBO and the true log-likelihood. Training alternates between these two steps. Source: Prince, <em>Understanding Deep Learning</em>, CC BY-NC-ND. Used without modification.</div>
</div>

### Interpreting the Two Terms

The two terms of the ELBO correspond exactly to the two intuitions from Section 2.

**Reconstruction term** $$\mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)]$$: sample a latent code $$z$$ from the encoder, pass it through the decoder, and measure how well the original protein $$x$$ is recovered.
Maximizing this term encourages faithful reconstruction.
In practice, this is implemented as the negative cross-entropy between the decoder's output distribution and the true amino-acid sequence.

**KL term** $$D_{\mathrm{KL}}(q_\phi(z \mid x) \,\|\, p(z))$$: this penalizes the encoder for producing distributions that stray too far from the prior $$\mathcal{N}(0, I)$$.
This is the formal version of the inference-time mismatch argument: without this term, the encoder's proposed noise values would not overlap with the standard normal we sample from at generation time.

### Closed-Form KL for Gaussians

When both $$q_\phi(z \mid x)$$ and $$p(z)$$ are Gaussian, the KL divergence has a closed-form expression.
Let $$q_\phi(z \mid x) = \mathcal{N}(\mu, \mathrm{diag}(\sigma^2))$$ with $$\mu \in \mathbb{R}^J$$ and $$\sigma \in \mathbb{R}^J$$, and let $$p(z) = \mathcal{N}(0, I)$$.
Then:

$$D_{\mathrm{KL}} = -\frac{1}{2} \sum_{j=1}^{J} \bigl(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\bigr)$$

```python
def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Closed-form KL divergence: KL(N(mu, sigma^2) || N(0, I)).

    Args:
        mu: encoder mean, shape [batch_size, latent_dim]
        logvar: encoder log-variance, shape [batch_size, latent_dim]

    Returns:
        KL divergence per example, shape [batch_size]
    """
    # logvar = log(sigma^2), so exp(logvar) = sigma^2
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
```

---

## 4. The Reparameterization Trick

### The Problem

Training the ELBO requires computing gradients of the reconstruction term with respect to the encoder parameters $$\phi$$.
But this term involves *sampling* $$z$$ from $$q_\phi(z \mid x)$$, and sampling is not a differentiable operation.
Gradient-based optimizers cannot backpropagate through a random number generator.

### The Solution

Kingma and Welling's reparameterization trick rewrites the sampling step as a deterministic transformation of a fixed noise source.
Instead of drawing $$z \sim \mathcal{N}(\mu, \sigma^2 I)$$ directly, we draw auxiliary noise $$\epsilon \sim \mathcal{N}(0, I)$$, where $$\epsilon \in \mathbb{R}^J$$, and compute:

$$z = \mu + \sigma \odot \epsilon$$

where $$\odot$$ denotes element-wise multiplication.
The randomness now resides entirely in $$\epsilon$$, which does not depend on any learnable parameter.
The mapping from $$\epsilon$$ to $$z$$ is a deterministic, differentiable function of $$\mu$$ and $$\sigma$$, so standard backpropagation applies.

Think of it this way: rather than asking "what is the gradient of a coin flip?", we ask "what is the gradient of a shift-and-scale operation?"
The latter is elementary calculus.

<div class="col-sm-8 mt-3 mb-3 mx-auto">
    <img class="img-fluid rounded" src="{{ '/assets/img/teaching/udl/VAEReparam.png' | relative_url }}" alt="The reparameterization trick">
    <div class="caption mt-1"><strong>The reparameterization trick.</strong> Instead of sampling \(\mathbf{z}^*\) directly from \(q(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta})\) (which blocks gradient flow), we sample noise \(\boldsymbol{\epsilon}^* \sim \text{Norm}_{\epsilon}[\mathbf{0}, \mathbf{I}]\) and compute \(\mathbf{z}^* = \boldsymbol{\mu} + \boldsymbol{\Sigma}^{1/2} \boldsymbol{\epsilon}^*\). The randomness is now in \(\boldsymbol{\epsilon}^*\), while the dependence on the encoder outputs \(\boldsymbol{\mu}\) and \(\boldsymbol{\Sigma}\) is deterministic and differentiable. Source: Prince, <em>Understanding Deep Learning</em>, CC BY-NC-ND. Used without modification.</div>
</div>

```python
def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Sample z = mu + sigma * epsilon, where epsilon ~ N(0, I).

    This makes the sampling step differentiable w.r.t. mu and logvar.
    """
    std = torch.exp(0.5 * logvar)   # sigma = exp(log(sigma^2) / 2)
    eps = torch.randn_like(std)      # epsilon ~ N(0, I)
    return mu + eps * std
```

---

## 5. Diffusion Models: Controlled Destruction

### A Different Philosophy

VAEs learn generation by compressing data into a structured latent space.
Diffusion models take an entirely different approach: they learn generation by learning to *undo corruption*.

Sharpening a blurry photograph is far easier than painting a photorealistic image from a blank canvas.  Diffusion models exploit this asymmetry: instead of generating from scratch, they decompose generation into many small denoising steps.

<div class="col-sm mt-3 mb-3 mx-auto">
    <img class="img-fluid rounded" src="{{ '/assets/img/teaching/diffusion_image_noising.png' | relative_url }}" alt="An image progressively corrupted by Gaussian noise (forward process) and recovered by denoising (reverse process)">
    <div class="caption mt-1"><strong>Diffusion on an image.</strong> The forward process \(q\) progressively adds Gaussian noise until the image becomes indistinguishable from random noise.  The reverse process \(p_\theta\) learns to undo each step, recovering the clean image from pure noise.</div>
</div>

The same idea applies to any continuous data.  Imagine taking the 3D coordinates of a protein backbone and jittering every atom by a tiny random displacement.
The resulting structure is still recognizable, and a neural network could plausibly predict the original positions from this lightly perturbed version.
Repeat the corruption many times until the coordinates become indistinguishable from random points in space.
If the network can reverse *each individual step*, chaining all the reverse steps together recovers a clean structure from pure noise.

<div class="col-sm mt-3 mb-3 mx-auto">
    <img class="img-fluid rounded" src="{{ '/assets/img/teaching/diffusion_pointcloud_noising.png' | relative_url }}" alt="A 2D point cloud progressively corrupted by Gaussian noise and recovered by denoising">
    <div class="caption mt-1"><strong>Diffusion on a point cloud.</strong> Structured coordinates (left) are progressively corrupted until they resemble a random Gaussian scatter (right).  The reverse process learns to recover the original structure step by step.  For proteins, the points represent residue positions in 3D space.</div>
</div>

### The Forward Process: Adding Noise

Let $$x_0 \in \mathbb{R}^D$$ denote a clean data point—say, the 3D coordinates of a protein backbone or a continuous embedding of a sequence.
The **forward process** produces a sequence of increasingly noisy versions $$x_1, x_2, \ldots, x_T$$ by adding Gaussian noise at each step:

$$q(x_t \mid x_{t-1}) = \mathcal{N}\!\bigl(x_t;\; \sqrt{1 - \beta_t}\, x_{t-1},\; \beta_t I\bigr)$$

Here $$\beta_t \in (0, 1)$$ is a scalar controlling how much noise is added at step $$t$$, and $$T$$ is the total number of steps (typically 1000).
The collection $$\{\beta_1, \beta_2, \ldots, \beta_T\}$$ is called the **noise schedule**.
It usually starts small (gentle corruption early on) and increases over time (aggressive corruption later).

<div class="col-sm-8 mt-3 mb-3 mx-auto">
    <img class="img-fluid rounded" src="{{ '/assets/img/teaching/udl/DiffusionForward2.png' | relative_url }}" alt="Forward diffusion process">
    <div class="caption mt-1"><strong>Forward diffusion process.</strong> (a) Three example trajectories: clean data \(x\) (top) is progressively corrupted through noisy versions \(z_{20}, z_{40}, \ldots, z_{100}\), converging to pure noise. (b) The conditional distributions \(q(z_1|x)\), \(q(z_{41}|z_{40})\), \(q(z_{81}|z_{80})\) at selected steps. Each step adds a small amount of Gaussian noise, so the conditional is a narrow Gaussian centered near the previous value. As diffusion progresses, the distributions widen and overlap, erasing information about the starting point. Source: Prince, <em>Understanding Deep Learning</em>, CC BY-NC-ND. Used without modification.</div>
</div>

<div class="col-sm-9 mt-3 mb-3 mx-auto">
    <img class="img-fluid rounded" src="{{ '/assets/img/teaching/diffusion_noise_schedule.png' | relative_url }}" alt="Diffusion noise schedule">
    <div class="caption mt-1">A linear noise schedule over 1000 timesteps. β_t (noise variance per step) increases linearly. √ᾱ_t (signal coefficient) decays from 1 to near 0 — the clean signal is gradually destroyed. √(1−ᾱ_t) (noise coefficient) grows from 0 to 1 — by t=T, the data is pure noise.</div>
</div>

A key mathematical property eliminates the need to apply noise sequentially.
Define $$\alpha_t = 1 - \beta_t$$ and $$\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$$ (the cumulative product).
Then we can jump directly from $$x_0$$ to any $$x_t$$:

$$q(x_t \mid x_0) = \mathcal{N}\!\bigl(x_t;\; \sqrt{\bar{\alpha}_t}\, x_0,\; (1 - \bar{\alpha}_t) I\bigr)$$

Equivalently:

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

This closed-form expression is essential for efficient training—we can sample any noisy version of the data in a single operation without iterating through all previous timesteps.

```python
class DiffusionSchedule:
    """Precomputes noise schedule quantities for efficient training."""

    def __init__(self, T: int = 1000, beta_start: float = 1e-4,
                 beta_end: float = 0.02):
        self.T = T

        # Linear schedule from beta_start to beta_end
        self.betas = torch.linspace(beta_start, beta_end, T)

        # Precompute alpha, cumulative alpha, and their square roots
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor,
                  noise: torch.Tensor = None) -> torch.Tensor:
        """Sample x_t from q(x_t | x_0) using the closed-form expression.

        Args:
            x0: clean data, shape [batch, ...]
            t: timestep indices, shape [batch]
            noise: optional pre-sampled noise (same shape as x0)
        """
        if noise is None:
            noise = torch.randn_like(x0)

        # Reshape for broadcasting: [batch, 1, 1, ...]
        sqrt_ab = self.sqrt_alpha_bars[t].view(-1, *([1] * (x0.dim() - 1)))
        sqrt_1m_ab = self.sqrt_one_minus_alpha_bars[t].view(-1, *([1] * (x0.dim() - 1)))

        return sqrt_ab * x0 + sqrt_1m_ab * noise
```

---

## 6. The Reverse Process and Training Objective

### Learning to Denoise

The forward process is fixed—no learnable parameters.
All the learning happens in the **reverse process**, which starts from pure noise $$x_T \sim \mathcal{N}(0, I)$$ and iteratively recovers the data.
Each reverse step is modeled as:

$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\!\bigl(x_{t-1};\; \mu_\theta(x_t, t),\; \sigma_t^2 I\bigr)$$

where $$\mu_\theta$$ is a neural network with parameters $$\theta$$ that predicts the denoised mean, and $$\sigma_t^2$$ is typically set to $$\beta_t$$ (or a related fixed quantity).

<div class="col-sm-8 mt-3 mb-3 mx-auto">
    <img class="img-fluid rounded" src="{{ '/assets/img/teaching/udl/DiffusionReverse.png' | relative_url }}" alt="Reverse denoising process">
    <div class="caption mt-1"><strong>Reverse denoising process.</strong> (a) The marginal distribution \(q(z_t)\) (heatmap) spreads out as \(t\) increases. Sampled points \(z_3^*, z_{10}^*, z_{20}^*\) are shown at selected timesteps. (b) At each step, the forward conditional \(q(z_{t+1}|z_t)\) (brown) and reverse conditional \(q(z_t|z_{t+1}^*)\) (teal) are both narrow Gaussians, while the marginal \(q(z_t)\) (gray) is broad. The reverse conditional is tractable because it depends on a single known value \(z_{t+1}^*\), making each denoising step a small, learnable correction. Source: Prince, <em>Understanding Deep Learning</em>, CC BY-NC-ND. Used without modification.</div>
</div>

### Noise Prediction Parameterization

The reverse process requires choosing what quantity the neural network should predict.
Three equivalent parameterizations exist: the network can predict the clean data $$x_0$$, the posterior mean $$\mu_\theta$$, or the noise $$\epsilon$$.
Ho et al. (2020) found that predicting the noise leads to the simplest and most stable training.

Rather than predicting the mean $$\mu_\theta$$ directly, we train the network to predict the *noise* $$\epsilon$$ that was added to obtain $$x_t$$ from $$x_0$$.
Once the network predicts $$\epsilon_\theta(x_t, t) \in \mathbb{R}^D$$, we can recover the mean via:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\, \epsilon_\theta(x_t, t) \right)$$

### The Simple Training Objective

The training loss is the mean squared error between the true noise and the predicted noise, averaged over random timesteps and random noise draws:

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t,\, x_0,\, \epsilon}\!\left[\lVert \epsilon - \epsilon_\theta(x_t, t) \rVert^2\right]$$

Despite their different intuitions, diffusion models and VAEs are mathematically closer than they appear.
A diffusion model can be viewed as a **hierarchical VAE** with $$T$$ latent layers $$x_1, x_2, \ldots, x_T$$, where the "encoder" (forward process) is fixed rather than learned, and each layer has the same dimensionality as the data.
The training objective is in fact an ELBO, decomposed into $$T$$ per-timestep KL divergence terms instead of the single KL term in a standard VAE.
Because both the forward posterior $$q(x_{t-1} \mid x_t, x_0)$$ and the reverse model $$p_\theta(x_{t-1} \mid x_t)$$ are Gaussian, each KL term reduces to an MSE between their means.
Ho et al. <sup id="cite-a"><a href="#ref-a">[a]</a></sup> showed that dropping the per-timestep weighting coefficients yields $$\mathcal{L}_{\text{simple}}$$, which works better in practice.
For the full derivation, see Luo (2022) [^elbo-derivation].

<div class="col-sm-8 mt-3 mb-3 mx-auto">
    <img class="img-fluid rounded" src="{{ '/assets/img/teaching/blog/luo_hvae.png' | relative_url }}" alt="Hierarchical VAE graphical model with T latent layers forming a Markov chain">
    <div class="caption mt-1"><strong>Diffusion as a hierarchical VAE.</strong> A Markovian Hierarchical VAE with \(T\) latent layers. The generative (reverse) process \(p_\theta\) flows top-down along the chain; the inference (forward) process \(q\) flows bottom-up. A diffusion model is this structure with the forward process fixed to Gaussian noise addition and all latent layers sharing the data dimensionality. Source: Luo, <em>Understanding Diffusion Models: A Unified Perspective</em> (2022).</div>
</div>

<div class="col-sm mt-3 mb-3 mx-auto">
    <img class="img-fluid rounded" src="{{ '/assets/img/teaching/blog/luo_elbo_denoising.png' | relative_url }}" alt="ELBO decomposition: each timestep matches the learned denoising step to a tractable ground-truth posterior">
    <div class="caption mt-1"><strong>Per-timestep ELBO decomposition.</strong> At each step, the learned reverse distribution \(p_\theta(x_{t-1}|x_t)\) (green) is trained to match the tractable ground-truth denoising posterior \(q(x_{t-1}|x_t, x_0)\) (pink). Because both are Gaussians with matched variance, the KL divergence between them reduces to an MSE between means --- which further simplifies to the noise-prediction loss. Source: Luo, <em>Understanding Diffusion Models: A Unified Perspective</em> (2022).</div>
</div>

[^elbo-derivation]: Luo, C. (2022). "Understanding Diffusion Models: A Unified Perspective." *arXiv preprint arXiv:2208.11970*. Sections 3–4 derive the diffusion ELBO from the hierarchical VAE perspective.

The training procedure for a single batch is:

1. Draw a batch of clean data $$x_0$$.
2. Sample random timesteps $$t \sim \mathrm{Uniform}\{1, \ldots, T\}$$.
3. Sample random noise $$\epsilon \sim \mathcal{N}(0, I)$$.
4. Compute noisy data $$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon$$.
5. Predict $$\hat{\epsilon} = \epsilon_\theta(x_t, t)$$.
6. Minimize $$\lVert \epsilon - \hat{\epsilon} \rVert^2$$.

```python
def diffusion_training_loss(model: nn.Module, x0: torch.Tensor,
                            schedule: DiffusionSchedule) -> torch.Tensor:
    """Compute the simplified diffusion training loss (noise prediction MSE).

    Args:
        model: noise prediction network, takes (x_t, t) -> predicted noise
        x0: clean data batch, shape [batch, ...]
        schedule: DiffusionSchedule with precomputed quantities
    """
    batch_size = x0.size(0)

    # Step 1: sample random timesteps for each example
    t = torch.randint(0, schedule.T, (batch_size,), device=x0.device)

    # Step 2: sample Gaussian noise
    noise = torch.randn_like(x0)

    # Step 3: compute noisy version x_t
    x_t = schedule.add_noise(x0, t, noise)

    # Step 4: predict the noise
    noise_pred = model(x_t, t)

    # Step 5: MSE between true and predicted noise
    return nn.functional.mse_loss(noise_pred, noise)
```

This objective has a deep connection to **denoising score matching**[^score].
The score function $$\nabla_{x} \log p(x)$$ points in the direction of increasing data density.
Predicting the noise $$\epsilon$$ is mathematically equivalent to estimating the score at timestep $$t$$, up to a scaling factor.
This connection links diffusion models to the broader framework of score-based generative modeling (Song and Ermon, 2019).

[^score]: The *score* of a distribution $$p(x)$$ is the gradient of its log-density, $$\nabla_x \log p(x)$$.  Score matching trains a network to approximate this gradient without knowing the normalizing constant of $$p$$.

<div class="col-sm-8 mt-3 mb-3 mx-auto">
    <img class="img-fluid rounded" src="{{ '/assets/img/teaching/blog/yangsong_score_contour.jpg' | relative_url }}" alt="Score function visualized as a vector field over a mixture of two Gaussians">
    <div class="caption mt-1"><strong>The score function as a vector field.</strong> Contour plot of a mixture of two Gaussians, overlaid with the score \(\nabla_x \log p(x)\) at each point. Arrows point toward the modes (high-density regions). The denoising network implicitly estimates this vector field at each noise level. Source: Song, <em>Generative Modeling by Estimating Gradients of the Data Distribution</em> (2021).</div>
</div>

---

## 7. The Denoising Loop and Network Architecture

### Generation by Iterative Denoising

<div class="col-sm mt-3 mb-3 mx-auto">
    <img class="img-fluid rounded" src="{{ '/assets/img/teaching/mermaid/s26-05-generative-models_diagram_1.png' | relative_url }}" alt="Diffusion reverse process: iterative denoising from pure noise to clean protein structure over T timesteps">
</div>

Once the noise prediction network is trained, generation proceeds by simulating the reverse process.
Starting from pure noise $$x_T \sim \mathcal{N}(0, I)$$, we apply the learned denoising step $$T$$ times:

```python
@torch.no_grad()
def ddpm_sample(model: nn.Module, schedule: DiffusionSchedule,
                shape: tuple, device: str = "cpu") -> torch.Tensor:
    """Generate samples via DDPM reverse process.

    Args:
        model: trained noise prediction network
        schedule: DiffusionSchedule with precomputed quantities
        shape: desired output shape, e.g. (n_samples, n_residues, 3)
        device: torch device
    """
    # Start from pure Gaussian noise
    x = torch.randn(shape, device=device)

    for t in reversed(range(schedule.T)):
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)

        # Predict the noise component
        noise_pred = model(x, t_batch)

        # Retrieve schedule quantities for this timestep
        alpha = schedule.alphas[t]
        alpha_bar = schedule.alpha_bars[t]
        beta = schedule.betas[t]

        # Compute the predicted mean of x_{t-1}
        mean = (1.0 / torch.sqrt(alpha)) * (
            x - (beta / torch.sqrt(1.0 - alpha_bar)) * noise_pred
        )

        # Add stochastic noise for all steps except the final one
        if t > 0:
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(beta) * noise
        else:
            x = mean

    return x
```

At each step except the last ($$t = 0$$), we inject a small amount of fresh noise.
This stochasticity ensures diversity: different initial noise samples produce different outputs, and even the same initial noise can yield slightly different results due to the injected randomness during denoising.

### Timestep Conditioning

The noise prediction network must know *which* timestep it is operating at.
A heavily corrupted input ($$t$$ near $$T$$) requires aggressive denoising, while a lightly corrupted input ($$t$$ near 0) needs only gentle refinement.
The standard approach borrows **sinusoidal position embeddings** from the transformer literature to encode the scalar timestep $$t$$ as a high-dimensional vector:

```python
import math

class SinusoidalTimestepEmbedding(nn.Module):
    """Encodes scalar timestep t into a d-dimensional vector.

    Uses the same sinusoidal scheme as transformer positional encodings.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=device) / half
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
```

This embedding is then injected into the denoising network—for example, by adding it to intermediate feature maps or concatenating it with the input.

### Architecture Choices for Proteins

The choice of denoising network architecture depends on the data representation.

U-Nets dominate image generation (Stable Diffusion) and medical image segmentation thanks to their multi-scale skip connections.  Transformers dominate text generation thanks to their ability to capture long-range dependencies.  For proteins, the choice depends on the data representation:

For **spatial data** (images, 3D point clouds, protein backbone coordinates), U-Net architectures with skip connections are common.
The encoder half progressively reduces spatial resolution, capturing long-range context, while the decoder half restores resolution.
Skip connections preserve fine-grained spatial details that would otherwise be lost during downsampling.

For **sequential data** (text, protein sequences), transformer architectures work well because attention allows every position to interact with every other position, capturing long-range dependencies (as discussed in Lecture 1).

---

## 8. VAEs vs. Diffusion: Choosing the Right Tool

Both VAEs and diffusion models are principled approaches to learning the distribution over proteins.
They differ in architecture, training, generation speed, and sample quality.
The table below summarizes the key trade-offs.

| Aspect | VAE | Diffusion |
|--------|-----|-----------|
| **Latent space** | Low-dimensional, explicitly structured | Same dimensionality as data; no explicit latent space |
| **Training** | Single forward pass per example | Multiple noise levels sampled per example |
| **Sampling speed** | Fast—one decoder pass | Slow—hundreds to thousands of denoising steps |
| **Sample quality** | Good; can be blurry or mode-averaged | State-of-the-art; captures fine-grained detail |
| **Diversity** | High | Very high |
| **Controllability** | Latent-space manipulation, conditional decoding | Classifier guidance, classifier-free guidance |
| **Interpretability** | Latent dimensions can align with meaningful properties | Less interpretable |

The practical upshot: VAEs win on speed (milliseconds per sample) and latent-space interpretability, making them the default for large-scale screening; diffusion wins on sample quality, making it the default for targeted design of physically plausible backbones.
Latent diffusion models (Rombach et al., 2022) bridge the gap by running diffusion in a VAE's compressed latent space, trading some quality for much faster generation.

---

## 9. Conditional Generation

Both VAEs and diffusion models can be extended to **conditional generation**—steering the model toward data with specific desired properties.  Text-to-image models like DALL-E and Stable Diffusion generate images conditioned on text prompts; class-conditional ImageNet models generate images of specific object categories.  Two general strategies have emerged for diffusion models.

**Classifier guidance.** Dhariwal and Nichol (2021) train a separate classifier that operates on noisy inputs.  At each denoising step, the gradient of the classifier's log-probability with respect to the noisy input is added to the predicted denoising direction, pushing the sample toward regions associated with the desired class.

**Classifier-free guidance** (Ho and Salimans, 2022) eliminates the need for a separate classifier.
During training, the conditioning signal is randomly dropped with some probability, so the model learns both conditional and unconditional generation.
At inference time, the two predictions are blended:

$$\hat{\epsilon} = \epsilon_\theta(x_t, t, \varnothing) + s \cdot \bigl(\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \varnothing)\bigr)$$

where $$c$$ is the condition, $$\varnothing$$ denotes the null condition, and $$s > 1$$ amplifies the effect of conditioning.
This approach is simpler and often more effective than classifier guidance.  Stable Diffusion uses the guidance scale $$s$$ to trade prompt adherence against visual diversity.

---

## Key Takeaways

This lecture covered two foundational frameworks for generative modeling.

**Variational autoencoders** learn a probabilistic, low-dimensional latent space.
The ELBO training objective balances reconstruction fidelity against latent-space regularity (via the KL divergence).
The reparameterization trick makes end-to-end training through stochastic sampling possible.
Generation is fast—a single decoder pass—and the structured latent space enables interpolation, clustering, and property-guided optimization.

**Diffusion models** learn to reverse a gradual noise-corruption process.
The forward process has a closed-form expression for any timestep, enabling efficient training.
The reverse process is learned by predicting the noise added at each step.
Generation requires many sequential denoising steps but produces state-of-the-art sample quality.

**Conditional generation** steers either framework toward desired properties via classifier guidance or classifier-free guidance.

The choice between VAEs and diffusion depends on the application: VAEs for speed and interpretability, diffusion for sample quality, latent diffusion for a middle ground.

---

## Further Reading

- Lilian Weng, ["From Autoencoder to Beta-VAE"](https://lilianweng.github.io/posts/2018-08-12-vae/) — an accessible walkthrough of variational autoencoders, from vanilla AE to disentangled representations.
- Lilian Weng, ["What are Diffusion Models?"](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) — a thorough introduction to diffusion models covering the forward process, reverse denoising, and connections to score matching.
- Yang Song, ["Generative Modeling by Estimating Gradients of the Data Distribution"](https://yang-song.net/blog/2021/score/) — score functions, noise-perturbed distributions, Langevin sampling, and SDEs for diffusion, by the score-matching pioneer.
- Calvin Luo, ["Understanding Diffusion Models: A Unified Perspective"](https://calvinyluo.com/2022/08/26/diffusion-tutorial.html) — derives diffusion from hierarchical VAEs and connects the ELBO to the score-based view via Tweedie's formula.

## References

<p id="ref-a"><a href="#cite-a">[a]</a> Ho, J., Jain, A., and Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models." <em>Advances in Neural Information Processing Systems (NeurIPS)</em>.</p>

---
