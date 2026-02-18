---
layout: post
title: "Code Walkthrough: nano-alphafold2"
date: 2026-03-30
description: "Build AlphaFold2 from scratch in ~650 lines of PyTorch — Pairformer, SE(3) diffusion, and FAPE loss."
course: "2026-spring-protein-ai"
course_title: "Protein & Artificial Intelligence"
course_semester: "Spring 2026"
lecture_number: 8
preliminary: false
toc:
  sidebar: left
related_posts: false
---

<div style="background: var(--global-code-bg-color); border-left: 4px solid var(--global-theme-color); padding: 1em 1.2em; margin-bottom: 2em; border-radius: 4px;">
<strong>Code:</strong> <a href="https://github.com/sungsoo-ahn/nano-protein-ais/tree/master/alphafold2">sungsoo-ahn/nano-protein-ais/alphafold2</a><br>
<strong>Files:</strong> <code>model.py</code> (~650 lines) + <code>train.py</code><br>
<strong>Parameters:</strong> 8.8M &nbsp;|&nbsp; <strong>Training:</strong> 20 min on 1x RTX 3090
</div>

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Companion to <a href="{{ '/lectures/08-alphafold/' | relative_url }}">Lecture: AlphaFold</a>. The lecture covers the theory; this page builds it from scratch.</em>
</p>

You give it a string of amino acids. It gives you back a 3D protein structure. That is the problem AlphaFold2 solved and what we are going to build here, from scratch, in about 650 lines of PyTorch.

The real AlphaFold2 trains on ~200,000 protein structures with 128 GPUs. Ours trains on 9 proteins in 20 minutes on an RTX 3090 and gets 2-3 Angstrom RMSD on training structures. That is not useful for biology, but every architectural piece is here -- the Pairformer, SE(3) frame diffusion, IGSO3, FAPE loss -- and you can read through it top to bottom in an afternoon.

## The Problem

A protein is a chain of amino acids (there are 20 types). The chain folds into a specific 3D shape that determines what the protein does. Predicting that shape from the sequence alone is the "protein folding problem," and it was one of the biggest open problems in biology for 50 years.

Our input is a sequence like `TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN` (that is crambin, 46 residues). Our output is three 3D coordinates per residue -- the backbone atoms N, CA, C -- giving us the protein's shape.

## The Data

We use 9 small proteins downloaded straight from the RCSB PDB. Crambin (1CRN, 46 residues), ubiquitin (1UBQ, 76 residues), protein G (2GB1, 56 residues), and a handful of others. The `train.py` script downloads them automatically. Each PDB file gets parsed into backbone coordinates (N, CA, C) and the amino acid sequence. That is all you need.

## The Architecture: Two Stages

The model has two stages. First, the **Pairformer** reads the sequence and builds rich representations of individual residues and their pairwise relationships. Second, the **diffusion module** uses those representations to iteratively denoise random frames into the correct 3D structure.

### Stage 1: Pairformer

This is a simplified Evoformer from AlphaFold2 with the MSA axis dropped entirely. We work with two representations:

- **Single representation** `[L, 256]` -- one vector per residue
- **Pair representation** `[L, L, 64]` -- one vector per pair of residues

The input embedding turns the amino acid sequence into one-hot vectors, projects them into single and pair space, and adds relative position encodings. Then we run 4 Pairformer blocks. Each block updates the pair representation with **triangular multiplicative updates** (outgoing and incoming), **triangular attention** (starting and ending), and a transition FFN. Then it updates the single representation with attention that uses the pair representation as bias.

The triangular updates are the key insight. If residues i and j are close in 3D, and j and k are close, then i and k are geometrically constrained. The triangular multiplicative update captures exactly this:

```python
# Outgoing: aggregate over shared index k
out = torch.einsum("bikc,bjkc->bijc", left, right)
# Incoming: aggregate over shared index k
out = torch.einsum("bkic,bkjc->bijc", left, right)
```

This is the mechanism that lets the pair representation reason about spatial consistency. After 4 blocks, the pair representation encodes a rich map of which residues should be near each other.

### Stage 2: SE(3) Diffusion

Now we need to go from "which residues are close" to actual 3D coordinates. We represent each residue as a **rigid frame** in SE(3) -- a rotation matrix (3x3) plus a translation vector (3,). The rotation orients the residue in space; the translation places it. From these frames, we reconstruct backbone atoms using ideal bond geometry.

The prediction uses **denoising diffusion**. During training, we take the true frames, corrupt them with noise, and train the model to predict the clean frames from the noisy ones. At inference, we start from pure noise and iteratively denoise over 100 steps.

The noise model is a **product diffusion** on $$SE(3) = SO(3) \times \mathbb{R}^3$$. Translations get standard Gaussian noise (cosine DDPM schedule). Rotations get IGSO3 noise.

## IGSO3: Noise on Rotations

You cannot just add Gaussian noise to a rotation matrix -- you would get something that is not a valid rotation. Instead, IGSO3 (Isotropic Gaussian on SO(3)) samples a rotation perturbation by:

1. Pick a random axis (unit vector in $$\mathbb{R}^3$$)
2. Pick a random angle from a Gaussian with standard deviation $$\sigma$$
3. Apply the Rodrigues formula to get a valid rotation matrix

```python
def sample_igso3(shape, sigma, device):
    omega = torch.abs(torch.randn(*shape, device=device) * sigma)
    axis = torch.randn(*shape, 3, device=device)
    axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-8)
    return axis_angle_to_rotation_matrix(axis * omega.unsqueeze(-1))
```

The Rodrigues formula (`axis_angle_to_rotation_matrix`) turns an axis-angle vector into a rotation matrix: $$R = I + \sin(\theta) K + (1 - \cos(\theta)) K^2$$, where $$K$$ is the skew-symmetric matrix of the axis. Small $$\sigma$$ means small perturbations; large $$\sigma$$ means nearly random rotations. The schedule goes from $$\sigma=0$$ (clean) to $$\sigma=1.5$$ (heavily corrupted).

## The Denoiser

The diffusion module takes noisy frames, a timestep, and the Pairformer outputs, and predicts clean frames. It flattens each frame into a 12-dimensional vector (9 from the rotation matrix + 3 from the translation), projects it to dimension 256, adds the single representation, conditions on time via adaptive layer norm, and runs 4 transformer blocks with pair bias. The output is a 6-dimensional correction per residue (3 axis-angle + 3 translation) that gets composed onto the noisy frame.

## FAPE Loss

Frame Aligned Point Error measures structural quality in each residue's local coordinate frame. For every frame f, transform all CA positions into f's local coordinates, then measure the distance between predicted and true local positions:

```python
pred_delta = pred_ca.unsqueeze(1) - pred_frames.trans.unsqueeze(2)
pred_local = torch.einsum("bfij,bfaj->bfai",
    pred_frames.rots.transpose(-1, -2), pred_delta)
error = ((pred_local - true_local) ** 2).sum(-1).add(1e-8).sqrt()
```

FAPE is invariant to global rigid motion -- you could rotate the entire predicted structure and the loss would not change. This is essential because there is no "correct" global orientation.

The total loss combines four terms: translation MSE (weight 1.0), rotation geodesic distance (weight 0.5), FAPE (weight 0.1), and pLDDT cross-entropy (weight 0.01). The direct frame losses provide stable gradients early in training; FAPE pushes for structural quality.

## Training

8.8M parameters. 9 proteins. Batch size 1. AdamW at lr=3e-3 with cosine decay. Gradient clipping at 1.0. On an RTX 3090 it takes about 20 minutes for 10,000 epochs. Here is the loss progression:

```
step=50     loss=22.4136  fape=6.345  trans=21.876  rot=0.487
step=1000   loss=8.2341   fape=5.891  trans=7.413   rot=0.512
step=5000   loss=3.1052   fape=4.723  trans=2.105   rot=0.468
step=10000  loss=1.8921   fape=4.312  trans=1.204   rot=0.443
step=50000  loss=1.0134   fape=3.987  trans=0.253   rot=0.411
```

The translation loss drops fast -- from ~22 down to ~0.25. That is the model learning where to place each residue. The FAPE loss is harder, going from ~6 to ~4. The rotation loss stays relatively flat around 0.4-0.5. Getting the orientations exactly right is the hard part.

## Results

After training, we run reverse diffusion (100 denoising steps) to predict structures:

| Protein | Residues | CA-RMSD (A) |
|---------|----------|-------------|
| 1CRN (crambin) | 46 | 3.20 |
| 1UBQ (ubiquitin) | 76 | 2.03 |
| 2GB1 (protein G) | 56 | 3.32 |

Average pLDDT is around 0.51. The real AlphaFold2 gets sub-1A RMSD and pLDDT above 0.9 on most proteins. But we are training on 9 proteins instead of 200,000, with 8.8M parameters instead of 93M, for 20 minutes instead of weeks.

The 2.03A on ubiquitin is genuinely decent -- that is close to experimental resolution for many X-ray structures. The model has clearly learned the relationship between sequence and structure for these training proteins.

## What Is Missing

Compared to the real AlphaFold2: no MSA (we dropped it entirely from Pairformer), no templates, no recycling, no structure module with IPA, no side-chain prediction, no ensembling. We use frame diffusion where the original uses an iterative structure module. The data is 9 proteins instead of the full PDB + UniRef + BFD.

But the core ideas are all here. The triangular updates that enforce geometric consistency. The SE(3) diffusion that respects the symmetry of 3D space. The FAPE loss that is invariant to global orientation. And the whole thing fits in two files you can read in an afternoon.
