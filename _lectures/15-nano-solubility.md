---
layout: post
title: "Code Walkthrough: nano-solubility"
date: 2026-03-03
description: "Build a protein solubility classifier from scratch — learned embeddings, 1D convolutions, and evaluation on real E. coli expression data."
course: "2026-spring-protein-ai"
course_title: "Protein & Artificial Intelligence"
course_semester: "Spring 2026"
lecture_number: 5
preliminary: true
toc:
  sidebar: left
related_posts: false
---

<div style="background: var(--global-code-bg-color); border-left: 4px solid var(--global-theme-color); padding: 1em 1.2em; margin-bottom: 2em; border-radius: 4px;">
<strong>Code:</strong> <a href="https://github.com/sungsoo-ahn/nano-protein-ais/tree/master/solubility">sungsoo-ahn/nano-protein-ais/solubility</a><br>
<strong>Files:</strong> <code>model.py</code> (~80 lines) + <code>train.py</code> (~100 lines)<br>
<strong>Parameters:</strong> 59.5K &nbsp;|&nbsp; <strong>Data:</strong> 18,453 proteins from <a href="https://dmol.pub/dl/layers.html">dmol.pub</a>
</div>

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Companion to <a href="{{ '/lectures/04-preliminary-improving-solubility/' | relative_url }}">Preliminary Note 4: Predicting Protein Solubility</a>. The note covers the full ML pipeline — data splits, class imbalance, early stopping. This page focuses on the model: upgrading from the MLP to a convolutional architecture.</em>
</p>

Note 4 builds a solubility predictor from hand-crafted sequence features --- amino acid frequencies, hydrophobicity, net charge. This works, but the model can only learn from the features we give it. If a solubility-relevant pattern (a specific sequence motif, a hydrophobic stretch at a particular position) is not captured by those 24 numbers, the MLP has no way to discover it.

Here we upgrade the architecture in two ways, following the approach in White et al.'s [*Deep Learning for Molecules and Materials*](https://dmol.pub/dl/layers.html):

1. **Learned embeddings** replace hand-crafted features --- the model discovers its own amino acid representations.
2. **1D convolutions** scan for local sequence patterns --- the model detects motifs directly from the raw sequence.

Same dataset, same task, better inductive bias.

## The dataset

The dmol.pub solubility dataset contains 18,453 proteins from *E. coli* expression experiments, each labeled soluble or insoluble. Sequences are pre-tokenized as integer arrays: 1--20 for the 20 amino acids, 0 for padding, all padded to length 200.

```python
import numpy as np
import urllib.request

urllib.request.urlretrieve(
    "https://github.com/whitead/dmol-book/raw/main/data/solubility.npz",
    "solubility.npz",
)
with np.load("solubility.npz") as data:
    pos_data, neg_data = data["positives"], data["negatives"]

print(f"Soluble: {pos_data.shape[0]}, Insoluble: {neg_data.shape[0]}")
print(f"Max sequence length: {pos_data.shape[1]}")
# Soluble: 9,667  Insoluble: 8,786
# Max sequence length: 200
```

Roughly balanced (52% soluble, 48% insoluble), so we don't need aggressive class weighting --- though Note 4 covers that technique for imbalanced settings.

```python
import torch
from torch.utils.data import TensorDataset, DataLoader

features = np.concatenate([pos_data, neg_data], axis=0)
labels = np.concatenate([
    np.ones(len(pos_data)),
    np.zeros(len(neg_data)),
])

# Shuffle
idx = np.random.permutation(len(labels))
features, labels = features[idx], labels[idx]

features = torch.from_numpy(features).long()
labels = torch.from_numpy(labels).float().unsqueeze(1)

N = len(labels)
split = int(0.1 * N)

test_data = DataLoader(
    TensorDataset(features[:split], labels[:split]),
    batch_size=16, shuffle=False,
)
val_data = DataLoader(
    TensorDataset(features[split:2*split], labels[split:2*split]),
    batch_size=16, shuffle=False,
)
train_data = DataLoader(
    TensorDataset(features[2*split:], labels[2*split:]),
    batch_size=16, shuffle=True,
)
```

The input tensor is `[B, 200]` --- integer token indices, not one-hot vectors. This is where embeddings come in.

## Embeddings: learning amino acid similarity

Note 4's MLP takes hand-crafted features: amino acid frequencies, mean hydrophobicity, net charge --- 24 numbers that summarize the whole protein. This is compact and interpretable, but it throws away all positional information and can only capture patterns the feature engineer anticipated.

An embedding layer learns a dense vector for each amino acid:

```python
import torch.nn as nn

embedding = nn.Embedding(num_embeddings=21, embedding_dim=2, padding_idx=0)
```

The 21 entries cover 20 amino acids plus the padding token (index 0, always mapped to a zero vector). Each amino acid gets a learned 2-dimensional vector. After training, amino acids with similar physicochemical properties --- leucine and isoleucine (both hydrophobic), aspartate and glutamate (both negatively charged) --- end up with similar vectors. The model discovers this on its own, purely from the solubility signal.

Why `embedding_dim=2`? It is the smallest dimension that can capture meaningful structure. Even 2 dimensions are enough for a rough hydrophobicity--charge plane. Larger values (16, 64) would give more capacity at the cost of more parameters.

The shape transformation: `[B, 200]` integer tokens $$\to$$ `[B, 200, 2]` dense vectors. Unlike Note 4's 24-dimensional global summary, this preserves the full sequence --- every position gets its own representation.

## Conv1d: scanning for sequence patterns

An MLP on flattened input treats position 1 and position 100 as unrelated dimensions. A 1D convolution slides a small window along the sequence, detecting local patterns:

```python
conv = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=5)
```

This creates 16 filters, each of size $$5 \times 2$$. Each filter slides along the sequence and outputs a scalar at every position --- a score for how well the local 5-residue window matches that filter's learned pattern. The output shape is `[B, 16, L-4]` (the length shrinks by `kernel_size - 1` without padding).

What might these 16 filters learn?

- A hydrophobic run detector: high activation when it sees LVAIL-like stretches (buried residues, likely insoluble if exposed)
- A charged cluster detector: high activation on DEKE-like stretches (surface residues, often soluble)
- A proline-glycine pattern: helix breakers that affect folding and aggregation

**Max pooling** then compresses each channel, keeping only the strongest activation:

```python
pool = nn.MaxPool1d(kernel_size=4)
```

"I don't care *where* in the sequence the hydrophobic run is --- just whether it exists and how strong the signal is." This is translation invariance, tailored for sequences.

## The full model

Three convolutional blocks with progressively smaller kernels extract local features, then three dense layers classify:

```python
import torch
import torch.nn as nn

class SolubilityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(21, 2, padding_idx=0)

        self.conv1 = nn.Conv1d(2, 16, kernel_size=5)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(16, 16, kernel_size=3)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(16, 16, kernel_size=3)
        self.pool3 = nn.MaxPool1d(2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(160, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)       # [B, 200] → [B, 200, 2]
        x = x.transpose(1, 2)       # [B, 2, 200] — Conv1d expects channels first

        x = self.pool1(self.relu(self.conv1(x)))   # [B, 16, 49]
        x = self.pool2(self.relu(self.conv2(x)))   # [B, 16, 23]
        x = self.pool3(self.relu(self.conv3(x)))   # [B, 16, 10]

        x = self.flatten(x)         # [B, 160]
        x = self.relu(self.fc1(x))  # [B, 256]
        x = self.relu(self.fc2(x))  # [B, 64]
        x = self.sigmoid(self.fc3(x))  # [B, 1]
        return x
```

### Forward pass data flow

Trace a batch of protein sequences through the model. Starting from integer tokens `[B, 200]`:

1. **Embedding:** `[B, 200] → [B, 200, 2]` via a learned lookup table (21 × 2). Transpose to `[B, 2, 200]` for Conv1d.
2. **Conv block 1:** kernel size 5, 16 filters. `[B, 2, 200] → [B, 16, 196]` → MaxPool(4) → `[B, 16, 49]`
3. **Conv block 2:** kernel size 3, 16 filters. `[B, 16, 49] → [B, 16, 47]` → MaxPool(2) → `[B, 16, 23]`
4. **Conv block 3:** kernel size 3, 16 filters. `[B, 16, 23] → [B, 16, 21]` → MaxPool(2) → `[B, 16, 10]`
5. **Flatten:** `[B, 16, 10] → [B, 160]`
6. **Dense layers:** `160 → 256 → 64 → 1` with ReLU activations and a final sigmoid

Total: 59,515 parameters. Note 4's MLP has ~28,000 parameters on its 24-dimensional hand-crafted features --- smaller, but limited to the patterns those features capture. The CNN has more parameters but operates on the raw sequence, so it can discover patterns the featurizer missed.

### Why sigmoid + BCE instead of softmax + cross-entropy?

Note 4's MLP outputs 2 logits and uses `CrossEntropyLoss`. This model outputs a single probability via sigmoid and uses `BCELoss`. Both formulations are equivalent for binary classification --- the single-output version just makes the binary nature explicit. `CrossEntropyLoss` with 2 classes is mathematically identical to `BCEWithLogitsLoss`.

## Training

Standard supervised training with BCE loss and Adam:

```python
import torch.optim as optim

def train(model, train_data, val_data, epochs=15):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0

        for batch_x, batch_y in train_data:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            correct += ((out > 0.5).float() == batch_y).sum().item()
            total += batch_y.size(0)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_data:
                out = model(batch_x)
                val_loss += criterion(out, batch_y).item()
                val_correct += ((out > 0.5).float() == batch_y).sum().item()
                val_total += batch_y.size(0)

        print(f"Epoch {epoch:2d} | "
              f"train_loss={train_loss/len(train_data):.4f} "
              f"train_acc={correct/total:.4f} | "
              f"val_loss={val_loss/len(val_data):.4f} "
              f"val_acc={val_correct/val_total:.4f}")

model = SolubilityNet()
train(model, train_data, val_data, epochs=15)
```

### Results

After 15 epochs with default hyperparameters (matching the dmol.pub setup):

| Metric | Epoch 0 | Epoch 15 |
|--------|---------|----------|
| Train accuracy | ~52% | ~65% |
| Val accuracy | ~55% | ~56% |

This is barely above random (50%). Honest, but underwhelming. The [dmol.pub textbook](https://dmol.pub/dl/layers.html) reports the same behavior and notes that the state-of-the-art baseline on this dataset achieves ~77% accuracy.

What is going wrong? Several things:

1. **Embedding dimension 2 is tiny.** Two dimensions can represent a rough hydrophobicity axis, but not the full physicochemical diversity of 20 amino acids. Increasing to 16 or 32 helps substantially.
2. **15 epochs is not enough.** The model has not converged --- validation loss is still decreasing. Training for 50--100 epochs with early stopping (Section 7 of Note 4) gives the model time to learn.
3. **No regularization.** Note 4's MLP uses dropout; this model has none. Adding `nn.Dropout(0.3)` after the ReLU activations in the dense layers reduces overfitting.
4. **Learning rate.** Adam's default learning rate (1e-3) may be too aggressive for this architecture. Reducing to 3e-4 often helps.

These are exactly the training improvements covered in Note 4: class weighting (Section 6), early stopping (Section 7), and systematic debugging (Section 8). The architecture provides a better inductive bias; the training recipe turns that potential into actual performance.

## What convolutions learn on proteins

Even with the minimal 2D embeddings, the convolutional filters learn to detect meaningful sequence patterns. The three layers capture features at different scales:

- **Layer 1** (kernel size 5): detects short motifs --- dipeptide and tripeptide patterns that correlate with local secondary structure and solubility. A 5-residue window is roughly one turn of an alpha helix.
- **Layer 2** (kernel size 3, after pooling): operates on compressed representations, effectively seeing ~12-residue windows in the original sequence. This is the scale of small structural elements like beta hairpins.
- **Layer 3** (kernel size 3, after two pooling layers): sees ~24-residue effective windows. This is the scale of full secondary structure elements and small domains.

The max pooling layers between convolutions serve two purposes: they reduce the sequence length (making the model efficient) and they make the features position-invariant (a hydrophobic patch near the N-terminus produces the same pooled signal as one near the C-terminus).

## MLP vs. CNN

| | Note 4 MLP | This CNN |
|---|---|---|
| Input representation | Hand-crafted features (24 dims) | Integer tokens → embedding (2 dims/position) |
| Positional information | None (global summary only) | Full sequence preserved |
| Feature design | Manual (AA frequencies, hydrophobicity, charge) | Learned (embedding + conv filters) |
| Parameters | ~28K | 59.5K |
| Interpretability | High (each input feature has a clear meaning) | Lower (learned filters, less transparent) |

The tradeoff is clear: Note 4's hand-crafted features are compact, interpretable, and require domain knowledge to design. The CNN learns its own features from raw sequence tokens --- more expressive, but a black box. When data is plentiful, learned representations win; when data is scarce or interpretability matters, hand-crafted features remain competitive.

## Running it yourself

```bash
python train.py
```

The code is two files. `model.py` has the embedding, convolutions, and dense layers. `train.py` downloads the data, sets up loaders, trains, and evaluates. No config files, no abstractions. Adapted from the solubility example in White et al.'s [*Deep Learning for Molecules and Materials*](https://dmol.pub/dl/layers.html).

For a better result, try these changes:
- `embedding_dim=16` instead of 2
- Add `nn.Dropout(0.3)` after each ReLU in the dense layers
- Train for 50 epochs with early stopping (patience=10)
- Use `BCEWithLogitsLoss` and remove the sigmoid from the model (numerically stabler)
