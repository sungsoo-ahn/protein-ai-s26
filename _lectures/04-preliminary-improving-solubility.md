---
layout: post
title: "Case Study: Predicting Protein Solubility"
date: 2026-03-03
description: "An end-to-end case study—building an MLP solubility predictor from sequence features, and learning to evaluate honestly with sequence-identity splits, class weighting, and early stopping."
course: "2026-spring-protein-ai"
course_title: "Protein & Artificial Intelligence"
course_semester: "Spring 2026"
lecture_number: 4
preliminary: true
toc:
  sidebar: left
related_posts: false
collapse_code: true
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;"><em>This is Preliminary Note 4 for the Protein &amp; Artificial Intelligence course (Spring 2026), co-taught by Prof. Sungsoo Ahn and Prof. Homin Kim at KAIST. It applies everything from Preliminary Notes 1--3 in a complete case study. You should work through this note before the first in-class lecture.</em></p>

## Introduction

Time to build something real.
This note is a single, end-to-end project: predicting whether a protein will be soluble when expressed in *E. coli*.

The model is an MLP that takes flattened one-hot encoded sequences --- the same representation from Preliminary Note 2 --- and outputs a solubility prediction.
The architecture is deliberately simple so we can focus on the problems that arise *around* the model: misleading evaluation, class imbalance, and overfitting.
Each section follows the same arc: *observe a problem → understand why it happens → introduce a technique that addresses it → show the improvement*.

By the end, you will have a working solubility predictor and a practical toolkit for diagnosing and fixing the most common training problems in protein machine learning.
The main lectures introduce more powerful architectures (CNNs, transformers, GNNs) and additional input modalities (3D structure, learned embeddings).

### Roadmap

| Section | Topic | What You Will Learn |
|---|---|---|
| 1 | The Solubility Prediction Problem | Why this problem matters and what makes it amenable to ML |
| 2 | The Model: An MLP on Sequence Features | Building a solubility predictor from flattened one-hot encodings |
| 3 | Data Preparation | Dataset/DataLoader setup, train/val/test splitting |
| 4 | Training and Evaluation | Training script, evaluation metrics beyond accuracy, precision-recall |
| 5 | Evaluating Properly: Sequence-Identity Splits | Why random splits overestimate performance, and how to fix it |
| 6 | Handling Class Imbalance | Weighted loss functions for imbalanced datasets |
| 7 | Knowing When to Stop: Early Stopping | Detecting the overfitting point and saving the best model |
| 8 | Debugging and Reproducibility | NaN detection, shape checks, single-batch overfit test, seed setting |

### Prerequisites

This note assumes you have worked through Preliminary Notes 1--3: tensors, neural network architectures, loss functions, optimizers, the training loop, data loading, and validation.

---

## 1. The Solubility Prediction Problem

### Why Solubility Prediction Matters

Expressing recombinant proteins is a core technique in structural biology, biotechnology, and therapeutic development.
When a target protein aggregates into inclusion bodies instead of dissolving in the cytoplasm, downstream applications --- crystallography, assays, drug formulation --- become much harder or impossible.
A computational model that predicts solubility from sequence alone can guide construct design and save weeks of experimental effort.

### What Makes This Problem Amenable to Machine Learning?

Solubility is influenced by sequence-level properties: amino acid composition, charge distribution, hydrophobicity patterns, and the presence of certain sequence motifs.
These patterns are learnable from data.

This is a **binary classification** task: given a protein sequence, predict whether it will be soluble (1) or insoluble (0).
We use the tools from Preliminary Note 3: cross-entropy loss, data loading with `DataLoader`, and the training loop.

---

## 2. The Model: An MLP on Sequence Features

The input representation is the flattened one-hot encoding from Preliminary Note 2: each protein of length $$L$$ becomes a matrix of shape $$(L, 20)$$, padded to a fixed maximum length $$L_{\max}$$, then flattened to a single vector of dimension $$L_{\max} \times 20$$.
This discards the sequential ordering --- the MLP treats position 1 and position 100 as unrelated inputs --- but it is enough to learn composition-level and position-level patterns that correlate with solubility.

The architecture mirrors the `SolubilityPredictor` from Preliminary Note 2, with an additional hidden layer and dropout for regularization:

```python
import torch
import torch.nn as nn

class SolubilityMLP(nn.Module):
    """
    MLP for predicting protein solubility from flattened one-hot sequences.

    Architecture:
    1. Input: flattened padded one-hot vector (max_len × 20)
    2. Three hidden layers with ReLU activation and dropout
    3. Linear output: predict soluble (1) vs. insoluble (0)
    """

    def __init__(self, input_dim=2000, hidden_dim=128, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x shape: (batch, input_dim) — flattened padded one-hot
        return self.net(x)
```
<div class="caption mt-1">With <code>max_len=100</code> and 20 amino acids, <code>input_dim=2000</code>. The three hidden layers (128 → 128 → 64) progressively compress the representation before the final classification layer.</div>

With $$L_{\max} = 100$$ and 20 amino acids, the input dimension is 2,000.
The model has roughly 290,000 parameters --- large relative to typical protein dataset sizes, which is why regularization (dropout) and early stopping (Section 7) matter.

An MLP on flattened sequences has a clear limitation: it treats each input dimension independently, with no built-in notion of *locality*.
A cluster of five hydrophobic residues in a row is a strong signal for a transmembrane helix (likely insoluble), but the MLP must learn this from data rather than having it built into the architecture.
The main lectures introduce convolutional networks (which exploit sequential structure) and transformers (which learn long-range dependencies) to address this limitation.

---

## 3. Data Preparation

Split the dataset into train/validation/test with `stratify` to maintain class balance, wrap the flattened feature tensors and labels in a `TensorDataset`, and create `DataLoader` objects with `shuffle=True` for training and `shuffle=False` for evaluation.

---

## 4. Training and Evaluation

### The Training Script

The training pipeline combines single-epoch training with validation monitoring: after each pass through the training data, the model is evaluated on the validation set, and the best-performing checkpoint is saved.

```python
def train_model(model, train_loader, val_loader, epochs=100, lr=1e-3):
    """Full training pipeline with validation monitoring."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # --- Training phase ---
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # --- Validation phase ---
        val_loss, _, _ = evaluate(model, val_loader, criterion, device)

        # --- Save best model ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')

        # --- Logging ---
        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f}")

    # Load the best model before returning
    model.load_state_dict(torch.load('best_model.pt'))
    return model

# Instantiate the model and inspect its size
model = SolubilityMLP(input_dim=2000, hidden_dim=128, num_classes=2)
n_params = sum(p.numel() for p in model.parameters())
print(f"SolubilityMLP: {n_params:,} parameters")

# Train with validation monitoring
trained_model = train_model(model, train_loader, val_loader, epochs=50)
```

### Evaluation: Beyond Accuracy

Accuracy alone is deceptive.
If 70% of proteins are soluble, a model that *always* predicts "soluble" achieves 70% accuracy while being completely useless.

```python
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score)

def evaluate_classifier(model, test_loader, device):
    """Evaluate a binary classifier with multiple metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            x = batch_x.to(device)
            y = batch_y

            logits = model(x)                               # Raw scores
            probs = F.softmax(logits, dim=-1)               # Probabilities
            preds = logits.argmax(dim=-1)                   # Predicted class

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())     # P(soluble)

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary'
    )
    auc = roc_auc_score(all_labels, all_probs)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")

    return accuracy, precision, recall, f1, auc
```

### Understanding the Metrics

All classification metrics are defined in terms of four counts: true positives (TP), false positives (FP), true negatives (TN), and false negatives (FN).

$$
\text{Precision} = \frac{TP}{TP + FP}, \qquad \text{Recall} = \frac{TP}{TP + FN}
$$

$$
F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}
$$

The $$F_1$$ score is the harmonic mean of precision and recall --- it is high only when *both* are high.

**AUC-ROC** (Area Under the Receiver Operating Characteristic curve) plots the true positive rate (recall) against the false positive rate ($$FP / (FP + TN)$$) as the classification threshold varies from 0 to 1, and computes the area under this curve.
An AUC of 1.0 means the model achieves perfect separation at some threshold; 0.5 means no better than random.

| Metric | Question It Answers | Protein Example |
|---|---|---|
| **Accuracy** | What fraction of all predictions are correct? | 85% of solubility predictions correct |
| **Precision** | Of positive predictions, what fraction truly are? | Of proteins predicted soluble, how many truly are? |
| **Recall** | Of true positives, what fraction did we detect? | Of truly soluble proteins, how many did we find? |
| **F1 Score** | Harmonic mean of precision and recall | Balance between missing soluble proteins and wasting experiments |
| **AUC-ROC** | How well does the model separate classes across all thresholds? | Overall ability to distinguish soluble from insoluble |

The precision-recall tradeoff deserves special attention.
In a drug discovery setting, where expressing each candidate is expensive, a biologist might want **high precision**: "I only want to express proteins that are very likely to be soluble."
By raising the classification threshold from 0.5 to 0.8, we predict fewer proteins as soluble but are more confident in those predictions.

Conversely, in a high-throughput screening setting with thousands of candidates, a biologist might prefer **high recall**: "I don't want to miss any potentially soluble protein."
Lowering the threshold to 0.3 captures more true positives at the cost of more false positives.

The AUC-ROC summarizes this tradeoff across all possible thresholds.
An AUC of 1.0 means perfect separation; 0.5 means the model is no better than random.

### Analyzing the Loss Curves

After training for 50 epochs, examine the loss curves.

<div class="col-sm-9 mt-3 mb-3 mx-auto">
    <img class="img-fluid rounded" src="{{ '/assets/img/teaching/overfitting_curves.png' | relative_url }}" alt="Training vs validation loss showing overfitting">
    <div class="caption mt-1"><strong>Training and validation loss curves.</strong> Training loss decreases steadily, but validation loss begins increasing after ~40 epochs --- the model is memorizing the training data rather than learning generalizable patterns.</div>
</div>

If the training loss decreases smoothly but the validation loss rises after approximately 40 epochs, the model is overfitting.
The gap between training accuracy (~98%) and validation accuracy (~72%) confirms this.
Sections 6 and 7 address this problem with weighted losses and early stopping.

---

## 5. Evaluating Properly: Sequence-Identity Splits

### The Problem

The solubility predictor achieves 85% validation accuracy.
Impressive? Not necessarily --- it depends on *how* we split the data.

If we used a random train/validation/test split, there is a high probability that some test proteins are closely related (>90% sequence identity) to proteins in the training set.
These homologous proteins almost certainly share the same solubility status.
The model can score well by memorizing similar sequences rather than learning true sequence-to-solubility patterns.
This is **data leakage** --- the test set contains information that was effectively available during training.

### The Solution: Sequence-Identity Splits

The fix relies on **sequence clustering**.
The idea is to group proteins so that any two proteins in the same group are similar (above some sequence-identity threshold), while proteins in different groups are dissimilar.

Concretely, a clustering tool like MMseqs2 compares every pair of proteins in the dataset and computes their **sequence identity**[^seqid] --- the fraction of aligned positions where the amino acids match.

[^seqid]: Sequence identity is reported as a percentage. Rough rules of thumb: above 90% identity, two proteins almost certainly share the same structure and function; above 30--50%, they likely share the same fold; below 20%, the relationship is ambiguous (the "twilight zone" of sequence alignment).
It then groups proteins into **clusters** using a threshold (say 30%): if protein A and protein B share $$\geq$$ 30% identity, they end up in the same cluster.
Each cluster has a representative sequence, and every other member is reachable from the representative through a chain of pairwise alignments above the threshold.

For example, with 10,000 proteins and a 30% identity threshold, you might get 3,000 clusters.
Some clusters contain a single unique protein; others contain dozens of close homologs.
The crucial property is that **proteins in different clusters share less than 30% identity**, so they are genuinely different from one another.

We then split at the **cluster** level, not the individual protein level: all members of a cluster go into the same split (train, validation, or test).
This ensures that no test protein is closely related to any training protein.

```python
import subprocess
import numpy as np

def create_sequence_identity_splits(fasta_file, identity_threshold=0.3, train_ratio=0.8):
    """Split proteins into train/val/test sets respecting sequence identity.

    Requires MMseqs2 to be installed (https://github.com/soedinglab/MMseqs2).
    """
    # Step 1: Cluster proteins at the specified identity threshold
    subprocess.run([
        'mmseqs', 'easy-cluster',
        fasta_file, 'clusters', 'tmp',
        '--min-seq-id', str(identity_threshold)
    ])

    # Step 2: Parse cluster assignments
    clusters = parse_cluster_file('clusters_cluster.tsv')

    # Step 3: Shuffle and split clusters (not individual proteins)
    cluster_ids = list(clusters.keys())
    np.random.shuffle(cluster_ids)

    n_clusters = len(cluster_ids)
    n_train = int(n_clusters * train_ratio)
    n_val = int(n_clusters * 0.1)

    train_clusters = cluster_ids[:n_train]
    val_clusters = cluster_ids[n_train:n_train + n_val]
    test_clusters = cluster_ids[n_train + n_val:]

    # Step 4: Collect protein IDs from the assigned clusters
    train_ids = [pid for c in train_clusters for pid in clusters[c]]
    val_ids = [pid for c in val_clusters for pid in clusters[c]]
    test_ids = [pid for c in test_clusters for pid in clusters[c]]

    return train_ids, val_ids, test_ids
```

### The Reality Check

When we retrain our model using sequence-identity splits instead of random splits, the test accuracy typically drops by 5--15 percentage points.
This drop reflects the true difficulty of the task: predicting solubility for proteins that are genuinely different from anything in the training set.

The random-split accuracy was a mirage.
The sequence-identity-split accuracy is the honest answer.
Any paper that reports performance without controlling for sequence similarity should be read with skepticism.

A word of caution: even 30% sequence identity splits may not be sufficient for all tasks.
Proteins from the same CATH[^cath] superfamily can share structural features despite having diverged below 30% identity.
For the most rigorous evaluation, consider splitting at the fold or superfamily level.

[^cath]: CATH is a hierarchical classification of protein domain structures: **C**lass (secondary structure content), **A**rchitecture (spatial arrangement), **T**opology (fold), and **H**omologous superfamily.

---

## 6. Handling Class Imbalance

### The Problem

After switching to sequence-identity splits, we notice another issue: the model's performance on **insoluble** proteins is much worse than on soluble ones.
Looking at the data, we find that 70% of our dataset is soluble and only 30% is insoluble.
The model has learned a shortcut: predicting "soluble" for everything gives 70% accuracy with no effort.

### Weighted Loss Functions

The simplest correction: assign higher weights to underrepresented classes, so that misclassifying an insoluble protein incurs a larger penalty:

```python
def compute_class_weights(labels, num_classes):
    """Compute inverse-frequency weights for class-balanced training."""
    counts = torch.bincount(labels.flatten(), minlength=num_classes).float()
    weights = 1.0 / (counts + 1)                    # Inverse frequency
    weights = weights / weights.sum() * num_classes  # Normalize
    return weights

# Apply to our solubility dataset
class_weights = compute_class_weights(train_labels, num_classes=2)
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

With weighted loss, the model is penalized more heavily for misclassifying the minority class (insoluble proteins).
This forces it to pay attention to features that distinguish insoluble proteins rather than defaulting to "soluble."

### Effect on the Model

After applying class weights, the overall accuracy may drop slightly (the model can no longer cheat by always predicting the majority class), but the **F1 score and recall for the minority class improve substantially**.
This is the metric that matters: a model that correctly identifies insoluble proteins is far more useful than one that achieves high accuracy by ignoring them.

---

## 7. Knowing When to Stop: Early Stopping

### The Problem

Even with regularization (dropout in our model), there comes a point when continued training hurts more than it helps.
Validation loss may start rising again after reaching its best value, indicating that the model is beginning to overfit to the training data.

### Early Stopping

**Early stopping** is a form of regularization based on *time* rather than architecture.
The idea: monitor validation performance during training and stop when it stops improving.

Why does this work as regularization?
In the early phases of training, the model learns general, transferable patterns.
As training continues, it gradually begins to memorize training-specific noise.
The point at which validation performance peaks is the sweet spot between underfitting and overfitting.

```python
class EarlyStopping:
    """Stop training when validation loss stops improving."""
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss):
        """Call once per epoch. Returns True if this is a new best model."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True   # New best — save checkpoint
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False  # No improvement
```

The training loop checks validation loss after every epoch and saves a checkpoint whenever validation improves.
If no improvement occurs for a set number of epochs (the **patience**), training halts and the best checkpoint is restored.

```python
early_stopping = EarlyStopping(patience=15)

for epoch in range(max_epochs):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, _, _ = evaluate(model, val_loader, criterion, device)

    if early_stopping.step(val_loss):
        torch.save(model.state_dict(), 'best_model.pt')

    if early_stopping.should_stop:
        print(f"Early stopping at epoch {epoch}")
        break

# Load the best model for final evaluation
model.load_state_dict(torch.load('best_model.pt'))
```
<div class="caption mt-1">Integrating the <code>EarlyStopping</code> class into the training loop. Each epoch, <code>step()</code> returns <code>True</code> when validation loss improves (save a checkpoint) and sets <code>should_stop = True</code> after <code>patience</code> epochs without improvement.</div>

For protein models with small datasets (and therefore noisy validation estimates), a patience of 10 to 20 epochs is typical — long enough to ride out random fluctuations, short enough to avoid wasting computation on a model that has stopped improving.

---

## 8. Debugging and Reproducibility

### Debugging Neural Networks

Neural networks can fail silently.
The code runs, the loss decreases, but predictions are useless.
A systematic debugging checklist:

1. **Check for NaN gradients** — iterate over `model.named_parameters()` and check `torch.isnan(param.grad).any()`. NaN gradients indicate numerical instability (often from a learning rate that is too large).

2. **Verify output range** — run a forward pass with `torch.no_grad()` and print `output.min()` / `output.max()`. For logits, values should be in a reasonable range (roughly $$[-10, 10]$$).

3. **Check shapes** — print input and output shapes at every stage. Shape mismatches (especially after `transpose` or `unsqueeze`) are the most common bug.

4. **Single-batch overfit test** — train on a single batch for 200 steps. If the loss does not approach zero, there is a bug in the architecture, loss function, or data pipeline. This is the single most important debugging technique.

### Reproducibility

Set all random seeds (`random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`) at the start of every experiment.
For full GPU determinism, additionally set `torch.backends.cudnn.deterministic = True` (may slow training by 10--20%).

### A Practical Checklist

Before declaring a model "trained," verify the following:

1. Training loss decreases steadily over epochs.
2. Validation loss decreases initially, then plateaus (not increases --- that signals overfitting).
3. The model can perfectly overfit a single batch (sanity check for bugs).
4. Gradients are finite (no NaN or Inf values).
5. Metrics on the test set are consistent with validation set metrics.
6. Results are reproducible when the same seed is used.

---

## Key Takeaways

1. **Solubility prediction** is a representative binary classification task that exercises every component of the ML pipeline: data preparation, model architecture, training, and evaluation.

2. **An MLP on flattened one-hot sequences** is a simple but effective starting point. It treats each input dimension independently, which suffices for learning composition-level and position-level patterns. The main lectures introduce architectures (CNNs, transformers) that exploit sequential and structural information.

3. **Evaluation metrics** must go beyond accuracy. Precision, recall, F1, and AUC-ROC tell a more complete story, especially for imbalanced datasets.

4. **Sequence-identity splits are mandatory** for honest evaluation of protein models. Random splits systematically overestimate performance due to data leakage from homologous sequences.

5. **Address class imbalance** with weighted losses. High accuracy on an imbalanced dataset is meaningless if the model ignores the minority class.

6. **Early stopping** saves the best model and prevents wasted computation. Use a patience of 10--20 epochs for protein tasks.

7. **Systematic debugging** catches silent failures. The single-batch overfit test is the most important sanity check.

---
