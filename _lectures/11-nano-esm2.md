---
layout: post
title: "Code Walkthrough: nano-esm2"
date: 2026-03-25
description: "Build ESM2 from scratch in 288 lines of PyTorch — masked language modeling for protein sequences."
course: "2026-spring-protein-ai"
course_title: "Protein & Artificial Intelligence"
course_semester: "Spring 2026"
lecture_number: 7
preliminary: false
toc:
  sidebar: left
related_posts: false
---

<div style="background: var(--global-code-bg-color); border-left: 4px solid var(--global-theme-color); padding: 1em 1.2em; margin-bottom: 2em; border-radius: 4px;">
<strong>Code:</strong> <a href="https://github.com/sungsoo-ahn/nano-protein-ais/tree/master/esm2">sungsoo-ahn/nano-protein-ais/esm2</a><br>
<strong>Files:</strong> <code>model.py</code> (288 lines) + <code>train.py</code> (195 lines)<br>
<strong>Parameters:</strong> 50.4M &nbsp;|&nbsp; <strong>Training:</strong> 13 min on 1x RTX 3090
</div>

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Companion to <a href="{{ '/lectures/07-protein-language-models/' | relative_url }}">Lecture: Protein Language Models</a>. The lecture covers the theory; this page builds it from scratch.</em>
</p>

You know how GPT learns language by predicting the next word? ESM2 does the same thing for proteins, except it fills in the blanks -- BERT-style. Mask out some amino acids, ask the model to guess them from context, and the representations it builds along the way encode deep biological knowledge: which residues are structurally important, which substitutions are tolerable, even hints of 3D structure. All from sequence alone.

Let's build one from scratch. 288 lines of model code, 50.4M parameters, trains in 13 minutes on a single RTX 3090.

## Proteins are sentences

A protein is a chain of amino acids -- there are 20 of them (Alanine, Cysteine, ..., Tyrosine), each represented by a single letter: `ACDEFGHIKLMNPQRSTVWY`. A typical protein might be 200-500 residues long. So a protein sequence looks like:

```
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKLVERR
```

This is not random. Like natural language, protein sequences have grammar. Hydrophobic residues cluster in the core, charged residues appear on the surface, cysteines pair into disulfide bonds. Certain motifs -- `GxGxxG` for nucleotide binding, `CxxC...CxxC` for zinc fingers -- recur across millions of proteins. A language model can learn all of this.

Our vocabulary is tiny: 20 amino acids plus 5 special tokens (PAD, MASK, CLS, EOS, UNK) = 25 tokens total. Compare that to GPT's 50K+ tokens. Protein language is compact.

## The architecture

ESM2 is a standard Transformer encoder, but with two modern upgrades that matter:

**RoPE (Rotary Position Embeddings).** Instead of adding a learned position vector to each token, RoPE rotates the query and key vectors by a position-dependent angle. The core idea is elegant:

```python
q_rotated = q * cos(pos * freq) + rotate_half(q) * sin(pos * freq)
```

Why bother? Because the dot product between two rotated vectors depends only on their *relative* distance, not their absolute positions. Position 10 attending to position 7 looks the same as position 100 attending to position 97. This is exactly what you want for proteins -- a helix motif works the same way whether it starts at residue 5 or residue 50.

**SwiGLU activation.** The feed-forward network uses a gated mechanism instead of a plain ReLU:

```python
# SwiGLU(x) = W3 * (SiLU(W1*x) * W2*x)
def forward(self, x):
    return self.w3(F.silu(self.w1(x)) * self.w2(x))
```

That element-wise multiply is the gate -- it lets the network selectively pass information. Consistently outperforms ReLU in practice. The cost is an extra weight matrix (three projections instead of two), but the tradeoff is worth it.

The full stack: token embedding (25 -> 512), 12 Transformer blocks with pre-LayerNorm, 16 attention heads, then a final LayerNorm and linear head back to 25 logits. That's 50.4M parameters. The `TransformerBlock` itself is just four lines of logic:

```python
# Pre-LayerNorm: normalize before attention/FFN, not after
attn_out, attn_weights = self.attention(self.ln1(x), attention_mask)
x = x + self.dropout(attn_out)
x = x + self.dropout(self.ffn(self.ln2(x)))
```

Pre-LayerNorm means you normalize before each sublayer rather than after. It makes training more stable, especially for deeper models. This is now standard practice.

## Masked Language Modeling

Training is BERT's masked language modeling (MLM). For each sequence, randomly select 15% of the amino acid positions, then apply the 80/10/10 strategy:

- **80%** of selected positions get replaced with `[MASK]`
- **10%** get replaced with a random amino acid
- **10%** stay unchanged

Why not just mask everything with `[MASK]`? Because at inference time there are no `[MASK]` tokens. The 10% random and 10% unchanged force the model to maintain good representations for *every* position, not just masked ones.

```
Original: M A E K T G V L
Selected: . . * . . * . .    (15% selected)
Masked:   M A [MASK] K T G V L   (position 3: MASK, position 6: kept as-is)
Labels:   - - E - - G - -        (only compute loss on selected positions)
```

The loss is standard cross-entropy, but only computed on the selected positions (everything else gets `ignore_index=-100`).

## Training: from random to 92.5% accuracy

Here are the real numbers from training on 500 UniRef50 sequences for 200 epochs:

| Metric | Start | End |
|--------|-------|-----|
| Loss | 4.47 | 0.24 |
| Masked accuracy | 2.8% | 92.5% |

The initial loss of 4.47 is a bit above $$\ln(25) = 3.2$$ (uniform random over the vocab) -- the freshly initialized model's predictions are worse than uniform because the random weights create uneven logit distributions. Within a few epochs, the loss drops quickly as the model learns amino acid frequencies. Then it grinds lower as it picks up local motifs and positional patterns.

Training takes about 13 minutes on an RTX 3090. The learning rate schedule is cosine decay with linear warmup (500 steps), AdamW with weight decay 0.01, gradient clipping at 1.0.

Now, 92.5% accuracy on 500 sequences means the model has *memorized* them. That's deliberate -- this is an overfitting run to verify the architecture works end to end. Can the model fit this data perfectly? If not, something is wrong. If it can, your forward pass, loss, masking, and optimization are all correct. Then you scale to more data.

## What does the model actually learn?

Even on 500 sequences, the model picks up real patterns. First, amino acid frequencies -- Leucine (L) appears ~9% of the time, Tryptophan (W) barely 1%. That's the easy part of the loss. Then local motifs: certain pairs and triplets co-occur, hydrophobic residues cluster in runs, Proline follows Glycine. The model learns to exploit these short-range dependencies through attention. Finally, positional patterns -- Methionine almost always starts a protein, signal peptides have hydrophobic stretches near the N-terminus.

The real ESM2 (650M parameters, 250M sequences, weeks on hundreds of GPUs) learns enough from sequence alone that its embeddings can predict 3D protein structure. That was the headline result: language model representations, with no explicit structural training, encode enough information to fold proteins. Our version won't do that, but every architectural idea is the same.

## This vs. the real thing

| | nano-esm2 | ESM2 (Meta) |
|---|---|---|
| Parameters | 50.4M | 650M |
| Training data | 500 sequences | 250M sequences |
| Training time | 13 min / 1 GPU | Weeks / hundreds of GPUs |
| Masked accuracy | 92.5% (memorized) | ~50% (generalized) |
| Model code | 288 lines | Thousands |

The gap is instructive. Our model gets 92.5% by memorizing 500 sequences. The real ESM2 gets ~50% on held-out data from 250M sequences -- predicting the right amino acid half the time, across millions of unseen proteins. That means it has learned deep statistical regularities of protein evolution.

## Running it yourself

```bash
# Train the default 8M model
python train.py --data_path data/sequences/uniref50_subset.fasta

# Or scale up to match the 50.4M run
# (edit the constants at the top of model.py: NUM_LAYERS=12, HIDDEN_DIM=512, NUM_HEADS=16)
python train.py --data_path data/sequences/uniref50_subset.fasta
```

The code is two files. `model.py` (288 lines) has everything -- vocabulary, RoPE, SwiGLU, attention, the Transformer, masking, and loss. `train.py` adds a FASTA parser, dataset, and training loop. No config objects, no inheritance hierarchies, no abstractions. Fork it, break it, learn from it.

That's the whole thing. BERT for proteins, in a file you can read top to bottom in 15 minutes.
