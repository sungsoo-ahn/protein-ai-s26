---
layout: post
title: "Code Walkthrough: nano-proteinmpnn"
date: 2026-04-06
description: "Build ProteinMPNN from scratch in 448 lines of PyTorch — inverse folding with graph neural networks."
course: "2026-spring-protein-ai"
course_title: "Protein & Artificial Intelligence"
course_semester: "Spring 2026"
lecture_number: 12
preliminary: false
toc:
  sidebar: left
related_posts: false
---

<div style="background: var(--global-code-bg-color); border-left: 4px solid var(--global-theme-color); padding: 1em 1.2em; margin-bottom: 2em; border-radius: 4px;">
<strong>Code:</strong> <a href="https://github.com/sungsoo-ahn/nano-protein-ais/tree/master/proteinmpnn">sungsoo-ahn/nano-protein-ais/proteinmpnn</a><br>
<strong>Files:</strong> <code>model.py</code> (448 lines) + <code>train.py</code> (231 lines)<br>
<strong>Parameters:</strong> 2.6M &nbsp;|&nbsp; <strong>Training:</strong> ~100 sec on 1x RTX 3090
</div>

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Companion to <a href="{{ '/lectures/10-proteinmpnn/' | relative_url }}">Lecture: ProteinMPNN</a>. The lecture covers the theory; this page builds it from scratch.</em>
</p>

You give it a protein backbone -- just the 3D coordinates of the skeleton atoms -- and it tells you which amino acids should go at each position. That is **inverse folding**: going from structure back to sequence. Nature solves this problem the forward direction (sequence folds into structure), but we want to run the tape backwards.

Think of it like this: you have a building's steel frame, and you need to figure out which bricks, panels, and glass go where so the whole thing holds together. The backbone is the frame. The amino acids are the building materials. ProteinMPNN picks them.

Our implementation is 448 lines of model code, 2.6M parameters, trains on 9 proteins in about 100 seconds on an RTX 3090, and memorizes them perfectly. It will not generalize to new proteins (the real ProteinMPNN trains on 18,000+ structures), but it implements every architectural idea faithfully. Let's build it.

## The Data

We train on 9 small PDB files downloaded from RCSB -- tiny proteins like crambin (1CRN, 46 residues), ubiquitin (1UBQ, 76 residues), and protein G (2GB1, 56 residues). A PDB file is a text format from the 1970s where each `ATOM` line gives you an atom name, residue number, and x/y/z coordinates in angstroms. We only care about four backbone atoms per residue:

- **N** -- the amide nitrogen
- **CA** -- the alpha carbon (the central hub)
- **C** -- the carbonyl carbon
- **O** -- the carbonyl oxygen

The parser in `train.py` reads these four atoms per residue, converts the three-letter amino acid codes (`ALA` -> `A`, `ARG` -> `R`, ...) to indices in a vocabulary of 20, and returns coordinate tensors of shape `[L, 3]` where `L` is the protein length. For crambin, that is `[46, 3]` -- 46 residues, each with x/y/z for each backbone atom.

The native sequence for 1CRN is `TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN`. That is our ground truth. Can we recover it from just the backbone coordinates?

## Features: Turning Geometry into Tensors

The model never sees raw coordinates. Instead, we build a **k-nearest-neighbor graph** on the CA atoms (k=30), then compute rich features on the nodes and edges.

**Node features** (15-dimensional per residue): We compute three backbone dihedral angles (phi, psi, omega) and encode each as sin/cos pairs (6 values), then flatten the 3x3 local coordinate frame into 9 values. That gives each residue a 15-dimensional feature vector capturing its local geometry.

The local frame uses Gram-Schmidt orthogonalization on the N/CA/C triangle -- the same procedure as in AlphaFold2 and RFDiffusion:

```python
x = normalize(C - CA)           # x-axis: CA → C
z = normalize(cross(x, N - CA)) # z-axis: normal to backbone plane
y = cross(z, x)                 # y-axis: completes right-handed frame
R = stack([x, y, z], dim=-1)    # 3×3 rotation matrix
```

The 9 values are this 3×3 matrix flattened row-wise. Why include a local frame? It makes features **invariant to global rotation** -- the same residue geometry produces the same features regardless of how the protein is oriented in space. Rotate the entire protein by 90°, and every residue's local frame rotates with it, but the *relationship* between the frame and the backbone atoms stays identical.

**Edge features** (324-dimensional per edge): For each edge connecting residue `i` to neighbor `j`, we compute:
- **RBF distances** between all 4x4 pairs of backbone atoms (N-N, N-CA, N-C, N-O, CA-N, ...). Each distance is encoded with 16 Gaussian basis functions centered from 0 to 20 angstroms. That is 16 atom pairs times 16 RBFs = 256 features.
- **Sequence separation** -- how far apart `i` and `j` are in the chain, clamped to [-32, 32] and one-hot encoded (65 features).
- **Direction** -- the unit vector from `i` to `j` projected into `i`'s local coordinate frame (3 features).

The direction projection deserves unpacking. The raw vector from CA_i to CA_j is in global coordinates and would change if you rotated the protein. Projecting it into residue i's local frame via $$R_i^T \cdot (\text{CA}_j - \text{CA}_i)$$ gives the direction *from residue i's perspective*:

```python
R_src = rots[src]  # local frame rotation matrices for source residues
direction = coords_CA[dst] - trans[src]  # global direction vector
direction_local = torch.einsum("eij,ej->ei", R_src.transpose(-1, -2), direction)
direction_local = direction_local / (direction_local.norm(dim=-1, keepdim=True) + 1e-8)
```

If residue j is "above" residue i relative to i's backbone plane, that shows up as a positive z-component in local coordinates -- regardless of how the protein sits in the lab frame. Same principle as IPA's point transforms in AlphaFold2.

Total: 256 + 65 + 3 = 324 edge features. This is the geometry the model works with.

## The Model: Encode Structure, Decode Sequence

The architecture has two halves: an **MPNN encoder** that processes the structure graph, and an **autoregressive decoder** that predicts amino acids one at a time.

**Encoder** (3 layers of message passing): Each layer does the same thing. For every edge, concatenate the source node embedding, the destination node embedding, and the edge features, push through an MLP to get a message, aggregate all incoming messages at each node via `index_add_`, then update the node with another MLP plus a residual connection and layer norm. After 3 rounds of this, each node's hidden state (dimension 192) has absorbed information from its structural neighborhood.

```python
# One message-passing step (simplified)
messages = message_mlp(cat([h[src], h[dst], edge_attr]))  # per edge
agg = zeros(L, hidden_dim).index_add_(0, dst, messages)   # aggregate at nodes
h = norm(h + update_mlp(cat([h, agg])))                    # update with residual
```

**Decoder** (3 layers of causal attention): The decoder takes the amino acid tokens (during training, the ground truth; during design, sampled tokens), embeds them, and runs them through transformer-style layers. Each layer has three sublayers in order:

1. **Causal self-attention** -- position i can only attend to positions decoded before it (determined by the random permutation mask). This prevents information leaking from future positions.
2. **Cross-attention to the encoder output** -- queries come from the decoder's current token representations, keys and values from the encoder's structural embeddings. Unlike self-attention, cross-attention is *not* masked -- every position can attend to all encoder states because the structure is fully known. This is how the decoder reads structural information while generating sequence.
3. **Feed-forward network** -- ReLU with 4× expansion (192 → 768 → 192), same pattern as the encoder.

Each sublayer uses pre-LayerNorm and a residual connection:

```python
def forward(self, h, encoder_output, causal_mask, seq_mask):
    h_attn, _ = self.self_attn(h, h, h, attn_mask=causal_mask)
    h = self.norm1(h + h_attn)
    h_cross, _ = self.cross_attn(h, encoder_output, encoder_output)  # no mask
    h = self.norm2(h + h_cross)
    h = self.norm3(h + self.ffn(h))
    return h
```

The output projection gives logits over 20 amino acids at each position.

The full forward pass for one protein:

```
backbone coords --> build k-NN graph --> compute node/edge features
                --> encoder (3x message passing) --> structural embeddings
                --> decoder (3x causal attention) --> logits [L, 20]
                --> cross-entropy loss against ground-truth sequence
```

## Random-Order Decoding: The Clever Trick

A standard autoregressive model decodes left-to-right: position 1, then 2, then 3. But proteins are not text. There is nothing special about the N-to-C direction -- residue 30 might be spatially close to residue 5, not residue 29.

ProteinMPNN decodes in a **random permutation order**. Each forward pass generates a new random permutation with `torch.randperm(L)`, then builds a causal mask where position `i` can only attend to positions that come earlier in *that particular* permutation:

```python
def create_decoding_mask(decoding_order):
    """mask[i,j] = True if j is decoded before or at position i."""
    rank = zeros(L)
    for step, pos in enumerate(decoding_order):
        rank[pos] = step
    return rank.unsqueeze(0) <= rank.unsqueeze(1)
```

During training, every epoch sees a different random order, so the model cannot rely on "the previous residue in the chain is always available." It has to learn to predict each amino acid from *any* subset of its neighbors. This is what makes ProteinMPNN so robust -- at inference time it works regardless of which positions are already known.

## Training: Memorizing 9 Proteins

The training setup is dead simple: full-batch on 9 proteins, AdamW with learning rate 1e-3, cosine schedule with warmup, gradient clipping at 1.0. We train for 500 epochs, which takes about 100 seconds on an RTX 3090.

The loss curve is satisfying. Cross-entropy starts at 3.0 (random guessing over 20 amino acids -- $$\ln(20) = 3.0$$, so the model starts at maximum entropy). It drops steadily and bottoms out near 0.0003 by epoch 500. Sequence recovery -- the fraction of positions where the predicted amino acid matches the native -- goes from 9% (roughly random at 1/20 = 5%, but not exactly due to amino acid frequency biases) to 100%.

The model has perfectly memorized all 9 training proteins. Given any of their backbones, it recovers the exact native sequence.

**But does it generalize?** On held-out proteins it has never seen:

| Protein | Recovery |
|---------|----------|
| 1CRN    | 2-13%    |
| 1UBQ    | 3-5%     |
| 2GB1    | 3-7%     |

That is essentially random. With only 9 training examples, there is no reason to expect generalization. The model has learned 9 specific structure-to-sequence mappings, not the general rules of protein design.

## The `design()` Method

At inference, the workflow changes. You encode the backbone structure once, then sample amino acids autoregressively:

```python
# Encode structure (done once)
encoder_out = self.encoder(node_feats, edge_index, edge_feats)

# Sample in random order
order = torch.randperm(L)
seq = torch.randint(0, 20, (L,))  # start with random tokens
logits = self.decoder(seq, encoder_out, order, mask)
probs = softmax(logits / temperature)
for pos in range(L):
    seq[pos] = multinomial(probs[pos], 1)
```

The `temperature` parameter controls diversity. At `temperature=0.1`, the model is nearly deterministic -- it picks the highest-probability amino acid almost every time. At `temperature=1.0`, you get diverse samples that explore more of sequence space.

## What This Teaches vs. The Real Thing

The real ProteinMPNN (Dauparas et al., 2022) trains on roughly 18,000 structures from the PDB and achieves 50-60% native sequence recovery on novel proteins. That is remarkable -- given only a backbone, it recovers more than half of the amino acids that evolution settled on.

Our nano implementation has 2.6M parameters (the real one is similar in scale), implements the same k-NN graph construction, the same RBF distance encoding, the same MPNN encoder architecture, the same random-order autoregressive decoder, and the same causal masking trick. The difference is entirely in data: 9 proteins vs. 18,000.

What you learn from reading this code:

1. **Graph neural networks on protein structure.** How to turn a set of 3D coordinates into a k-NN graph with rich geometric edge features.
2. **Message passing.** The MPNN pattern of compute-messages, aggregate, update -- the workhorse of geometric deep learning.
3. **Random-order autoregressive decoding.** A powerful idea that appears in XLNet (for language) and ProteinMPNN (for proteins). The causal mask is the key mechanism.
4. **The inverse folding problem.** Structure constrains sequence, but does not determine it uniquely. Multiple sequences can fold to the same backbone.

The whole thing is 448 lines of `model.py` and 231 lines of `train.py`. Fork it, swap in a larger dataset, and see how far you can push it.
