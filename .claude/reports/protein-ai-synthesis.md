# Protein & AI Course — Synthesis Report

Generated: 2026-02-18

## Note Index

| ID | Title | File | Lecture # | Status |
|----|-------|------|-----------|--------|
| P1 | AI Fundamentals | `protein-ai-s26-01-preliminary-ai-fundamentals.md` | 1 | processed |
| P2 | Features and Architectures | `protein-ai-s26-02-preliminary-features-and-architectures.md` | 2 | processed |
| P3 | Training | `protein-ai-s26-03-preliminary-training.md` | 3 | processed |
| P4 | Improving Solubility | `protein-ai-s26-04-preliminary-improving-solubility.md` | 4 | processed |
| L1 | Transformers and GNNs | `protein-ai-s26-04-transformers-gnns.md` | 5 | processed |
| L2 | Generative Models | `protein-ai-s26-05-generative-models.md` | 6 | processed |
| L3 | Protein Language Models | `protein-ai-s26-06-protein-language-models.md` | 7 | processed |
| L4 | AlphaFold | `protein-ai-s26-07-alphafold.md` | 8 | processed |
| L5 | RFDiffusion | `protein-ai-s26-08-rfdiffusion.md` | 9 | processed |
| L6 | ProteinMPNN | `protein-ai-s26-09-proteinmpnn.md` | 10 | processed |

---

## Concept Index

Alphabetical. Format: **Concept** — definition summary | Introduced in | Also used in

| Concept | Definition | Introduced | Used in |
|---------|-----------|------------|---------|
| Activation Function | Nonlinear function applied element-wise (ReLU, sigmoid, GELU, softmax) | P2 | P3, P4, L1 |
| Adam Optimizer | Adaptive learning rates per parameter; AdamW recommended as default | P3 | P4 |
| AlphaFold2 | DeepMind's structure prediction system; MSA+pair → Evoformer → structure module → 3D coords | L4 | L5, L6 |
| Attention Mechanism | Q/K/V projections, scaled dot-product, softmax weighting | L1 | L3, L4, L5, L6 |
| Autoregressive Decoding | Generates sequence one token at a time, each conditioned on previous | L6 | — |
| Bias-Variance Tradeoff | Too simple = high bias (underfit), too complex = high variance (overfit) | P3 | P4 |
| Binary Cross-Entropy | $$L_\text{BCE} = -\frac{1}{n}\sum[y\log\hat{y} + (1-y)\log(1-\hat{y})]$$ | P3 | P4 |
| Class Imbalance | Majority class dominates training; fixed with weighted loss | P4 | — |
| Classifier-Free Guidance | Interpolate conditional/unconditional predictions; s>1 strengthens conditioning | L2 | L5 |
| Contact Prediction | Predicting spatially close residue pairs from sequence | L3 | L4 |
| Cross-Entropy | $$L_\text{CE} = -\sum y_c \log \hat{y}_c$$; generalizes BCE to C classes | P3 | L3 |
| Design Pipeline | RFDiffusion → ProteinMPNN → AlphaFold validation workflow | L6 | — |
| Diffusion Model | Forward: add noise; Reverse: neural network predicts noise; $$L_\text{simple} = \mathbb{E}[\|\epsilon - \epsilon_\theta(x_t,t)\|^2]$$ | L2 | L5 |
| Early Stopping | Stop training when validation loss stops improving for patience epochs | P4 | — |
| ELBO | Evidence Lower Bound: $$\mathbb{E}[\log p(x|z)] - D_\text{KL}(q(z|x)\|p(z))$$ | L2 | — |
| ESM2 | Meta AI's protein language model family (8M–15B params), transformer + MLM on UniRef | L3 | L4 |
| ESMFold | Single-sequence structure prediction using ESM-2 embeddings + folding trunk | L3 | — |
| Evoformer | Core AlphaFold2 module: 48 blocks refining MSA and pair representations | L4 | L5 |
| FASTA Format | Standard protein sequence file format (header line starting with >) | P2 | P4 |
| Fully Connected Layer | $$h = \sigma(Wx + b)$$; every input connects to every output | P2 | P3, L1 |
| Function Approximation | ML as searching for $$f_\theta$$ that approximates unknown $$f^*$$ | P1 | P2, P3 |
| GAT | Graph Attention Network: learned attention coefficients between nodes | L1 | — |
| GCN | Graph Convolutional Network: degree-normalized neighbor aggregation | L1 | — |
| Generalization | Accurate predictions on unseen data; bias-variance tradeoff | P1 | P3, P4 |
| Gradient Descent | $$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$ | P1 | P3 |
| IGSO(3) | Isotropic Gaussian on SO(3): native noise distribution for rotation manifold | L5 | — |
| Invariant Point Attention | SE(3)-invariant attention in AlphaFold2's structure module; scalar + point Q/K + pair bias | L4 | L5 |
| Inverse Folding | Given target 3D structure, find sequences that fold into it | L6 | — |
| LoRA | Low-Rank Adaptation: freeze pretrained weights, add trainable low-rank matrices BA | L3 | — |
| Loss Function | Single number measuring prediction error $$L(\theta)$$; MSE, BCE, CE | P1 | P2, P3, P4 |
| Masked Language Modeling | Mask ~15% of amino acids and predict from context (BERT-style) | L3 | — |
| Mean Squared Error | $$L_\text{MSE} = \frac{1}{n}\sum(y_i - \hat{y}_i)^2$$ | P1 | P3 |
| Message Passing | $$h_i^{(l+1)} = \phi(h_i^{(l)}, \text{agg}_j(\psi(h_i, h_j, e_{ij})))$$ | L1 | L4, L5, L6 |
| Mini-Batch SGD | Random subset of B examples per gradient update | P3 | P4 |
| Motif Scaffolding | Fix functional motif positions, denoise surrounding scaffold (inpainting) | L5 | L6 |
| MPNN | Fully learnable message and update functions (MLPs); includes edge features | L1 | L6 |
| MSA Representation | $$N_\text{seq} \times L \times c_m$$ tensor encoding evolutionary variation | L4 | — |
| Multi-Head Attention | h independent attention heads in parallel, concatenated and projected | L1 | L3, L4 |
| Multi-Layer Perceptron | Stack of FC layers with activations: $$h_l = \sigma(W_l h_{l-1} + b_l)$$ | P2 | P3, P4, L1 |
| One-Hot Encoding | Binary vector of length 20 for amino acids; protein → (L, 20) matrix | P2 | P4, L1, L3 |
| Overfitting | Training loss ↓ while validation loss ↑ | P3 | P4 |
| Pair Representation | $$L \times L \times c_z$$ tensor encoding pairwise residue relationships | L4 | L5 |
| Positional Encoding | Sinusoidal (fixed) or learned or RoPE; needed because attention is permutation-equivariant | L1 | L3 |
| ProteinMPNN | Inverse folding via k-NN graph + 3-layer MP encoder + autoregressive decoder | L6 | — |
| Reparameterization Trick | $$z = \mu + \sigma \cdot \epsilon$$, $$\epsilon \sim \mathcal{N}(0,I)$$; makes sampling differentiable | L2 | — |
| RFDiffusion | De novo protein backbone generation via SE(3) diffusion; adapts RoseTTAFold | L5 | L6 |
| SE(3) Equivariance | Model outputs transform consistently under rotations and translations | L1 | L4, L5 |
| SE(3) Diffusion | Diffusion combining Gaussian noise (translations) and IGSO(3) noise (rotations) | L5 | — |
| Sequence-Identity Split | Cluster proteins by sequence similarity, split at cluster level | P4 | — |
| Tensor | Multi-dimensional array; protein batch shape: (batch_size, seq_length, features) | P1 | P2, P3, P4 |
| Transformer Block | Self-attention + FFN + residual connections + layer norm | L1 | L3, L4 |
| Triangular Update | Enforces geometric consistency in pair representation via triangle inequality | L4 | L5 |
| VAE | Encoder → $$q(z|x) = \mathcal{N}(\mu, \sigma^2)$$, decoder reconstructs; trained with ELBO | L2 | — |

**Total: 53 concepts tracked**

---

## Notation Table

Key notation symbols used across the series. Scope: G = global (used in 3+ notes), L = local (1–2 notes).

| Symbol | Meaning | Introduced | Scope |
|--------|---------|------------|-------|
| $$\theta$$ | Model parameters | P1 | G |
| $$f_\theta$$ | Parameterized model / function approximator | P1 | G |
| $$L(\theta)$$ | Loss function | P1 | G |
| $$\eta$$ | Learning rate | P1 | G |
| $$\nabla L$$ | Gradient of loss | P1 | G |
| $$\sigma(\cdot)$$ | Activation function (generic) | P2 | G |
| $$W, b$$ | Weight matrix, bias vector | P2 | G |
| $$h^{(l)}$$ | Hidden representation at layer $$l$$ | P2 | G |
| $$\hat{y}$$ | Model prediction | P1 | G |
| $$x_t$$ | Noisy data at diffusion step $$t$$ | L2 | L (L2, L5) |
| $$\epsilon_\theta$$ | Noise prediction network | L2 | L (L2, L5) |
| $$\bar{\alpha}_t$$ | Cumulative noise schedule product | L2 | L (L2, L5) |
| $$Q, K, V$$ | Query, Key, Value matrices (attention) | L1 | G (L1, L3, L4) |
| $$d_k$$ | Key/query dimension (attention scaling) | L1 | L (L1, L4) |
| $$e_{ij}$$ | Edge features between nodes $$i, j$$ | L1 | L (L1, L6) |
| $$z_{ij}$$ | Pair representation entry | L4 | L (L4, L5) |
| $$m_{si}$$ | MSA representation entry | L4 | L (L4) |
| $$T_i$$ | Residue frame (rotation + translation) | L4 | L (L4, L5) |

**No notation conflicts detected.**

---

## Unified Bibliography

### Academic Papers

Deduplicated across all notes. Format: Authors, Title, Venue, Year.

| ID | Citation | Used in |
|----|----------|---------|
| R1 | Goodfellow, I., Bengio, Y., & Courville, A. *Deep Learning* (Ch. 6–8). MIT Press, 2016. | P1 |
| R2 | Paszke, A. et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library. *NeurIPS*, 2019. | P1 |
| R3 | Kingma, D.P. & Ba, J. Adam: A Method for Stochastic Optimization. *ICLR*, 2015. | P3 |
| R4 | Loshchilov, I. & Hutter, F. Decoupled Weight Decay Regularization. *ICLR*, 2019. | P3 |
| R5 | Ho, J., Jain, A., & Abbeel, P. Denoising Diffusion Probabilistic Models. *NeurIPS*, 2020. | L2, L5 |
| R6 | Ho, J. & Salimans, T. Classifier-Free Diffusion Guidance. *NeurIPS Workshop*, 2022. | L5 |
| R7 | Rives, A. et al. Biological Structure and Function Emerge from Scaling Unsupervised Learning to 250M Protein Sequences. *PNAS*, 2021. | L3 |
| R8 | Lin, Z. et al. Evolutionary-Scale Prediction of Atomic-Level Protein Structure with a Language Model. *Science*, 2023. | L3 |
| R9 | Hu, E.J. et al. LoRA: Low-Rank Adaptation of Large Language Models. *ICLR*, 2022. | L3 |
| R10 | Elnaggar, A. et al. ProtTrans: Toward Understanding the Language of Life through Self-Supervised Learning. *IEEE TPAMI*, 2022. | L3 |
| R11 | Meier, J. et al. Language Models Enable Zero-Shot Prediction of the Effects of Mutations on Protein Function. *NeurIPS*, 2021. | L3 |
| R12 | Jumper, J. et al. Highly Accurate Protein Structure Prediction with AlphaFold. *Nature*, 2021. | L4 (implicit) |
| R13 | Baek, M. et al. Accurate Prediction of Protein Structures and Interactions Using a Three-Track Neural Network. *Science*, 2021. | L5 |
| R14 | Watson, J.L. et al. De Novo Design of Protein Structure and Function with RFdiffusion. *Nature*, 2023. | L5 |
| R15 | Dauparas, J. et al. Robust Deep Learning-Based Protein Sequence Design Using ProteinMPNN. *Science*, 2022. | L5, L6 |

### Blog Posts & Tutorials (Further Reading)

| Source | Title | Used in |
|--------|-------|---------|
| Lilian Weng | Attention? Attention! | L1 |
| Jay Alammar | The Illustrated Transformer | L1 |
| Sanchez-Lengeling et al. | A Gentle Introduction to Graph Neural Networks (Distill) | L1 |
| Daigavane et al. | Understanding Convolutions on Graphs (Distill) | L1 |
| Fabian Fuchs | SE(3)-Transformers | L1 |
| Andrew White | Equivariant Neural Networks (dmol.pub) | L1 |
| Lilian Weng | From Autoencoder to Beta-VAE | L2 |
| Lilian Weng | What are Diffusion Models? | L2, L5 |
| Yang Song | Generative Modeling by Estimating Gradients | L2 |
| Calvin Luo | Understanding Diffusion Models: A Unified Perspective | L2 |
| Stephen Malina | Protein Language Models (Parts 1 & 2) | L3 |
| Evolutionary Scale | ESM Cambrian / ESM3 blogs | L3 |
| Elana Simon & Jake Silberg | The Illustrated AlphaFold | L4 |
| Oxford Protein Informatics Group | AlphaFold 2: What's Behind the Structure Prediction Miracle | L4 |
| Fabian Fuchs | AlphaFold 2 & Equivariance | L4 |
| Stephan Heijl | A New Protein Design Era with Protein Diffusion | L5 |
| Baker Lab | A Diffusion Model for Protein Design (blog) | L5 |
| 310.ai | ProteinMPNN: Message Passing on Protein Structures | L6 |

---

## Gap Report

### Missing References Sections

| Note | Issue | Severity |
|------|-------|----------|
| P2 (Features and Architectures) | No references or further reading section | Low — preliminary note, mostly self-contained |
| P4 (Improving Solubility) | No references or further reading section | Low — case study note |

### Cross-Reference Issues Fixed

| Note | Issue | Resolution |
|------|-------|------------|
| P1 | "Note 2" → "Preliminary Note 2" (line 91) | Fixed |
| P1 | "Note 4" → "Preliminary Note 4" (line 100) | Fixed |

### Issues Fixed in Previous Run

| Note | Issue | Resolution |
|------|-------|------------|
| P1 | Duplicate autograd sentence | Removed duplicate |
| P2 | Figure placement (shallow_net/shallow_functions in wrong sections) | Moved to correct sections |
| P2, P3 | `$$...$$` in HTML captions | Converted to `\(...\)` |
| P3 | Citation [6]→[5] for AdamW paper | Fixed numbering |
| P3 | Missing blank line before Prerequisites heading | Added |
| L2 | Nonexistent Exercises row in roadmap table | Removed |
| L3 | 6 `{% cite %}` tags remaining | Replaced with [1]–[5] numbered refs |
| L3 | MSA cross-reference pointed to L1 instead of L4 | Corrected |
| L5 | 6 `{% cite %}` tags remaining | Replaced with [1],[4],[5],[7] numbered refs |
| L6 | Duplicate bioRxiv reference | Clarified as preprint source for figures |

### Potential Improvements (Not Required)

- P2 and P4 could benefit from a short "Further Reading" section pointing to textbooks or tutorials
- L4 (AlphaFold) has no formal `[a], [b]` references section — the original paper (Jumper et al., 2021) is discussed throughout but not listed in a numbered references block
- Notation entities were not individually tracked in the knowledge graph (concepts only); a future pass could add explicit `Notation` entities for all symbols

---

## Per-Note Reports

### Note P1: AI Fundamentals

**Status:** processed
**Changes made (this run):**
- Fixed "Note 2" → "Preliminary Note 2" (line 91)
- Fixed "Note 4" → "Preliminary Note 4" (line 100)

**Changes made (previous run):**
- Fixed duplicate autograd sentence (line 404–405)

**Issues found:** None remaining
**Concepts introduced:** 5 (Function Approximation, Loss Function, Gradient Descent, Tensor, Generalization)

### Note P2: Features and Architectures

**Status:** processed
**Changes made (previous run):**
- Figure placement fixed (shallow_net and shallow_functions moved to correct sections)
- All HTML captions converted from `$$` to `\(...\)`

**Issues found:** No references section (low severity)
**Concepts introduced:** 5 (One-Hot Encoding, Activation Function, MLP, Fully Connected Layer, FASTA Format)

### Note P3: Training

**Status:** processed
**Changes made (previous run):**
- Fixed citation [6]→[5] for AdamW paper
- Fixed missing blank line before Prerequisites heading
- All HTML captions converted from `$$` to `\(...\)`

**Issues found:** None remaining
**Concepts introduced:** 7 (MSE, BCE, CE, Mini-Batch SGD, Adam Optimizer, Overfitting, Bias-Variance Tradeoff)

### Note P4: Improving Solubility

**Status:** processed
**Changes made (previous run):** None needed
**Issues found:** No references section (low severity)
**Concepts introduced:** 3 (Sequence-Identity Split, Early Stopping, Class Imbalance)

### Note L1: Transformers and GNNs

**Status:** processed
**Changes made (previous run):** None needed
**Issues found:** None
**Concepts introduced:** 9 (Attention Mechanism, Multi-Head Attention, Transformer Block, Positional Encoding, Message Passing, GCN, GAT, MPNN, SE(3) Equivariance)

### Note L2: Generative Models

**Status:** processed
**Changes made (previous run):**
- Removed nonexistent Exercises row from roadmap table

**Issues found:** None remaining
**Concepts introduced:** 5 (VAE, ELBO, Diffusion Model, Reparameterization Trick, Classifier-Free Guidance)

### Note L3: Protein Language Models

**Status:** processed
**Changes made (previous run):**
- Replaced 6 `{% cite %}` tags with numbered references [1]–[5]
- Fixed MSA cross-reference: L1 → L4

**Issues found:** None remaining
**Concepts introduced:** 5 (Masked Language Modeling, ESM2, LoRA, ESMFold, Contact Prediction)

### Note L4: AlphaFold

**Status:** processed
**Changes made (previous run):** None needed
**Issues found:** No formal references block (Jumper et al. discussed but not listed)
**Concepts introduced:** 6 (AlphaFold2, Evoformer, Triangular Update, IPA, Pair Representation, MSA Representation)

### Note L5: RFDiffusion

**Status:** processed
**Changes made (previous run):**
- Replaced 6 `{% cite %}` tags with numbered references

**Issues found:** None remaining
**Concepts introduced:** 6 (SE(3) Equivariance [detailed], IGSO(3), SE(3) Diffusion, RFDiffusion, Motif Scaffolding, Classifier-Free Guidance [reuse])

### Note L6: ProteinMPNN

**Status:** processed
**Changes made (previous run):**
- Clarified duplicate bioRxiv reference as preprint source for figures

**Issues found:** None remaining
**Concepts introduced:** 4 (ProteinMPNN, Inverse Folding, Autoregressive Decoding, Design Pipeline)

---

## Statistics

- **Notes processed:** 10/10
- **Concepts tracked:** 53
- **Academic references:** 15 (deduplicated)
- **Blog/tutorial references:** 18
- **Cross-reference issues fixed:** 4 (2 in P1, 1 in L3, 1 in P3)
- **Rendering issues fixed:** 2 (P2, P3 — `$$` in HTML captions)
- **`{% cite %}` tags replaced:** 12 (L3: 6, L5: 6)
- **Notation conflicts:** 0
- **Build errors:** 0
