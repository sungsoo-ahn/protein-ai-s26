---
name: academic-writing
description: Write and edit academic papers and teaching notes with a focus on simplicity, clarity, and top-down structure. Use when drafting, revising, or giving feedback on paper sections — abstracts, introductions, methods, experiments, or related work.
---

# Academic Writing Skill

Write clear, simple, and well-structured academic papers and teaching notes. Every sentence should be easy to read on the first pass.

## Core Philosophy

Good academic writing is **simple, precise, and top-down**. The reader should never have to re-read a sentence to understand it. Every paragraph should flow naturally from what came before.

---

## Rules

### 1. Top-Down Structure

- **General to specific.** Start every section, subsection, and paragraph with the big picture, then narrow down to details.
- **Lead with the main point.** The first sentence of each paragraph states the paragraph's purpose. Supporting details follow.
- **No surprises.** The reader should always know *why* they are reading the current sentence before they read it. Set up context before diving into specifics.

**Bad (bottom-up — details before context):**
> "We compute $\nabla_x \log p_t(x)$ using a neural network $s_\theta(x, t)$ trained with denoising score matching. This allows us to define an ODE whose trajectories transform noise into data. We use this to generate molecules."

**Good (top-down — purpose first, then details):**
> "We generate molecules by learning a continuous transformation from noise to data. Specifically, we define an ODE whose vector field is given by a learned score function $s_\theta(x, t) \approx \nabla_x \log p_t(x)$. We train $s_\theta$ with denoising score matching."

### 2. Terminology Discipline

- **Define before you use.** Every technical term, abbreviation, or notation must be introduced and explained *before* it appears in the text. Never assume the reader already knows what a term means.
- **One term, one meaning.** Once you pick a word for a concept, use that same word consistently throughout the paper. Do not alternate between synonyms for the same concept.
- **One meaning, one term.** Conversely, do not reuse the same word for different concepts in different parts of the paper.
- **Minimize jargon.** If a simpler word works, use it. Prefer "use" over "utilize," "show" over "demonstrate," "because" over "due to the fact that."

**Bad (synonym swapping — confuses the reader):**
> "We train a score network $s_\theta$ to estimate the score function. ... The denoiser predicts the clean data from the noisy input. ... We parameterize the noise predictor as a U-Net."

(Are "score network," "denoiser," and "noise predictor" the same thing or different things? The reader cannot tell.)

**Good (one term, used consistently):**
> "We train a score network $s_\theta$ to estimate the score function $\nabla_x \log p_t(x)$. ... The score network predicts the direction toward clean data from a noisy input. ... We parameterize the score network as a U-Net."

### 3. Sentence-Level Clarity

- **Short sentences.** Break long sentences into two. If a sentence has more than one main idea, split it.
- **Active voice by default.** Prefer "We train the model" over "The model is trained." Use passive voice only when the actor is irrelevant or unknown.
- **Subject-verb-object.** Keep the subject and verb close together. Avoid long phrases between the subject and the verb.
- **No dangling modifiers.** Make sure every modifier clearly refers to the right noun.
- **Avoid filler.** Remove words that add no meaning: "it is worth noting that," "it should be mentioned that," "interestingly," "importantly," "note that." Just state the fact directly.
- **Avoid vague references.** "This" or "it" at the start of a sentence should clearly refer to something specific. If ambiguous, name the referent explicitly.

**Bad (long, filler-heavy, vague "this"):**
> "It is worth noting that the model, which was trained on a dataset of 10K molecules with a learning rate of 1e-4 using the Adam optimizer, achieves state-of-the-art results. This is important because it shows the effectiveness of our approach."

**Good (short, direct, specific reference):**
> "The model achieves state-of-the-art results when trained on 10K molecules. This strong performance suggests that our flow matching objective learns a better noise-to-data mapping than the baseline diffusion loss."

### 4. Paragraph Structure

- **One idea per paragraph.** If a paragraph covers two ideas, split it into two paragraphs.
- **Topic sentence first.** The first sentence tells the reader what the paragraph is about.
- **Logical flow between paragraphs.** Each paragraph should connect naturally to the next. Use brief transitions when needed, but don't over-transition.

### 5. Explain with Concrete Examples

- **Ground abstract claims with examples.** Whenever you introduce a concept, method, or design choice, follow it with a concrete example that makes it tangible.
- **Examples after the general statement.** The general claim comes first (top-down), then the example reinforces it.
- **Prefer small, self-contained examples.** A good example is short, specific, and doesn't require extra background to understand.
- **Use examples to clarify, not to replace explanation.** The reader should understand the general idea from the text; the example makes it concrete and memorable.

**Bad:**
> "Our method can handle diverse graph structures."

**Good:**
> "Our method can handle diverse graph structures. For example, given a molecular graph with aromatic rings and branching chains, our method generates valid bond configurations without additional preprocessing."

**Bad:**
> "We use a masking strategy to avoid information leakage."

**Good:**
> "We use a masking strategy to avoid information leakage. Specifically, when predicting the bond type for atom pair $(i, j)$, we mask all edges connected to $i$ and $j$ so the model cannot trivially copy the answer from its input."

### 6. Notation and Math

- **Introduce notation explicitly.** Before using a symbol, write a sentence defining it: "Let $x \in \mathbb{R}^d$ denote the input feature vector."
- **Be consistent.** Use the same symbol for the same quantity throughout the paper. Never redefine a symbol.
- **Minimize notation.** Only introduce symbols you actually need. If a quantity appears only once, describe it in words.
- Notation convention: vectors as lowercase bold, matrices as uppercase bold.
- Use descriptive subscripts: $$\theta_{\text{enc}}$$ not $$\theta_1$$.
- Follow every equation with an explanation of its terms.

### 7. Figures and Tables

- **Self-contained captions.** A reader should understand a figure or table from its caption alone, without reading the main text.
- **Reference before appearance.** Always refer to a figure or table in the text *before* it appears on the page.

### 8. Related Work and Citations

- **Describe what a paper does, not just that it exists.** Instead of "Smith et al. [1] proposed a method for X," write "Smith et al. [1] generate molecules by first predicting bonds, then assembling atoms." Give the reader a concrete understanding.
- **Group related citations.** Organize related work by theme, not by chronological order or by paper.

### 9. Abstract and Introduction

- **Abstract: Problem → Approach → Result.** State the problem in 1-2 sentences, describe your approach in 2-3 sentences, and summarize key results in 1-2 sentences.
- **Introduction: funnel structure.** Start broad (why the problem matters), narrow to the specific gap, state your contribution, and preview results.
- **Contributions should be concrete.** Not "We propose a novel method." Instead: "We propose X, which does Y, achieving Z on benchmark W."

---

## Common Mistakes to Watch For

| Mistake | Fix |
|---|---|
| Using a term before defining it | Move the definition earlier or add one |
| Switching between synonyms for the same concept | Pick one term and use it everywhere |
| Long sentence with multiple clauses | Split into shorter sentences |
| Starting a paragraph with details instead of context | Rewrite so the first sentence gives the big picture |
| Vague "this" or "it" | Replace with the specific noun |
| "We propose a novel method for X" (no detail) | "We propose [specific method] that [specific mechanism], achieving [specific result]" |
| Passive voice hiding the actor | Rewrite in active voice |
| Orphan notation (symbol used once) | Replace with words |
| Figure/table not referenced in text | Add a reference sentence before the figure/table |

---

## Editing Checklist

When reviewing or editing a draft, check the following in order:

1. **Structure**: Does each section/paragraph follow the top-down pattern? Is the main point stated first?
2. **Terminology**: Is every term defined before first use? Is terminology consistent throughout?
3. **Sentences**: Are sentences short and clear? Is the subject-verb distance small?
4. **Filler**: Can any words or phrases be removed without losing meaning?
5. **Notation**: Is every symbol defined? Are symbols used consistently?
6. **Flow**: Does each paragraph connect logically to the next?
7. **Figures/Tables**: Are captions self-contained? Are all figures/tables referenced in the text?

---

## How to Apply This Skill

When asked to **write** a section: Follow all rules above. Use top-down structure. Define terms before using them. Keep sentences short.

When asked to **edit** or **review** a draft: Identify violations of the rules above. Suggest concrete fixes. Prioritize issues in this order: (1) structural problems, (2) terminology issues, (3) sentence-level clarity, (4) minor polish.

When asked to **give feedback**: Point out the most important issues first. Give specific examples of the problem and the fix. Don't just say "this is unclear" — explain *why* and show a rewrite.

