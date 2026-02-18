# Teaching Notes Writing Guide

Instructions for writing and editing lecture notes in `_lectures/`.

**Skills:** `/academic-writing` for prose style, `/jekyll-writing` for rendering rules, `/refine-lecture-notes` for cross-note consistency passes.

## Note Frontmatter

Every teaching note must include this frontmatter block:

```yaml
---
layout: post
title: "Your Title Here"
date: YYYY-MM-DD
description: "One-sentence summary."
lecture_number: 2
preliminary: true
toc:
  sidebar: left
related_posts: false
---
```

## Author Note

Notes begin with an author note immediately after the frontmatter:

```html
<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>This is Preliminary Note N for the Protein &amp; Artificial Intelligence course...</em>
</p>
```

## Teaching-Specific Content Rules

- Biology students may not know ML terminology — always explain jargon on first use
- Build from concrete to abstract — introduce a concept with a specific example (e.g., one protein) before generalizing to the matrix/batch form
- Each section should have a clear prerequisite chain: what the student must already know, and what this section enables
- Start with a roadmap table mapping sections to "why it's needed"
- Footnotes work well for alternate terminology or historical context without cluttering the main text

## Prose Style

Teaching notes should read like a direct narrator guiding the student, not like a textbook. The content and structure stay rigorous; the voice changes.

### Kill recaps and scaffolding

- **No recap paragraphs.** Don't open a section with "In Note N, we learned X. Now we..." — the student just read that section. Open with the new idea directly.
- **No table-of-contents prose.** Don't write "This note covers X. First... Second... Third... Finally..." — the roadmap table already does this. Instead, tell the student what they'll be able to *do* by the end.
- **No "previous sections introduced..."** at section boundaries. The reader is already there.

### Direct transitions

- Replace "Let us" / "We will now" / "We shall" with direct statements or imperatives: "Consider...", "Trace the encoding of...", "The gradient computation is straightforward."
- Replace "What makes X special? Two things:" with "X differs from Y in two ways that matter:"
- Replace "We show the model..." / "We formalize..." with active constructions that don't start with "We" as the grammatical subject doing pedagogical actions.

### Lead with tension

- Put the punchline first: "Training performance is a mirage." not "A model that perfectly reproduces answers... is not necessarily useful."
- Lead with *why it matters* or *what goes wrong*, then explain.

### Compress list-heavy sections

- Six identical "**Stage N: Name.** Sentence. Sentence." blocks read like PowerPoint. Rewrite as 2–3 flowing paragraphs with narrative momentum.
- Five "**X** tasks have Y outputs. The model does Z." paragraphs should be compressed into terse sentences — one per formulation.

### Don't over-compress

- Worked examples, code blocks, math derivations, and the Key Takeaways section are fine as-is.
- The concrete-to-abstract build-up (one protein → batch of proteins) should stay intact.
- Footnotes, figures, and the roadmap table are conventions shared with blog posts — keep them.

## Collapsed Code

Some notes set `collapse_code: true` in their frontmatter. This hides code blocks behind a `<details>` toggle so biology-background students can follow the narrative without reading Python. The collapse-code JS (`assets/js/collapse-code.js`) also pulls any `<div class="caption">` immediately after a code fence into the collapsed region.

### When to collapse

Only collapse code in notes where code **illustrates** a domain concept. Don't collapse code in notes that **teach coding itself**.

| Note | Collapse? | Rationale |
|------|-----------|-----------|
| P1 (AI Fundamentals) | No | Teaches tensors, GPU usage, gradients — code IS the curriculum |
| P2 (Protein Data) | Yes | Uses code to illustrate protein formats, features, architectures |
| P3 (Training) | No | Teaches the training loop, optimizers, autograd — code IS the curriculum |
| P4 (Solubility Case Study) | Yes | Uses code to illustrate a protein-science workflow |

### Self-complete prose rule

When `collapse_code: true` is set, **prose must read coherently without expanding any code block.** This means:

1. **Move Python-specific details into captions.** Place a `<div class="caption mt-1">` immediately after the closing ` ``` ` (no blank line). The JS pulls it inside the collapsed `<details>`.

   ```markdown
   ` `` `python
   # code here
   ` `` `
   <div class="caption mt-1">Implementation detail about the code above.</div>
   ```

2. **Rewrite variable/function references as plain language.** If visible prose says `` `DataLoader` handles shuffling ``, change to "A data loader handles shuffling." If it says "This function produces a 24-dimensional vector", change to "The resulting feature vector has 24 dimensions: ..."

3. **Allowed exceptions:**
   - Concept names used as nouns (e.g., `nn.Module` in a section title)
   - References to code in *other* notes where code is visible (e.g., P4 referencing P3's `ProteinDataset`)
   - General PyTorch API references in standalone debugging/reference sections

## Figures

- Store teaching figure images in `assets/img/`
- Embed figures using raw HTML for layout control:

```html
<div class="col-sm-8 mt-3 mb-3 mx-auto">
    <img class="img-fluid rounded" src="{{ '/assets/img/your_figure.png' | relative_url }}" alt="Description">
    <div class="caption mt-1">Caption text here.</div>
</div>
```
