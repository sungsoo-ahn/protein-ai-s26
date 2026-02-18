---
name: download-paper-figures
description: Download figures from academic papers and add them to blog posts with proper citations. Use when incorporating paper figures into content.
disable-model-invocation: true
---

# Download and Cite Paper Figures

Workflow for incorporating figures from academic papers into blog posts.

## Step 1: Access the Paper

Use arXiv MCP tools:
```
mcp__arxiv__download_paper(paper_id="XXXX.XXXXX")
mcp__arxiv__read_paper(paper_id="XXXX.XXXXX")
```

For non-arXiv papers, use `WebFetch` on the paper URL.

## Step 2: Check Copyright

**Do NOT reproduce figures directly** from these publishers:
- ACS (American Chemical Society)
- Elsevier
- Wiley
- Springer/Nature
- AAAS (Science)

**ArXiv CC-BY papers**: Can extract figures, but prefer redrawing in our visual style for consistency.

**General rule**: Always redraw figures using approximate data from the paper. This avoids copyright issues and ensures visual consistency with the blog's style.

## Step 3: Redraw in Our Style

- Add figure functions to the blog's figure generation script
- Use the same color palette and styling conventions
- Hardcode approximate data values from the paper (not pixel-exact reproduction)
- Match the key message of the original figure, not its exact appearance

## Step 4: Add Citations

### In Figure Caption

Use this format:
```
"Data adapted from [Author et al., Year]."
```

Example:
```liquid
{% include figure.liquid loading="eager" path="assets/img/blog/figure.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Volcano plot showing catalytic activity vs. adsorption energy. Data adapted from Nørskov et al., 2004." %}
```

### In References Section

Add full citation in the blog post's References section:
```markdown
- Nørskov, J. K., et al. (2004). Origin of the overpotential for oxygen reduction at a fuel-cell cathode. *J. Phys. Chem. B*, 108(46), 17886-17892.
```

## Workflow Summary

1. Read paper via arXiv MCP or WebFetch
2. Identify key figures to include
3. Check publisher copyright (assume restrictive unless CC-BY)
4. Redraw figures in the blog's matplotlib style
5. Add "Data adapted from..." in caption
6. Add full citation in References section
