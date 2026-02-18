---
name: refine-lecture-notes
description: Iteratively refine lecture notes for consistency, clarity, and notation
  coherence using the Memory MCP knowledge graph. Use when reviewing a series of teaching notes.
---

# Refine Lecture Notes

A repeatable procedure for ensuring a series of lecture notes is internally consistent, properly cross-referenced, and pitched at the right level. Uses the Memory MCP knowledge graph to track concepts and notation across notes.

## Prerequisites

Before starting, load these skills:
- `/academic-writing` — prose style rules
- `/jekyll-writing` — MathJax/KaTeX rendering rules for Jekyll

Also read the folder's `CLAUDE.md` (e.g., `_lectures/CLAUDE.md`) for frontmatter and figure conventions.

## Audience Assumptions

Set per invocation. Default: **undergraduate students with basic math (linear algebra, calculus, probability) but no ML knowledge.** Neural networks, loss functions, gradient descent, generative models — all must be explained before use.

## Knowledge Graph Schema

Use the Memory MCP to track definitions across notes.

### Entity Types

| entityType | Naming Convention | Example | Observations to record |
|---|---|---|---|
| `Concept` | Title Case noun phrase | "Attention Mechanism" | definition, source note/section, prerequisites |
| `Notation` | `symbol: meaning` | `x_t: noisy data at step t` | LaTeX form, meaning, scope, source |
| `LectureNote` | `Note P1: title` / `Note L1: title` | `Note L4: AlphaFold` | file path, status, section count |

### Relations

| Relation | From → To | Meaning |
|---|---|---|
| `depends_on` | Concept → Concept | Concept A requires understanding of Concept B |
| `introduced_in` | Concept/Notation → LectureNote | First defined in this note |
| `used_in` | Concept/Notation → LectureNote | Referenced (but not first defined) in this note |
| `precedes` | LectureNote → LectureNote | Reading order |

### Selection Criteria

Track concepts that are:
- Explicitly defined in the text
- Appear in 2+ notes
- ML or biology terms the target audience wouldn't know

Skip: programming constructs (e.g., "for loop"), single-use shorthand, standard math (e.g., "matrix multiplication").

## Per-Section Procedure (7 Steps)

For each H2 section in a note:

1. **Read** the section carefully.
2. **Extract** all concepts and notation symbols used or introduced.
3. **Check graph** — for each concept/symbol, query the knowledge graph (`search_nodes` or `open_nodes`).
4. **Classify** each into one of these categories:

   | Category | Meaning | Action |
   |---|---|---|
   | `NEW` | Not in graph | Add entity, write definition if missing in text |
   | `CONSISTENT_REUSE` | In graph, same meaning | Add `used_in` relation |
   | `INCONSISTENT_REUSE` | In graph, different meaning | Flag, fix in text |
   | `NOTATION_CONFLICT` | Same symbol, different meaning | Flag, resolve |
   | `UNDEFINED_REFERENCE` | Used without prior definition | Add explanation or cross-reference |
   | `CROSS_REF_ERROR` | Wrong note number/title | Fix |

5. **Edit** the text — add missing definitions, fix inconsistencies, add cross-references.
6. **Update graph** — create entities, add relations.
7. **Record** — note what was changed and why (for the synthesis report).

## Per-Note Procedure

For each lecture note, in dependency order:

1. `read_graph()` to load current knowledge graph state.
2. Read the full note.
3. **Frontmatter check:** verify `lecture_number`, dates, `preliminary` flag, and cross-references in the author note.
4. **Section loop:** process each H2 section using the 7-step procedure above.
5. **Exercises check:** verify exercises reference only concepts covered in this note or prior notes.
6. **References check:** ensure all cited papers are in the references section.
7. **Update note entity** status to "processed" in the knowledge graph.
8. **Build verification:** `jekyll build --incremental` to check for Liquid errors.

## Final Synthesis

After all notes are processed:

1. Dump the full knowledge graph (`read_graph()`).
2. Generate a synthesis report with:
   - **Concept index** — alphabetical, with definition, source note, all notes that use it
   - **Notation table** — all symbols, meanings, sources, conflict flags
   - **Unified bibliography** — deduplicated across all notes, categorized (Required / Supplementary / Original)
   - **Gap report** — unexplained concepts, unresolved conflicts, missing figures

Save to `.claude/reports/<course>-synthesis.md`.

## Quality Criteria

A note is "done" when:
- Every concept is defined before first use (in this note or a prior note)
- Every notation symbol is introduced with its meaning
- Cross-references use correct note numbers and titles
- Exercises reference only covered concepts
- `jekyll build --incremental` succeeds without errors
- Section numbering is consistent (all `## N. Title` format)

## Report Format

Per-note report (appended to synthesis):

```markdown
### Note [P/L]N: Title

**Status:** processed
**Changes made:**
- [list of edits]

**Issues found:**
- [list of issues and resolutions]

**Concepts introduced:** [count]
**Notation symbols introduced:** [count]
```
