# CLAUDE.md

## Project Overview

Standalone course site for **Protein & Artificial Intelligence** (Spring 2026, KAIST), built with a lightweight Jekyll setup derived from [al-folio](https://github.com/alshedivat/al-folio).

**Live site:** https://sungsoo-ahn.github.io/protein-ai-s26

## Common Commands

```bash
# Local development (requires Homebrew Ruby, not system Ruby)
/opt/homebrew/opt/ruby/bin/bundle install
/opt/homebrew/opt/ruby/bin/bundle exec jekyll serve  # Opens at http://localhost:4000/protein-ai-s26/

# If port 4000 is already in use:
lsof -ti:4000 | xargs kill -9
```

**Note:** Use Homebrew Ruby (`/opt/homebrew/opt/ruby/bin/`), not macOS system Ruby.

**Important:** Do not kill and restart the Jekyll server on every file edit — this disconnects the user's browser. Leave the server running while editing. Only restart (kill + serve) when the user explicitly asks to open/preview the site.

## Key Files

| File | Purpose |
|------|---------|
| `_config.yml` | Site configuration (baseurl: `/protein-ai-s26`) |
| `_data/course.yml` | Course metadata (title, semester, description) |
| `index.md` | Course landing page |
| `_lectures/` | Lecture notes collection (sorted by `lecture_number`) |
| `assets/img/` | Figures and images |

## Skills

Writing style and rendering rules are managed as skills:

- `/academic-writing` — top-down, rigorous style for papers and teaching notes
- `/jekyll-writing` — MathJax/KaTeX rendering rules for this Jekyll site
- `/download-paper-figures` — incorporating figures from academic papers
- `/refine-lecture-notes` — iterative consistency pass across a series of teaching notes

Folder-specific guidelines (frontmatter, figures, audience) are in `_lectures/CLAUDE.md`.

## Architecture

- **Framework:** Jekyll (lightweight, no al-folio gem dependency)
- **Hosting:** GitHub Pages (auto-deploy on push to main)
- **Content:** Lecture notes in `_lectures/`, ordered by `lecture_number` frontmatter field
