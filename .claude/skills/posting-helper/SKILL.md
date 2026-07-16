---
name: posting-helper
description: >
  Helper for posting paper-analysis articles on this blog (kangnok-choi.github.io, MkDocs).
  Use for requests like "start a new paper post", "create the page", "check my draft
  against the paper", "localize images", "open a PR". Covers exactly four jobs:
  new-post page setup (frontmatter + title + paper header only), fact verification
  against the original paper, image localization, and PR help. Each user works on
  their own branch and publishes via PR.
---

# Posting Helper

This repository is a personal/group research blog built on MkDocs (Material). This
skill helps with four jobs when posting a paper-analysis article. Judge which job the
user needs from where they are in the process.

## Ground rules (always)

- **Branch**: everyone works on their own user branch. Never write/commit docs directly
  on `main`. Create the branch if it doesn't exist (`git switch -c <branch>`), reuse it
  if it does.
- **PR required**: any document change goes to `main` via PR only —
  `git push -u origin <branch>`, then `gh pr create --base main`. Do not use the repo's
  `push.sh` (it pushes directly to the current branch) in this workflow.
- **Facts first**: never rely on memory for paper content — always open the original
  paper to verify.
- **Preserve the user's voice**: when improving a draft, keep the user's sentences,
  tone, and structure; add accuracy and detail only. Never rewrite wholesale.

---

## 1. New post setup

1. **Gather info**: paper (title, authors, year, venue, arXiv link), target location,
   user (= branch). Ask only for what's missing.
   - Target: `docs/<user>/<category>/` for personal posts, `docs/<group>/` for group
     posts (e.g. `docs/multilingual_research_group/`, `docs/world_model_group/`).
2. **Check/create the branch** (per ground rules).
3. **Create the page** at `docs/<target>/<slug>.md` (lowercase kebab-case slug).
   The page must contain ONLY this structure — no template sections, no placeholder
   prompts; the body is the user's to write:

   ```markdown
   ---
   title: Short Title
   ---

   # Short Title

   ##### *Author et al.* Full Paper Title *(Year Venue)*
   ```

4. Tell the user the page is ready for their draft.

## 2. Fact verification

Goal: check the user's draft against the original paper.

- Open the paper:
  - Overview/claims: `https://arxiv.org/abs/<id>`
  - Tables, appendix, equations: `https://ar5iv.labs.arxiv.org/abs/<id>` (includes
    appendix content missing from the abs page — if it redirects, fetch the redirect URL)
- Compare each claim in the draft against the source:
  - **Correct** → keep it, reinforcing with exact numbers/names/quotes.
  - **Wrong/imprecise** → fix it and leave the evidence (exact figure or quote).
  - **Not found in the source** → don't guess; mark with ⚠️ "needs confirmation" and ask
    the user.
- If a figure/table is involved, read the image directly and cross-check the numbers.

## 3. Image localization

External URLs (especially `github.com/user-attachments/assets/...`) can break. Download
them to `docs/<target>/post_images/<slug>/` and replace with:

```markdown
<figure markdown="span"> ![alt](post_images/<slug>/<file>){ width="90%" }</figure>
```

- user-attachments URLs require auth, so plain curl 404s. Use the user's gh login token:
  `curl -L -H "Authorization: token $(gh auth token)" -o <path> "<url>"`
- Verify the downloaded file is a real image (size, magic bytes / preview with Read).

## 4. PR help

1. **Local preview**: `./serve.sh -a 127.0.0.1:8000` (background, live-reload). If the
   user is on a remote (SSH) machine, tell them to open a tunnel from their local
   machine: `ssh -N -L 8000:localhost:8000 <user>@<host>`.
2. After the user confirms: `git add . && git commit -m "docs: add <paper> analysis"`
   → `git push -u origin <branch>` → `gh pr create --base main`.
- Commit messages use the `docs: ...` form.
