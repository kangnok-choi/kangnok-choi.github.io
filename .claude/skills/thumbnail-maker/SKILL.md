---
name: thumbnail-maker
description: >
  Generates SVG covers (thumbnails) matching this blog's (kangnok-choi.github.io, MkDocs)
  visual identity for posts, papers, and categories. Use for requests like "make a
  thumbnail", "cover image", "cover.svg", "card image", "profile background" — and also,
  even when SVG isn't explicitly mentioned, whenever a new paper-analysis post or a new
  category/group page needs an image for its card.
---

# Blog SVG cover (thumbnail) generation

This repository's card UIs (post-card, cat-banner, profile-card) each use
`cover_images/*.svg` as backgrounds. This skill covers how to build an SVG that
expresses a post's core concept as a visual metaphor, in the site's style.

## Cover types and specs

| Use | File location | viewBox | Display (extra.css) |
|---|---|---|---|
| **Post cover** (post-card) | `docs/<target>/cover_images/<slug>.svg` | `0 0 600 360` (5:3) | Card is also `aspect-ratio: 5/3` → almost no cropping |
| **Category banner** (cat-banner) | `docs/<user>/<category>/cover_images/cover.svg` | `0 0 1200 240` | Fixed 195px height, `background-position: center right` → **left side gets cropped**. Title text is overlaid on the left |
| **Profile background** (profile-card) | `docs/<user|group>/cover_images/profile.svg` | `0 0 480 480` | Sits behind text at `opacity: 0.26`; the card is portrait (5/6) so left/right get cropped |

- Name post covers after the slug (e.g. `induction-heads.svg`). The section's
  `cover.svg` is the fallback when no dedicated cover exists.
- All covers use `preserveAspectRatio="xMidYMid slice"` — edges may be cropped
  depending on card size, so keep **key shapes inside the safe area**:
  - Post cover: ~30px margin from the edges
  - Banner: artwork in the **right half (x ≥ 600)**; left side background/pattern only
    (the title covers it)
  - Profile: uniform pattern/texture that survives left/right cropping; no dark shapes
    that hurt text readability

## Visual identity (consistent with existing covers)

Open 1–2 existing covers (`docs/*/cover_images/*.svg`) and match their tone. Shared
grammar:

- **Background**: light blue gradient. `#f3f6fc → #dbe6f7` or `#fafbfd → #e8eef9`.
  Covers stay light even in dark mode — that's the existing convention (the card
  contrast actually looks better).
- **Single accent color**: `#1e4ea3` (the site `--accent`). Instead of adding colors,
  build hierarchy with `fill-opacity` (0.05–0.85). Only one emphasis is dark; the rest
  stay faint.
- **Texture**: subtle dot pattern (`r=1`, `fill-opacity="0.08"`, 14–16px spacing) —
  optional.
- **Text**: only when truly needed, `font-family="JetBrains Mono, monospace"`, at
  token/label level only. **Never put the post title on the cover** — the card body
  already shows the title.
- **Avoid ID collisions**: prefix gradient/pattern ids with the slug (e.g.
  `id="qurating-bg"`).

## Design process

1. **Read the post and pick ONE core mechanism.** Not the paper's topic — *the single
   idea that makes the best picture*. A cover is a diagram, not an illustration.
   - e.g. induction heads → `A B C D A ?` token sequence with a back-reference arrow
   - e.g. attention → lower-triangular attention-matrix tiles
   - e.g. data selection (QuRating) → a grid of documents with only a few highlighted
2. **Express it with the shape grammar**: rect tiles, circle nodes + curved edges
   (`Q` Bézier), monospace tokens, dashed lines (`stroke-dasharray`) cover most needs.
   Arrowheads are small `polygon`s.
3. **Validate**:
   - XML parse: `python3 -c "import xml.dom.minidom,sys; xml.dom.minidom.parse(sys.argv[1])" <file>`
   - Render check: on the page if `./serve.sh` is running, otherwise open the SVG
     directly in a browser. If remote (SSH), tell the user to open a tunnel
     (`ssh -N -L 8000:localhost:8000 ...`).
4. **Wire it to the card**: replace/add the cover path in the card in the relevant
   `index.md`.
   ```html
   <div class="post-card__cover" style="background-image: url('cover_images/<slug>.svg')"></div>
   ```
   (This skill only covers SVG creation and path wiring — not registering the post
   itself.)

## Post cover skeleton example

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 360" preserveAspectRatio="xMidYMid slice">
  <defs>
    <linearGradient id="myslug-bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#fafbfd"/>
      <stop offset="1" stop-color="#e8eef9"/>
    </linearGradient>
    <pattern id="myslug-dots" width="14" height="14" patternUnits="userSpaceOnUse">
      <circle cx="1" cy="1" r="1" fill="#1e4ea3" fill-opacity="0.08"/>
    </pattern>
  </defs>
  <rect width="600" height="360" fill="url(#myslug-bg)"/>
  <rect width="600" height="360" fill="url(#myslug-dots)"/>

  <!-- core metaphor: inside the central safe area (30px margins) -->
  <g transform="translate(60, 110)" fill="#1e4ea3">
    <!-- ... -->
  </g>
</svg>
```

## Cautions

- Pure SVG only — no external image/font links, `<script>`, or CSS `@import`
  (loaded as background-image on GitHub Pages, so external references break rendering).
- When making multiple drafts, save them under different names in the same folder and
  let the user choose.
- Always ask the user before overwriting an existing cover.
