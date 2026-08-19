# FeatureLens design system

FeatureLens is a research instrument. Its interface should help someone **operate** an experiment and **read** evidence; it is not a landing page, growth dashboard, or software-company product surface.

## Visual direction

The visual reference is a compact scientific workbench: editorial headings, neutral controls, dense but readable tables, restrained plots, and very little decorative chrome. The interface should feel designed by someone who expects to inspect numbers for a while.

### Typography

- Display / section headings: `Georgia`, `Cambria`, `Times New Roman`, serif fallback.
- Body, controls, table cells: `Segoe UI`, `Helvetica Neue`, `Arial`, sans-serif fallback.
- Monospace is reserved for token chips and literal identifiers where fixed-width text helps scanning.
- Hierarchy comes from scale, weight, and spacing—not from badges, italics, all-caps kickers, or a different treatment for every subsection.
- Long explanatory prose should stay near 70–80 characters per line even though the analytical canvas is wide.

### Color roles

- Accent: muted teal `#6F8984`.
- Secondary series: umber `#8A735D`, red-grey `#8C6A67`, plum `#786F82`, stone `#82827E`, blue-grey `#687982`.
- Neutrals come from Gradio's grey theme tokens so light and dark appearance remain usable.
- Saturated default categorical chart colors should be avoided when a fixed series set is known.
- No gradients, neon accents, glows, glass effects, or decorative beige/cream surfaces.

### Geometry and surfaces

- Radius: approximately 2 px. Controls may be slightly rounded but should never look pill-shaped.
- No nested-card hierarchy. Groups exist for layout/state only and are visually flat.
- Avoid drop shadows. The only shadow permitted is the temporary in-place focus mode for dense tables/plots.
- Use rules, alignment, and whitespace instead of containers-within-containers.
- Do not add left-border "accent cards" or side-tab callouts.

### Spacing

- Spacing is intentionally uneven by semantic role: related controls are tight; a new experiment has more separation.
- Section headings should sit close to the controls/results they introduce.
- Table titles use the otherwise-empty table toolbar band rather than consuming another row of vertical space.
- Do not introduce uniform 16/24/32 px spacing everywhere simply because it is convenient.

## Components

### Header

One title and one factual subtitle. Do not show version badges, status chips, marketing claims, or repository trivia in the live header.

### Tabs

Flat text navigation with one active underline. No tab pills or raised nav cards.

### Buttons

- Primary experiment action: compact muted-teal button.
- Utility action (`Copy TSV`, candidate hand-off): quiet secondary button.
- Button text should describe the action (`Rank candidates`, `Run intervention`) rather than generic calls to action (`Continue`, `Get started`).
- Do not make every button full-width on desktop.

### Forms

Labels are literal and stable because validation instructions refer to them. Helper text is only included when the field's semantics are not obvious from the label.

### Tables

Tables are primary research objects, not decorative cards. Use explicit headings, normal-weight data, tabular numerals, bounded height, horizontal scrolling for genuinely wide schemas, and a quiet `Copy TSV` utility.

### Plots

Plots use a small, restrained palette and preserve their aspect ratio in focus mode. A plot should not expand to fill every available screen dimension. Export filenames should describe the figure rather than defaulting to `chart.png`.

### Results copy

Measured values first. Caveats should be local and short; long methodological qualification belongs in **Method** or the offline report. Avoid paragraphs that restate what the immediately adjacent table already shows.

## Information architecture

- **Guide** — how the evidence ladder works.
- **Workbench** — inspect one residual location and intervene on one feature.
- **Feature sets** — joint-feature interventions and geometry.
- **Features** — discovery, triage, controlled comparison, target profiles, and feature diagnostics.
- **Paraphrases** — robustness to rewording.
- **Layers** — early/middle/late SAE trajectory.
- **Study** — artifact-backed offline results only.
- **Method** — formulas, control discipline, and interpretation boundaries.

The global context line is the single source of truth for the current Workbench prompt/layer/token. Do not repeat separate context cards in each tab.

## Anti-patterns

Do not add any of the following without a specific functional reason:

- card grids for simple prose;
- cards nested inside cards;
- status-chip or badge collections;
- gradients, glassmorphism, glow, or oversized decorative backgrounds;
- AI/SaaS language such as "unlock", "supercharge", "powerful", or "all-in-one";
- eyebrow/kicker labels above ordinary headings;
- tiny numbered or Roman-numeral section labels;
- every section centered or symmetrically weighted;
- the same font for headings, prose, code, and tabular data;
- full-width primary buttons by default;
- explanatory text that narrates a bug fix or implementation history;
- visible release/version marketing in the public application.

## Review checklist

Before changing the public UI, check:

1. Does the change improve operating or reading the research tool?
2. Is it using an existing type, color, spacing, and button role?
3. Can a border/card/help paragraph be removed without losing meaning?
4. Does the result remain readable in dark and light appearance?
5. Did we preserve exact field/button terminology used by validation docs?
6. Did we avoid spending ZeroGPU quota merely to test an unchanged inference path?
