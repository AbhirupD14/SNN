# From Spikes to Symbols — theory slideshow

A polished 16:9 deck presenting the *theoretical architecture* developed in this
repository: growing intelligence from local spiking circuits that learn continuously,
form explicit symbols, compose them hierarchically, and know the boundaries of their
knowledge. It is a vision/theory deck, not an implementation-status report.

## Deliverables

| File | What it is |
|---|---|
| `from_spikes_to_symbols.html` | **Editable slideshow source** — self-contained, dark-theme, inline-SVG diagrams, 20 pages at 1280×720. |
| `From_Spikes_to_Symbols.pdf` | **Final 16:9 PDF** (16 core slides + a "vs. symbolic/neuro-symbolic AI" contrast, predicted-vs-measured note, closing message, and sources appendix). |
| `architecture_summary_onepage.html` / `Architecture_Summary_OnePage.pdf` | **One-page architecture summary** (built today vs. designed-to-enable vs. expected properties). |
| `SPEAKER_NOTES.md` | Full per-slide speaker notes (the detail the slides keep off-screen). |
| `SOURCES.md` | Sources appendix — repository grounding + scientific motifs. |
| `build.py` | Renders the HTML sources to PDF. |

## Rebuilding the PDFs

The sources are plain HTML; PDFs are rendered with [WeasyPrint](https://weasyprint.org/)
(pure-Python HTML→PDF, no browser needed):

```bash
.venv/bin/pip install weasyprint      # one-time
.venv/bin/python slides/build.py       # writes both PDFs next to the sources
```

To edit: change the HTML (each `<section class="slide">` is one page) and re-run
`build.py`. Diagrams are inline SVG — note that WeasyPrint renders radial-gradient fills
unreliably inside transformed groups, so neuron fills use solid colors with a highlight
circle rather than gradients.

## Design notes

- Dark scientific theme; electric blue = excitatory, amber = inhibitory, gold = symbols,
  green = known/continuous-learning, grey = unknown.
- Diagram-driven with ~20–40 words per slide; explanations live in `SPEAKER_NOTES.md`.
- One deliberate caveat slide (17) separates architectural predictions from measured
  results; no numeric hardware benchmarks are invented.
