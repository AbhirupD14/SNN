#!/usr/bin/env python3
"""Render the From Spikes to Symbols deck (and the one-page summary) to PDF.

Usage:
    .venv/bin/python slides/build.py
Requires: weasyprint (pip install weasyprint). Pure-Python HTML->PDF, no browser.
"""
import sys
from pathlib import Path
from weasyprint import HTML

HERE = Path(__file__).resolve().parent

TARGETS = [
    ("from_spikes_to_symbols.html", "From_Spikes_to_Symbols.pdf"),
    ("architecture_summary_onepage.html", "Architecture_Summary_OnePage.pdf"),
]

def main():
    for src, out in TARGETS:
        srcp = HERE / src
        if not srcp.exists():
            print(f"skip (missing): {src}")
            continue
        HTML(str(srcp)).write_pdf(str(HERE / out))
        print(f"wrote {out}")

if __name__ == "__main__":
    sys.exit(main())
