#!/usr/bin/env python3
"""Render the XIIV benchmark comparison chart from MEASURED results.

Input: a results JSON (schema below). Output: a dark-themed SVG
(plus PNG if matplotlib happens to be installed).

Integrity rules (enforced by construction, not by trust):
  - Panel A plots ONLY rows whose provenance is "measured-local"
    (XIIV, run here) or "measured-same-hardware" (open-weight
    competitors run through eval/adapters on this machine), on the
    shared 0-10 harness categories. A null score renders as "n/m"
    (not measured) — never as a zero bar, never as a win.
  - Published frontier numbers NEVER share Panel A's axes (they are
    different benchmarks). They appear in Panel B as a labelled
    reference table: name, benchmark, score, source, date.
  - Every row must carry a "detail" provenance string (checkpoint
    SHA / source citation). A row without one aborts the render.

Schema:
{
  "title": "XIIV benchmark comparison",
  "date": "2026-11-18",
  "categories": ["factual_explanation", ...],        // 0-10 each
  "rows": [
    {"name": "XIIV-1B step19600", "provenance": "measured-local",
     "detail": "this machine, harness spec v1, ckpt 278d7502",
     "scores": {"factual_explanation": 8, ...}},     // null allowed
    {"name": "Qwen3-0.6B", "provenance": "measured-same-hardware",
     "detail": "hf_causal_adapter, same machine, 2026-11-18",
     "scores": {...}}
  ],
  "published_reference": [
    {"name": "GPT-6 Astra", "benchmark": "GPQA Diamond",
     "score": "94.1", "source": "vendor system card", "date": "2026-.."}
  ]
}

Usage:
  python3 make_benchmark_chart.py results.json --out chart.svg
"""

import argparse
import html
import json
import sys

W, ROW_H, PANEL_W = 1480, 0, 0  # computed below
BG = "#0b0e13"
PANEL = "#12161d"
GRID = "#232a35"
TEXT = "#e8edf4"
MUTED = "#8b96a5"
COLORS = {
    "measured-local": "#37c8f5",
    "measured-same-hardware": "#57d977",
}
SHORT = {
    "arithmetic_reasoning": "arithmetic",
    "code_short": "code",
    "coherence_no_glitch": "coherence",
    "constraint_compliance": "constraint",
    "factual_explanation": "factual",
    "general_knowledge_qa": "gen. knowledge",
    "instruction_following": "instruction",
    "latency_reliability": "latency",
    "proof_carrying": "proof",
    "summarisation": "summarisation",
}


def esc(s):
    return html.escape(str(s), quote=True)


def render(data, out_svg):
    rows = data["rows"]
    cats = data["categories"]
    for r in rows:
        if r["provenance"] not in COLORS:
            sys.exit(f"row {r['name']!r}: provenance "
                     f"{r['provenance']!r} cannot be plotted in "
                     f"Panel A (published numbers go in "
                     f"published_reference)")
        if not r.get("detail"):
            sys.exit(f"row {r['name']!r}: missing provenance detail "
                     f"(checkpoint SHA / source) — refusing to render")
    n_series = len(rows)
    group_w = 118
    bar_w = min(34, (group_w - 18) // max(1, n_series))
    chart_w = 150 + len(cats) * group_w + 30
    chart_h = 430
    refs = data.get("published_reference", [])
    ref_h = 0 if not refs else 56 + 30 * len(refs)
    total_h = chart_h + ref_h + 92

    s = []
    s.append(f'<svg xmlns="http://www.w3.org/2000/svg" '
             f'width="{chart_w}" height="{total_h}" '
             f'viewBox="0 0 {chart_w} {total_h}">')
    s.append(f'<rect width="100%" height="100%" fill="{BG}"/>')
    s.append(f'<text x="40" y="46" fill="{TEXT}" font-family="Inter,'
             f'Arial" font-size="25" font-weight="700">'
             f'{esc(data["title"])}</text>')
    s.append(f'<text x="40" y="72" fill="{MUTED}" font-family="Inter,'
             f'Arial" font-size="13">{esc(data["date"])} · Panel A: '
             f'measured on this machine, XIIV 100-prompt harness '
             f'(0–10 per category, gate = 9 in EVERY category)</text>')

    # legend
    lx = 40
    for prov, col in COLORS.items():
        s.append(f'<rect x="{lx}" y="88" width="14" height="14" '
                 f'fill="{col}"/>')
        s.append(f'<text x="{lx + 20}" y="100" fill="{TEXT}" '
                 f'font-family="Inter,Arial" font-size="12.5">'
                 f'{esc(prov)}</text>')
        lx += 20 + 8.2 * len(prov) + 26

    top, bottom = 128, chart_h - 58
    # gridlines 0..10
    for v in range(0, 11, 2):
        y = bottom - (bottom - top) * v / 10
        s.append(f'<line x1="64" y1="{y:.0f}" x2="{chart_w - 24}" '
                 f'y2="{y:.0f}" stroke="{GRID}"/>')
        s.append(f'<text x="52" y="{y + 4:.0f}" fill="{MUTED}" '
                 f'font-family="Inter,Arial" font-size="11.5" '
                 f'text-anchor="end">{v}</text>')
    # gate line at 9
    gy = bottom - (bottom - top) * 9 / 10
    s.append(f'<line x1="64" y1="{gy:.0f}" x2="{chart_w - 24}" '
             f'y2="{gy:.0f}" stroke="#f5a637" stroke-dasharray="6 4"/>')
    s.append(f'<text x="{chart_w - 28}" y="{gy - 6:.0f}" '
             f'fill="#f5a637" font-family="Inter,Arial" font-size="11.5" '
             f'text-anchor="end">gate 9/10</text>')

    for ci, cat in enumerate(cats):
        gx = 84 + ci * group_w
        for si, r in enumerate(rows):
            v = r["scores"].get(cat)
            bx = gx + si * (bar_w + 3)
            if v is None:
                s.append(f'<text x="{bx + bar_w / 2:.0f}" '
                         f'y="{bottom - 6}" fill="{MUTED}" '
                         f'font-family="Inter,Arial" font-size="10.5" '
                         f'text-anchor="middle">n/m</text>')
                continue
            bh = (bottom - top) * v / 10
            s.append(f'<rect x="{bx}" y="{bottom - bh:.0f}" '
                     f'width="{bar_w}" height="{bh:.0f}" '
                     f'fill="{COLORS[r["provenance"]]}"/>')
            s.append(f'<text x="{bx + bar_w / 2:.0f}" '
                     f'y="{bottom - bh - 5:.0f}" fill="{TEXT}" '
                     f'font-family="Inter,Arial" font-size="11.5" '
                     f'text-anchor="middle">{v:g}</text>')
        s.append(f'<text x="{gx + group_w / 2 - 10:.0f}" '
                 f'y="{bottom + 20}" fill="{TEXT}" '
                 f'font-family="Inter,Arial" font-size="12" '
                 f'text-anchor="middle">{esc(SHORT.get(cat, cat))}'
                 f'</text>')

    # row provenance details under Panel A
    dy = chart_h - 22
    for r in rows:
        s.append(f'<text x="40" y="{dy}" fill="{MUTED}" '
                 f'font-family="Inter,Arial" font-size="12">'
                 f'<tspan fill="{COLORS[r["provenance"]]}">■</tspan> '
                 f'{esc(r["name"])} — {esc(r["detail"])}</text>')
        dy += 18

    if refs:
        ry = chart_h + 44
        s.append(f'<text x="40" y="{ry}" fill="{TEXT}" '
                 f'font-family="Inter,Arial" font-size="16" '
                 f'font-weight="700">Panel B — published frontier '
                 f'reference (different benchmarks; NOT directly '
                 f'comparable to Panel A)</text>')
        ry += 12
        for ref in refs:
            ry += 30
            s.append(f'<rect x="40" y="{ry - 17}" '
                     f'width="{chart_w - 80}" height="24" '
                     f'fill="{PANEL}"/>')
            s.append(f'<text x="54" y="{ry}" fill="{TEXT}" '
                     f'font-family="Inter,Arial" font-size="13">'
                     f'{esc(ref["name"])} · {esc(ref["benchmark"])}: '
                     f'<tspan font-weight="700">{esc(ref["score"])}'
                     f'</tspan>  '
                     f'<tspan fill="{MUTED}">({esc(ref["source"])}, '
                     f'{esc(ref["date"])})</tspan></text>')
    s.append(f'<text x="40" y="{total_h - 14}" fill="{MUTED}" '
             f'font-family="Inter,Arial" font-size="11.5">Generated by '
             f'eval/make_benchmark_chart.py from measured results '
             f'only. n/m = not measured. No value in this chart is '
             f'estimated or projected.</text>')
    s.append("</svg>")
    svg = "\n".join(s)
    with open(out_svg, "w") as f:
        f.write(svg)
    print(f"wrote {out_svg} ({len(svg)} bytes)")
    png = out_svg.rsplit(".", 1)[0] + ".png"
    try:
        import cairosvg  # noqa: F401
        cairosvg.svg2png(bytestring=svg.encode(), write_to=png)
        print(f"wrote {png}")
    except ImportError:
        try:
            import matplotlib  # noqa: F401
            print("note: matplotlib present but SVG->PNG needs "
                  "cairosvg; SVG is the deliverable")
        except ImportError:
            print("note: PNG skipped (cairosvg not installed); "
                  "SVG is the deliverable")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--out", default="benchmark_chart.svg")
    args = ap.parse_args()
    data = json.load(open(args.results))
    render(data, args.out)


if __name__ == "__main__":
    main()
