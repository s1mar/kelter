"""Draw docs/benchmark.svg and docs/benchmark-dark.svg from the results files.

Six small panels: five metrics where openverdict's default variant (Q1) is compared
with four other systems run through the same evaluation code on the same
held-out sets, and the JevBench public tiers scored by JevBench's own harness.
Every number is read from results/, never typed here.

    python scripts/make_benchmark_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "results" / "eval"
JEVBENCH = ROOT / "results" / "jevbench" / "Q1chunk-20261104T162222Z" / "openverdict_summary.json"

SYSTEMS = [  # (label, results tag); the first row is this repository's default variant
    ("openverdict", "Q1"),
    ("Qwen2.5-1.5B", "qwen"),
    ("Kev 0.8B", "kev"),
    ("Laya 0.4", "laya"),
    ("TF-IDF logistic", "tfidf"),
]

PANELS = [  # (title, subtitle, getter, higher_is_better)
    ("Accuracy", "share of held-out questions answered correctly", lambda d: d["heldout_accuracy"], True),
    ("Calibration error", "expected calibration error, 15 bins", lambda d: d["ece_15"], False),
    ("Order sensitivity", "answers that change when options are reordered", lambda d: d["flip_rate_canonical"], False),
    ("Injection success", "answers moved by a forged instruction in the text", lambda d: d["forge_asr"]["asr"], False),
    ("Null detection", "unanswerable questions caught at 5% false positives", lambda d: d["null_recall_at_5fpr_corrected"]["recall"], True),
]

# "others" is a deliberate neutral for the systems that are not the subject (one entity vs the rest);
# every bar carries its name and value, which is the relief the low contrast calls for.
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
              "axis": "#c3c2b7", "ours": "#2a78d6", "others": "#b5b3ab"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
             "axis": "#383835", "ours": "#3987e5", "others": "#76746e"},
}

W, H = 960, 576
COLS, ROWS = 3, 2
PANEL_W, PANEL_H = 300, 200
LEFT, TOP = 24, 112
LABEL_W = 112           # room for the system names
BAR_H = 14              # under the 24px cap
ROW_H = 26
FONT = 'font-family="system-ui, -apple-system, &quot;Segoe UI&quot;, sans-serif"'


def load():
    rows = []
    for label, tag in SYSTEMS:
        rows.append((label, json.loads((EVAL / f"{tag}.json").read_text(encoding="utf-8"))))
    jb = json.loads(JEVBENCH.read_text(encoding="utf-8"))["tiers"]
    tiers = [(name, jb[t]["summary"]["accuracy"], jb[t]["summary"]["n_valid"]) for name, t in
             (("Easy", "easy"), ("Original", "original"), ("Hard", "hard"))]
    n_heldout = sum(v["n"] for v in rows[0][1]["heldout_accuracy_by_source"].values())
    return rows, tiers, n_heldout


def bar_path(x, y, w, h, r=4):
    """Square at the baseline (left), 4px rounded data-end (right)."""
    if w <= 0:
        return ""
    r = min(r, w, h / 2)
    return (f"M{x:.1f},{y:.1f} H{x + w - r:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} "
            f"V{y + h - r:.1f} Q{x + w:.1f},{y + h:.1f} {x + w - r:.1f},{y + h:.1f} H{x:.1f} Z")


def text(x, y, s, size, fill, anchor="start", weight="normal"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}" {FONT}>{s}</text>')


def panel(ox, oy, title, subtitle, items, t, highlight_first):
    """items: list of (label, value); values in [0, 1]."""
    out = [text(ox, oy + 14, title, 14, t["ink"], weight="600"), text(ox, oy + 30, subtitle, 11, t["ink2"])]
    plot_x = ox + LABEL_W
    plot_w = PANEL_W - LABEL_W - 44
    base_y = oy + 44
    for i, (label, v) in enumerate(items):
        y = base_y + i * ROW_H
        color = t["ours"] if (highlight_first and i == 0) or not highlight_first else t["others"]
        out.append(text(plot_x - 8, y + BAR_H - 3, label, 12, t["ink"] if (highlight_first and i == 0) else t["ink2"], anchor="end",
                        weight="600" if (highlight_first and i == 0) else "normal"))
        w = plot_w * max(0.0, min(1.0, v))
        if w > 0:
            out.append(f'<path d="{bar_path(plot_x, y, w, BAR_H)}" fill="{color}"/>')
        out.append(text(plot_x + w + 6, y + BAR_H - 3, f"{v:.2f}", 12, t["ink2"]))
    # baseline
    y0, y1 = base_y - 4, base_y + len(items) * ROW_H - (ROW_H - BAR_H) + 4
    out.append(f'<line x1="{plot_x}" y1="{y0}" x2="{plot_x}" y2="{y1}" stroke="{t["axis"]}" stroke-width="1"/>')
    return "\n".join(out)


def figure(theme: str) -> str:
    t = THEMES[theme]
    rows, tiers, n_heldout = load()
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
             f'aria-label="openverdict compared with four other systems on five measures, and its JevBench public-tier accuracy">',
             f'<rect width="{W}" height="{H}" fill="{t["surface"]}"/>',
             text(LEFT, 36, "openverdict against four other systems, same questions, same scoring code", 20, t["ink"], weight="600"),
             text(LEFT, 58, f"{n_heldout:,} held-out questions from sources none of the systems trained on: CLINC150 intents, ARC-Challenge science questions, synthetic policy tasks.", 12, t["ink2"]),
             text(LEFT, 75, "The other systems run zero-shot; the TF-IDF model is trained on our training split. All five are scored by the same code.", 12, t["ink2"]),
             # legend
             f'<rect x="{LEFT}" y="88" width="12" height="12" rx="2" fill="{t["ours"]}"/>',
             text(LEFT + 18, 98, "openverdict, default variant", 12, t["ink2"]),
             f'<rect x="{LEFT + 196}" y="88" width="12" height="12" rx="2" fill="{t["others"]}"/>',
             text(LEFT + 214, 98, "other systems", 12, t["ink2"])]
    for k, (title, subtitle, get, higher) in enumerate(PANELS):
        ox = LEFT + (k % COLS) * (PANEL_W + 12)
        oy = TOP + (k // COLS) * (PANEL_H + 16)
        items = [(label, float(get(d))) for label, d in rows]
        arrow = "higher is better" if higher else "lower is better"
        parts.append(panel(ox, oy, f"{title}  ({arrow})", subtitle, items, t, highlight_first=True))
    ox = LEFT + 2 * (PANEL_W + 12)
    oy = TOP + (PANEL_H + 16)
    parts.append(panel(ox, oy, "JevBench public tiers", "accuracy by tier, scored by JevBench's own harness",
                       [(f"{name} (n={n})", acc) for name, acc, n in tiers], t, highlight_first=False))
    parts.append(text(LEFT, H - 14, "Values from results/eval/*.json and results/jevbench/; drawn by scripts/make_benchmark_figure.py. "
                                    "Full tables: docs/results.md and results/baselines_table.md.", 11, t["muted"]))
    parts.append("</svg>\n")
    return "\n".join(parts)


def main() -> None:
    out = ROOT / "docs"
    (out / "benchmark.svg").write_text(figure("light"), encoding="utf-8", newline="\n")
    (out / "benchmark-dark.svg").write_text(figure("dark"), encoding="utf-8", newline="\n")
    print("wrote docs/benchmark.svg and docs/benchmark-dark.svg")


if __name__ == "__main__":
    main()
