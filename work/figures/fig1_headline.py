"""Headline figure: three bars, the main xarray comparison only (pooled counts from results_table).

One agent without notes (0/14), six agents sharing a repo without chat (1/6), with chat (9/9).
Thin whiskers = 95% Wilson intervals. Replications and defences stay in the text and tables.
"""
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from results_table import rows  # noqa: E402

NOMSG, MSG = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e9e8e4", "#ffffff"
X = "Main task (xarray), DeepSeek v4.1"
by = {(r["section"], r["label"]): r for r in rows()}


def pool(*labels):
    rs = [by[(X, l)] for l in labels]
    return sum(r["passes"] for r in rs), sum(r["n"] for r in rs)


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


BARS = [
    ("One agent,\nno notes", pool("Solo, 40 turns", "Solo, 240 turns", "Solo + git, 40 turns", "Solo + git, 240 turns"), NOMSG),
    ("Six agents,\nno chat", pool("Team, no chat, with devil", "Team, no chat, all peers"), NOMSG),
    ("Six agents,\nteam chat", pool("Team + chat, no devil", "Team + chat, control (devil)", "Team + chat, verifier angel"), MSG),
]

fig, ax = plt.subplots(figsize=(4.6, 3.4), dpi=300)
fig.patch.set_facecolor(SURFACE)
for i, (label, (k, n), color) in enumerate(BARS):
    p = k / n
    lo, hi = wilson(k, n)
    ax.bar(i, p, width=0.58, color=color, zorder=2)
    ax.plot([i, i], [lo, hi], color=INK2, lw=1, zorder=3)
    for v in (lo, hi):
        ax.plot([i - 0.07, i + 0.07], [v, v], color=INK2, lw=1, zorder=3)
    ax.text(i, max(p, hi) + 0.035, f"{k}/{n}", ha="center", va="bottom", fontsize=9, fontweight="bold", color=INK)
ax.set_xticks(range(len(BARS)), [b[0] for b in BARS], fontsize=8.5, color=INK)
ax.set_ylim(0, 1.15)
ax.set_yticks([0, 0.5, 1.0], ["0%", "50%", "100%"])
ax.tick_params(axis="y", labelsize=8, colors=INK2, length=0)
ax.tick_params(axis="x", length=0)
ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#a3a29d")
ax.set_ylabel("Worlds where the impossible\ntest passed", fontsize=8.5, color=INK2)
fig.tight_layout()
out = Path("figures")
out.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(out / f"fig1_headline.{ext}", facecolor=SURFACE, bbox_inches="tight")
print("wrote", out / "fig1_headline.png", [(b[0].replace("\n", " "), b[1]) for b in BARS])
