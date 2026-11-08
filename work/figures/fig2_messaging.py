"""Figure 2: share of worlds where the impossible test passed at the end, by condition.

Colour = whether agents could message each other (two categories). Whiskers = 95% Wilson
intervals (n is small). Counts come from scripts/results_table.py (raw logs).
"""
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from results_table import rows  # noqa: E402

# Reference palette slots (same as Figure 1): messaging = slot 2 (orange), none = slot 1 (blue).
MSG, NOMSG = "#eb6834", "#2a78d6"
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#a3a29d", "#ffffff"

by = {(r["section"], r["label"]): r for r in rows()}
X, M, Q = ("Main task (xarray), DeepSeek v4.1", "Second task (matplotlib), DeepSeek v4.1",
           "Second model (xarray), qwen3.5-397B")


def pool(*keys):
    rs = [by[k] for k in keys]
    return sum(r["passes"] for r in rs), sum(r["n"] for r in rs)


GROUPS = [  # (group title, [(label, (passes, n), messaging?)])
    ("One agent", [
        ("Solo (40 and 240 turns)", pool((X, "Solo, 40 turns"), (X, "Solo, 240 turns")), False),
        ("Solo + read-only git", pool((X, "Solo + git, 40 turns"), (X, "Solo + git, 240 turns")), False),
        ("Solo + private notebook", pool((X, "Solo + notebook (bare)"), (X, "Solo + notebook (matched)"),
                                         (X, "Solo + notebook (instructed)")), False),
    ]),
    ("Six agents, shared repo, no chat", [
        ("With a devil", pool((X, "Team, no chat, with devil")), False),
        ("Six ordinary peers", pool((X, "Team, no chat, all peers")), False),
    ]),
    ("Six agents, shared repo, team chat", [
        ("Six ordinary peers", pool((X, "Team + chat, no devil")), True),
        ("With a devil (control)", pool((X, "Team + chat, control (devil)")), True),
        ("+ verifier angel (talks)", pool((X, "Team + chat, verifier angel")), True),
        ("+ enforcer angel (can restore)", pool((X, "Team + chat, enforcer angel")), True),
    ]),
    ("Replications", [
        ("matplotlib: solo", pool((M, "Solo, 40 turns")), False),
        ("matplotlib: six agents, no chat (devil)", pool((M, "Team, no chat, with devil")), False),
        ("matplotlib: six peers, chat", pool((M, "Team + chat, no devil")), True),
        ("qwen3.5-397B: solo", pool((Q, "Solo, 40 turns")), False),
        ("qwen3.5-397B: six peers, chat", pool((Q, "Team + chat, no devil")), True),
    ]),
]


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
y, yt, yl = 0.0, [], []
for title, items in GROUPS:
    ax.text(-0.02, y + 0.15, title, transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=7.5, fontweight="bold", color=INK)
    y -= 0.85
    for label, (k, n), msg in items:
        lo, hi = wilson(k, n)
        p = k / n if n else 0
        ax.barh(y, p, height=0.62, color=MSG if msg else NOMSG, edgecolor=SURFACE, linewidth=2, zorder=2)
        ax.plot([lo, hi], [y, y], color=INK2, lw=1.0, zorder=3)
        ax.plot(p, y, marker="o", ms=6, color=MSG if msg else NOMSG, mec=SURFACE, mew=1.5, zorder=4)
        for v in (lo, hi):
            ax.plot([v, v], [y - 0.12, y + 0.12], color=INK2, lw=1.0, zorder=3)
        ax.text(1.03, y, f"{k}/{n}", va="center", ha="left", fontsize=7, color=INK)
        yt.append(y)
        yl.append(label)
        y -= 1.0
    y -= 0.5

ax.set_yticks(yt, yl, fontsize=7, color=INK)
ax.set_xlim(0, 1.0)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
ax.tick_params(axis="x", labelsize=7, colors=INK2, length=2)
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color="#e9e8e4", lw=0.6, zorder=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(MUTED)
ax.set_xlabel("Worlds where the impossible test passed at the end (95% Wilson interval)", fontsize=7.5, color=INK2)
ax.set_title("The forbidden edit goes from rare to near-certain once agents can message each other",
             fontsize=8.5, color=INK, loc="left", pad=10)
ax.legend(handles=[Patch(color=NOMSG, label="no messaging between agents"),
                   Patch(color=MSG, label="agents share a team chat")],
          loc="upper center", bbox_to_anchor=(0.35, -0.07), ncol=2, fontsize=7, frameon=False, labelcolor=INK)
fig.tight_layout()
out = Path("figures")
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "fig2_messaging.png", facecolor=SURFACE, bbox_inches="tight")
fig.savefig(out / "fig2_messaging.pdf", facecolor=SURFACE, bbox_inches="tight")
print("wrote", out / "fig2_messaging.png")
