"""
One image comparing the first live run with the run after the benchmark fixes.

    python before_after.py        # reads before_after.json, writes before_after.png

before_after.json holds both passes' summaries (see its "_about" field for
where each number came from), so the chart can be rebuilt without re-scoring.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).parent

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8983"
GRID = "#e6e5e0"
MODELS = {  # categorical slots 1 and 2 of the reference palette
    "claude": {"label": "Claude Sonnet 5.5", "color": "#2a78d6"},
    "gemini": {"label": "Gemini 3.8 Flash", "color": "#eb6834"},
}
# Where each first-run dip came from, keyed (model, depth) -> footnote number.
DIPS = {("claude", "5"): "1", ("gemini", "5"): "1", ("claude", "6"): "2"}

FINDINGS = [
    ("1", "Task bug",
     "T20 asked for a plan “if it’s viable”, but the Seattle test data always reported 68% cloud.",
     "Both models rightly declined, and were marked wrong.",
     "Prompt made unconditional."),
    ("2", "Fake data bug",
     "The transit tool returned the same time for every target.",
     "Claude spotted an impossible Ring Nebula transit and refused to plan on it.",
     "Positions now computed from real sky coordinates (within 0.4° of astropy)."),
    ("3", "Scoring bug",
     "A moon or darkness check before giving advice counted as an error.",
     "Both models made the same extra checks on the same tasks, every trial.",
     "Counted as acceptable on planning tasks and reported separately."),
]


def pct(x: float) -> str:
    return f"{x:.0%}"


def main() -> None:
    data = json.loads((HERE / "before_after.json").read_text(encoding="utf-8"))
    before, after = data["before"], data["after"]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "text.color": INK, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2})
    fig = plt.figure(figsize=(12, 12.4), dpi=150, facecolor=SURFACE)
    grid = fig.add_gridspec(3, 2, height_ratios=[1.3, 0.85, 1.0],
                            left=0.07, right=0.96, top=0.83, bottom=0.02,
                            hspace=0.30, wspace=0.14)

    fig.text(0.07, 0.955, "The model failures were benchmark bugs",
             fontsize=21, fontweight="bold", va="bottom")
    fig.text(0.07, 0.93,
             "Tool-routing eval: 24 astrophotography planning tasks needing 2–6 tool "
             "calls each, 3 trials per task, 72 runs per model per pass.",
             fontsize=11.5, color=INK_2, va="bottom")
    fig.text(0.07, 0.907,
             "Before: the first live run. After: the same models rerun once the evaluation "
             "itself was fixed.",
             fontsize=11.5, color=INK_2, va="bottom")

    # --- row 1: completion by depth, one panel per model --------------------
    depths = ["2", "3", "4", "5", "6"]
    for col, (key, spec) in enumerate(MODELS.items()):
        ax = fig.add_subplot(grid[0, col], facecolor=SURFACE)
        b = [before[key]["by_depth"][d] * 100 for d in depths]
        a = [after[key]["by_depth"][d] * 100 for d in depths]
        x = [int(d) for d in depths]
        ax.plot(x, b, color=spec["color"], lw=2, ls=(0, (4, 3)), alpha=0.6,
                marker="o", ms=8, mfc=SURFACE, mew=2, zorder=3)
        ax.plot(x, a, color=spec["color"], lw=2, marker="o", ms=8,
                mec=SURFACE, mew=2, zorder=4)
        for d, val in zip(depths, b):
            note = DIPS.get((key, d))
            if note:
                ax.annotate(f"{val:.0f}%  ← bug {note}", (int(d), val),
                            xytext=(10, -4), textcoords="offset points",
                            fontsize=10, color=INK, va="center")
        ax.text(6.12, 100, "100%", fontsize=10, color=INK, va="center")
        ax.text(0, 1.10, spec["label"], transform=ax.transAxes, fontsize=13,
                fontweight="bold", ha="left", va="bottom")
        ax.text(0, 1.03, f"completion {pct(before[key]['completion'])} → "
                f"{pct(after[key]['completion'])}", transform=ax.transAxes,
                fontsize=10.5, color=INK_2, ha="left", va="bottom")
        ax.set_xlim(1.7, 6.9)
        ax.set_ylim(70, 104)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{d} tools" for d in depths])
        ax.set_yticks([70, 80, 90, 100])
        ax.set_yticklabels([] if col else ["70%", "80%", "90%", "100%"])
        if not col:
            ax.set_ylabel("tasks completed correctly")
        ax.set_xlabel("tool calls the task needs")
        ax.grid(axis="y", color=GRID, lw=1)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(length=0)

    legend = [
        Line2D([], [], color=INK_2, lw=2, ls=(0, (4, 3)), marker="o", ms=7,
               mfc=SURFACE, mew=2, alpha=0.7, label="Before (first run)"),
        Line2D([], [], color=INK_2, lw=2, marker="o", ms=7, label="After (eval fixed, full rerun)"),
    ]
    fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(0.062, 0.9),
               ncol=2, frameon=False, fontsize=10.5, handlelength=3)

    # --- row 2: the numbers --------------------------------------------------
    ax = fig.add_subplot(grid[1, :], facecolor=SURFACE)
    ax.axis("off")
    rows = [
        ("Task completion", "completion", pct),
        ("Precision (share of calls that were needed)", "precision", pct),
        ("Runs marked failed", "failed_runs", str),
        ("Runs penalised for an extra tool call  ← bug 3", "runs_flagged_extra_call", str),
        ("Provider errors", "provider_errors", str),
        ("Mean time per task", "mean_latency_s", lambda v: f"{v:.1f} s"),
    ]
    cols_x = [0.0, 0.56, 0.80]
    top = 0.95
    step = 0.14
    ax.text(cols_x[0], top, "Metric (72 runs per model per pass)", fontsize=10.5,
            color=INK_2, fontweight="bold", transform=ax.transAxes)
    for cx, (key, spec) in zip(cols_x[1:], MODELS.items()):
        ax.plot([cx, cx + 0.018], [top + 0.025] * 2, color=spec["color"], lw=5,
                solid_capstyle="round", transform=ax.transAxes)
        ax.text(cx + 0.03, top, spec["label"], fontsize=10.5, fontweight="bold",
                transform=ax.transAxes)
    ax.plot([0, 1], [top - 0.045] * 2, color=GRID, lw=1, transform=ax.transAxes)
    for i, (label, field, fmt) in enumerate(rows):
        y = top - step * (i + 1)
        ax.text(cols_x[0], y, label, fontsize=11, transform=ax.transAxes)
        for cx, key in zip(cols_x[1:], MODELS):
            b, a = before[key][field], after[key][field]
            ax.text(cx + 0.03, y, f"{fmt(b)}  →  ", fontsize=11, color=INK_2,
                    transform=ax.transAxes)
            ax.text(cx + 0.03 + 0.088, y, fmt(a), fontsize=11, fontweight="bold",
                    transform=ax.transAxes)
    ax.plot([0, 1], [top - step * len(rows) - 0.06] * 2, color=GRID, lw=1,
            transform=ax.transAxes)
    ax.text(0, top - step * len(rows) - 0.10,
            "After the fix, 27 of 72 runs per model still made a reasonable extra check; these "
            "are now reported, not penalised.\n"
            "Gemini’s remaining 3: an unrequested darkness "
            "check on T19 (“No plan needed”), in every trial. Times are wall-clock "
            "from one machine, so treat them as rough.",
            fontsize=9.5, color=INK_2, transform=ax.transAxes, va="top", linespacing=1.5)

    # --- row 3: what the eval caught, and the fix ------------------------------
    ax = fig.add_subplot(grid[2, :], facecolor=SURFACE)
    ax.axis("off")
    ax.text(0, 1.0, "What reading the trajectories revealed, and the fix",
            fontsize=13, fontweight="bold", transform=ax.transAxes, va="top")
    for i, (num, kind, found, evidence, fix) in enumerate(FINDINGS):
        y = 0.78 - i * 0.31
        ax.text(0.012, y, num, fontsize=12, fontweight="bold", color=SURFACE,
                ha="center", va="center", transform=ax.transAxes,
                bbox={"boxstyle": "circle,pad=0.35", "fc": INK, "ec": INK})
        ax.text(0.04, y, kind, fontsize=11, fontweight="bold", va="center",
                transform=ax.transAxes)
        ax.text(0.17, y + 0.045, found, fontsize=10.5, va="center", transform=ax.transAxes)
        ax.text(0.17, y - 0.035, evidence, fontsize=10.5, color=INK_2, va="center",
                transform=ax.transAxes)
        ax.text(0.17, y - 0.11, "Fix: " + fix, fontsize=10.5, fontweight="bold",
                va="center", transform=ax.transAxes)

    out = HERE / "before_after.png"
    fig.savefig(out, facecolor=SURFACE)
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
