"""
Turn results.json into the README results section: a markdown table and a
small-multiples chart of completion rate against chain depth.

    python report.py

Two design notes, in case anyone asks:

- Small multiples rather than four overlaid lines. With four series every pair
  is on screen at once, and the fourth categorical slot does not clear the
  normal-vision separation floor against the second. Faceting removes the
  problem instead of working around it, and degradation curves are easier to
  compare side by side than tangled together.
- Light and dark PNGs are both emitted. The dark version is stepped for the
  dark surface, not an inverted copy.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": "#2a78d6",
        "suffix": "",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": "#3987e5",
        "suffix": "_dark",
    },
}


def load() -> dict:
    with open(HERE / "results.json", encoding="utf-8") as fh:
        return json.load(fh)["summary"]


def chart(summary: dict, theme_name: str) -> Path:
    t = THEMES[theme_name]
    models = list(summary)
    n = len(models)

    fig, axes = plt.subplots(
        1, n, figsize=(2.65 * n, 3.15), sharey=True,
        facecolor=t["surface"],
    )
    if n == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        s = summary[model]
        depths = sorted(int(d) for d in s["by_depth"])
        rates = [s["by_depth"][str(d)]["completion"] * 100
                 if str(d) in s["by_depth"]
                 else s["by_depth"][d]["completion"] * 100
                 for d in depths]

        ax.set_facecolor(t["surface"])
        ax.set_ylim(-6, 116)
        ax.set_xlim(min(depths) - 0.4, max(depths) + 0.4)
        ax.set_xticks(depths)

        ax.yaxis.grid(True, color=t["grid"], linewidth=0.8, zorder=0)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)

        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(t["axis"])
            ax.spines[spine].set_linewidth(0.8)

        ax.tick_params(colors=t["muted"], labelsize=9, length=0)

        ax.plot(depths, rates, color=t["series"], linewidth=2,
                marker="o", markersize=6.5,
                markeredgecolor=t["surface"], markeredgewidth=1.4, zorder=3)

        # Selective direct labels: first and last point only. Labels drop
        # below the marker near the ceiling so they clear the panel subtitle.
        for idx in {0, len(depths) - 1}:
            high = rates[idx] > 88
            ax.annotate(
                f"{rates[idx]:.0f}%",
                (depths[idx], rates[idx]),
                textcoords="offset points", xytext=(0, -17 if high else 9),
                ha="center", fontsize=9, color=t["secondary"],
            )

        ax.set_title(model, color=t["ink"], fontsize=11,
                     fontweight="semibold", pad=13)
        ax.text(0.5, 1.005, f"{s['completion']:.0%} overall · n={s['n_runs']}",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=8.5, color=t["muted"])

    axes[0].set_ylabel("task completion", color=t["secondary"], fontsize=9.5)
    axes[0].set_yticks([0, 25, 50, 75, 100])
    axes[0].set_yticklabels(["0", "25", "50", "75", "100%"])

    fig.supxlabel("required tool calls per task (chain depth)",
                  color=t["secondary"], fontsize=9.5, y=0.02)
    fig.tight_layout(rect=(0, 0.045, 1, 0.99))

    out = HERE / f"completion_by_depth{t['suffix']}.png"
    fig.savefig(out, dpi=200, facecolor=t["surface"])
    plt.close(fig)
    return out


def table(summary: dict) -> str:
    depths = sorted({int(d) for s in summary.values() for d in s["by_depth"]})
    head = ("| model | completion | coverage | precision | order valid | "
            + " | ".join(f"d{d}" for d in depths) + " |")
    rule = "|---" * (5 + len(depths)) + "|"
    rows = [head, rule]
    for model, s in sorted(summary.items(),
                           key=lambda kv: -kv[1]["completion"]):
        cells = [
            model,
            f"{s['completion']:.0%}",
            f"{s['coverage']:.0%}",
            f"{s['precision']:.0%}",
            f"{s['order_valid_rate']:.0%}",
        ]
        for d in depths:
            bucket = s["by_depth"].get(str(d)) or s["by_depth"].get(d)
            cells.append(f"{bucket['completion']:.0%}" if bucket else "—")
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main() -> None:
    summary = load()
    light = chart(summary, "light")
    dark = chart(summary, "dark")
    md = table(summary)

    n_runs = sum(s["n_runs"] for s in summary.values())
    n_tasks = max(s["n_tasks"] for s in summary.values())

    section = f"""## Results

{md}

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="bench/{dark.name}">
  <img alt="Task completion by chain depth, one panel per model"
       src="bench/{light.name}">
</picture>

**Method.** {n_tasks} astrophotography planning tasks requiring 2–6 tool calls
each, run across {len(summary)} models at temperature 0, {n_runs} runs total.
Tools return deterministic fixtures rather than live API responses, so the thing
measured is routing rather than third-party uptime. A task counts as complete
only if every required tool was called, dependency order was respected, every
argument check passed (including that coordinates match the geocoded city), and
no forbidden tool was invoked. Reference trajectories are partial orders — a model
is not penalised for choosing a different valid sequence.

**Limitations.** Single domain, one prompt phrasing per task, small sample.
The benchmark uses its own neutral system prompt, not the app's, which
prescribes a workflow and supplies coordinates directly.
Treat the ranking as indicative and the shape of the curves as the finding.
"""

    with open(HERE / "RESULTS.md", "w", encoding="utf-8") as fh:
        fh.write(section)

    print(md)
    print(f"\nwrote {light.name}, {dark.name}, RESULTS.md")


if __name__ == "__main__":
    main()
