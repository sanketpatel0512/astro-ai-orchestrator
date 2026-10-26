"""
Score runs.jsonl against the reference trajectories.

    python score.py                       # newest file in runs/
    python score.py --runs runs/a.jsonl runs/b.jsonl

Definitions, stated plainly because the whole result depends on them:

  coverage        fraction of required tools the model actually called
  precision       fraction of distinct calls that were required (spurious calls hurt)
  order_valid     every dependency was satisfied before the dependent call
  params_valid    fraction of argument assertions that passed. A missing
                  argument fails its assertion. When the task geocodes a known
                  city, every lat/lon passed afterwards is also asserted against
                  that city, so "geocoded Austin, then used Denver's coordinates"
                  is caught even though the order was right.
  forbidden_hit   the model called a tool the task says it should not
  redundant_calls calls that exactly repeat an earlier call (same tool, same args)
  complete        coverage == 1, order_valid, no forbidden call, and every
                  argument assertion passed

  first_error_step  1-indexed position of the first call that was spurious,
                    forbidden, out of order, redundant, or had bad arguments.
                    None if the trajectory was clean. This is the number that says WHERE chains break,
                    which is the interesting part and the reason the harness
                    records full trajectories rather than pass/fail.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).parent
COORD_TOL_DEG = 0.1  # models round geocode output; 0.1 deg is ~11 km
NUM_TOL = 0.05

from tools import city  # noqa: E402


def load_tasks() -> dict[str, dict]:
    with open(HERE / "tasks.yaml", encoding="utf-8") as fh:
        return {t["id"]: t for t in yaml.safe_load(fh)}


def expected_coords(task: dict) -> dict | None:
    """The fixture coordinates of the city the task geocodes, if any."""
    location = ((task.get("param_checks") or {}).get("geocode_location") or {}).get("location")
    hit = city(location) if location else None
    return {"lat": hit["lat"], "lon": hit["lon"]} if hit else None


def check_params(tool_name: str, args: dict, checks: dict,
                 coords: dict | None = None) -> tuple[int, int]:
    """Returns (passed, total) argument assertions for one call."""
    expected = dict((checks or {}).get(tool_name) or {})
    if coords:
        for key in ("lat", "lon"):
            if key in args:
                expected.setdefault(key, coords[key])
    if not expected:
        return (0, 0)
    passed = 0
    for key, want in expected.items():
        got = args.get(key)
        if got is None:
            continue  # a missing argument fails its assertion
        if isinstance(want, str):
            if want.lower() in str(got).lower():
                passed += 1
        else:
            tol = COORD_TOL_DEG if key in ("lat", "lon") else NUM_TOL
            try:
                if abs(float(got) - float(want)) <= tol:
                    passed += 1
            except (TypeError, ValueError):
                pass
    return (passed, len(expected))


def score_run(run: dict, task: dict) -> dict:
    required = list(task["required_tools"])
    deps = task.get("dependencies") or {}
    forbidden = set(task.get("forbidden_tools") or [])
    checks = task.get("param_checks") or {}
    coords = expected_coords(task)

    calls = run.get("calls") or []
    called_seq = [c["tool"] for c in calls]
    called_set = set(called_seq)

    covered = called_set & set(required)
    coverage = len(covered) / len(required) if required else 1.0
    precision = len(covered) / len(called_set) if called_set else 0.0

    # order + first error
    seen: set[str] = set()
    order_valid = True
    first_error = None
    params_passed = params_total = 0
    params_ok = True
    redundant = 0
    previous: list[tuple[str, str]] = []

    for idx, call in enumerate(calls, start=1):
        name = call["tool"]
        bad = False

        if name in forbidden:
            bad = True
        elif name not in required:
            bad = True  # spurious
        else:
            needed = set(deps.get(name, [])) & set(required)
            if not needed <= seen:
                order_valid = False
                bad = True

        args = call.get("args", {})
        p, t = check_params(name, args, checks, coords)
        params_passed += p
        params_total += t
        if t and p < t:
            params_ok = False
            bad = True

        signature = (name, json.dumps(args, sort_keys=True, default=str))
        if signature in previous:
            redundant += 1
            bad = True
        previous.append(signature)

        if bad and first_error is None:
            first_error = idx
        seen.add(name)

    return {
        "model": run["model"],
        "task_id": run["task_id"],
        "depth": run["depth"],
        "trial": run["trial"],
        "n_calls": len(calls),
        "coverage": round(coverage, 4),
        "precision": round(precision, 4),
        "order_valid": order_valid,
        "params_valid": round(params_passed / params_total, 4) if params_total else None,
        "forbidden_hit": bool(called_set & forbidden),
        "redundant_calls": redundant,
        "complete": (coverage == 1.0 and order_valid and params_ok
                     and not (called_set & forbidden)),
        "first_error_step": first_error,
        "errored": bool(run.get("error")),
        "hit_step_cap": bool(run.get("hit_step_cap")),
        "elapsed_s": run.get("elapsed_s"),
    }


def aggregate(scored: list[dict]) -> dict:
    by_model: dict[str, list[dict]] = defaultdict(list)
    for s in scored:
        by_model[s["model"]].append(s)

    summary = {}
    for model, rows in by_model.items():
        depths: dict[int, list[dict]] = defaultdict(list)
        for r in rows:
            depths[r["depth"]].append(r)

        by_depth = {}
        for depth in sorted(depths):
            group = depths[depth]
            by_depth[depth] = {
                "n": len(group),
                "completion": round(sum(r["complete"] for r in group) / len(group), 4),
                "coverage": round(statistics.mean(r["coverage"] for r in group), 4),
                "precision": round(statistics.mean(r["precision"] for r in group), 4),
            }

        # Per-task completion, used for a spread estimate that is honest about
        # how few tasks there are.
        per_task: dict[str, list[bool]] = defaultdict(list)
        for r in rows:
            per_task[r["task_id"]].append(r["complete"])
        task_rates = [sum(v) / len(v) for v in per_task.values()]

        errs = [r["first_error_step"] for r in rows if r["first_error_step"]]
        params = [r["params_valid"] for r in rows if r["params_valid"] is not None]

        summary[model] = {
            "n_runs": len(rows),
            "n_tasks": len(per_task),
            "completion": round(sum(r["complete"] for r in rows) / len(rows), 4),
            "completion_task_spread": [
                round(min(task_rates), 4), round(max(task_rates), 4)
            ] if task_rates else None,
            "coverage": round(statistics.mean(r["coverage"] for r in rows), 4),
            "precision": round(statistics.mean(r["precision"] for r in rows), 4),
            "order_valid_rate": round(
                sum(r["order_valid"] for r in rows) / len(rows), 4),
            "params_valid": round(statistics.mean(params), 4) if params else None,
            "redundant_rate": round(
                sum(r["redundant_calls"] > 0 for r in rows) / len(rows), 4),
            "step_cap_rate": round(
                sum(r["hit_step_cap"] for r in rows) / len(rows), 4),
            "forbidden_rate": round(
                sum(r["forbidden_hit"] for r in rows) / len(rows), 4),
            "provider_error_rate": round(
                sum(r["errored"] for r in rows) / len(rows), 4),
            "median_first_error_step": statistics.median(errs) if errs else None,
            "first_error_hist": {
                str(k): errs.count(k) for k in sorted(set(errs))
            },
            "mean_latency_s": round(
                statistics.mean([r["elapsed_s"] for r in rows if r["elapsed_s"]]), 2
            ) if any(r["elapsed_s"] for r in rows) else None,
            "by_depth": by_depth,
        }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=None,
                    help="run files to score; default the newest in runs/. "
                         "Never pass two files covering the same model and "
                         "tasks, or those runs are counted twice.")
    args = ap.parse_args()
    if args.runs:
        paths = [Path(p) for p in args.runs]
    else:
        found = sorted((HERE / "runs").glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        if not found:
            raise SystemExit("no runs/*.jsonl yet; run harness.py first")
        paths = [found[-1]]

    tasks = load_tasks()
    runs = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    runs.append(json.loads(line))
    print("scoring " + ", ".join(str(p) for p in paths))

    scored = [score_run(r, tasks[r["task_id"]]) for r in runs]
    summary = aggregate(scored)

    out = {"per_run": scored, "summary": summary}
    with open(HERE / "results.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print(f"scored {len(scored)} runs across {len(summary)} models\n")
    for model, s in summary.items():
        print(f"{model}")
        print(f"  completion       {s['completion']:.0%}  "
              f"(n={s['n_runs']} runs over {s['n_tasks']} tasks)")
        print(f"  coverage         {s['coverage']:.0%}")
        print(f"  precision        {s['precision']:.0%}")
        print(f"  order valid      {s['order_valid_rate']:.0%}")
        print(f"  params valid     {s['params_valid']:.0%}" if s["params_valid"] is not None
              else "  params valid     n/a")
        print(f"  redundant        {s['redundant_rate']:.0%}")
        print(f"  median 1st error step  {s['median_first_error_step']}")
        print("  by depth         " + "  ".join(
            f"d{d}:{v['completion']:.0%}" for d, v in s["by_depth"].items()))
        print()
    print("wrote results.json")


if __name__ == "__main__":
    main()
