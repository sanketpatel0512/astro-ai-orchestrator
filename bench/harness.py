"""
Run the routing benchmark and write raw trajectories to runs.jsonl.

    python harness.py --models mock_perfect,mock_degrading --trials 1
    python harness.py --models claude,gemini,openai,groq --trials 3

Each invocation writes a new file under runs/ and never overwrites an existing
one: raw trajectories are the expensive asset. Nothing is scored here. This file's only job is to produce an honest record of
what each model did, so that scoring can be re-run or re-defined later without
paying for the API calls again.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import yaml

import providers
import tools as toolmod

HERE = Path(__file__).parent
MAX_STEPS = 12

SYSTEM = (
    "You are an astrophotography session planning assistant. Use the tools "
    "available to answer the user's request completely. Call tools in a "
    "sensible order — some tools need coordinates, which come from "
    "geocode_location. "
    "Do not call tools that the request does not need. When you have "
    "everything you need, give a final answer in plain text."
)


def load_tasks() -> list[dict]:
    with open(HERE / "tasks.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def validate_tasks(tasks: list[dict]) -> list[str]:
    """Catch task-definition mistakes before they become bad results."""
    problems = []
    known = set(toolmod.TOOLS_BY_NAME)
    seen_ids = set()
    for task in tasks:
        tid = task["id"]
        if tid in seen_ids:
            problems.append(f"{tid}: duplicate id")
        seen_ids.add(tid)

        required = set(task["required_tools"])
        unknown = required - known
        if unknown:
            problems.append(f"{tid}: unknown tools {sorted(unknown)}")

        if task.get("depth") != len(required):
            problems.append(
                f"{tid}: depth {task.get('depth')} != {len(required)} required tools"
            )

        for tool_name, needs in (task.get("dependencies") or {}).items():
            if tool_name not in required:
                problems.append(f"{tid}: dependency on non-required {tool_name}")
            for need in needs:
                if need not in required:
                    problems.append(
                        f"{tid}: {tool_name} depends on non-required {need}"
                    )

        for tool_name, checks in (task.get("param_checks") or {}).items():
            if tool_name not in required:
                problems.append(
                    f"{tid}: param_check for non-required tool {tool_name}"
                )
            elif tool_name in toolmod.TOOLS_BY_NAME:
                # A check on an argument the tool doesn't have can never pass.
                params = set(toolmod.TOOLS_BY_NAME[tool_name].args)
                for key in set(checks or {}) - params:
                    problems.append(f"{tid}: {tool_name} has no argument {key!r}")

        location = ((task.get("param_checks") or {}).get("geocode_location") or {}).get("location")
        if location and not toolmod.city(location):
            problems.append(f"{tid}: {location!r} is not a fixture city")

        overlap = required & set(task.get("forbidden_tools") or [])
        if overlap:
            problems.append(f"{tid}: {sorted(overlap)} both required and forbidden")
    return problems


def run_mock(model, task: dict) -> dict:
    toolmod.reset_log()
    for call in model.plan(task):
        fn = toolmod.TOOLS_BY_NAME.get(call.tool)
        if fn is None:
            toolmod.record(call.tool, call.args)
            continue
        try:
            fn.invoke(call.args)
        except Exception as exc:  # noqa: BLE001 - recorded, not raised
            toolmod.record(call.tool, {"_error": str(exc), **call.args})
    return {"calls": toolmod.drain_log(), "steps": None, "hit_step_cap": False,
            "error": None, "final_text": "(mock)"}


def run_live(model, task: dict) -> dict:
    """Standard tool-calling loop. Records every call the model makes."""
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

    bound = model.bind_tools(toolmod.ALL_TOOLS)
    messages = [SystemMessage(SYSTEM), HumanMessage(task["prompt"])]
    toolmod.reset_log()
    steps = 0
    error = None
    final_text = ""
    answered = False

    try:
        while steps < MAX_STEPS:
            steps += 1
            reply = bound.invoke(messages)
            messages.append(reply)
            calls = getattr(reply, "tool_calls", None) or []
            if not calls:
                final_text = _text(reply.content)
                answered = True
                break
            for call in calls:
                fn = toolmod.TOOLS_BY_NAME.get(call["name"])
                if fn is None:
                    result = f"No such tool: {call['name']}"
                    toolmod.record(call["name"], call.get("args", {}))
                else:
                    try:
                        result = fn.invoke(call.get("args", {}))
                    except Exception as exc:  # noqa: BLE001
                        result = f"Tool error: {exc}"
                messages.append(
                    ToolMessage(content=str(result), tool_call_id=call["id"])
                )
    except Exception as exc:  # noqa: BLE001 - a provider failure is a data point
        error = f"{type(exc).__name__}: {exc}"

    return {"calls": toolmod.drain_log(), "steps": steps,
            "hit_step_cap": not answered and error is None,
            "error": error, "final_text": final_text}


def _text(content) -> str:
    """Anthropic and Gemini can return a list of content blocks, not a string."""
    if isinstance(content, str):
        return content
    return "\n".join(b.get("text", "") for b in content
                     if isinstance(b, dict) and b.get("type") == "text")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True,
                    help="comma-separated: claude,gemini,openai,groq or mock_*")
    ap.add_argument("--trials", type=int, default=3,
                    help="repeats per task; models are stochastic even at temp 0")
    ap.add_argument("--out", default=None,
                    help="output path; default runs/<timestamp>-<models>.jsonl")
    ap.add_argument("--tasks", default=None,
                    help="optional comma-separated task ids to restrict to")
    args = ap.parse_args()

    tasks = load_tasks()
    problems = validate_tasks(tasks)
    if problems:
        print("Task suite is invalid — fix these before running:")
        for p in problems:
            print("  -", p)
        raise SystemExit(1)

    if args.tasks:
        wanted = set(args.tasks.split(","))
        tasks = [t for t in tasks if t["id"] in wanted]

    names = [n.strip() for n in args.models.split(",")]
    if args.out:
        out_path = Path(args.out)
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = HERE / "runs" / f"{stamp}-{'_'.join(names)}.jsonl"
    if out_path.exists():
        raise SystemExit(f"{out_path} already exists; refusing to overwrite raw runs.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with open(out_path, "x", encoding="utf-8") as fh:
        for name in names:
            try:
                model = providers.load(name)
            except RuntimeError as exc:
                print(f"skipping {name}: {exc}")
                continue

            is_mock = getattr(model, "is_mock", False)
            print(f"running {name} — {len(tasks)} tasks x {args.trials} trials")

            for task in tasks:
                for trial in range(args.trials):
                    started = time.time()
                    result = (run_mock(model, task) if is_mock
                              else run_live(model, task))
                    record = {
                        "model": name,
                        "task_id": task["id"],
                        "depth": task["depth"],
                        "trial": trial,
                        "elapsed_s": round(time.time() - started, 2),
                        **result,
                    }
                    fh.write(json.dumps(record) + "\n")
                    written += 1
                    if result["error"]:
                        print(f"  {task['id']} t{trial}: {result['error']}")

    print(f"wrote {written} runs to {out_path}")


if __name__ == "__main__":
    main()
