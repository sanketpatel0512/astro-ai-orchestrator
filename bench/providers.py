"""
Model registry plus a scripted mock provider.

The mock exists so the whole pipeline — run loop, trajectory capture, scoring,
reporting — can be verified before a cent is spent on API calls. Run
`python harness.py --models mock_perfect,mock_degrading,mock_spurious,mock_order` and the
scorer should return metrics you can predict in advance. If it does not, the
harness is wrong, not the models.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable

# --- real providers ------------------------------------------------------
# Kept lazy: importing a provider package you have not installed should not
# break a mock-only run.

REAL_MODELS: dict[str, dict[str, Any]] = {
    "gemini": {
        "module": "langchain_google_genai",
        "cls": "ChatGoogleGenerativeAI",
        "kwargs": {"model": "gemini-2.5-pro", "temperature": 0},
        "env": "GOOGLE_API_KEY",
    },
    "claude": {
        "module": "langchain_anthropic",
        "cls": "ChatAnthropic",
        "kwargs": {"model": "claude-sonnet-5-5", "temperature": 0},
        "env": "ANTHROPIC_API_KEY",
    },
    "openai": {
        "module": "langchain_openai",
        "cls": "ChatOpenAI",
        "kwargs": {"model": "gpt-4.1", "temperature": 0},
        "env": "OPENAI_API_KEY",
    },
    "groq": {
        "module": "langchain_groq",
        "cls": "ChatGroq",
        "kwargs": {"model": "llama-3.3-70b-versatile", "temperature": 0},
        "env": "GROQ_API_KEY",
    },
}


def load_real(name: str):
    """Instantiate a provider. Raises with a useful message if unavailable."""
    spec = REAL_MODELS[name]
    if not os.environ.get(spec["env"]):
        raise RuntimeError(
            f"{name}: environment variable {spec['env']} is not set."
        )
    try:
        mod = __import__(spec["module"], fromlist=[spec["cls"]])
    except ImportError as exc:
        raise RuntimeError(
            f"{name}: pip install {spec['module'].replace('_', '-')}"
        ) from exc
    kwargs = dict(spec["kwargs"])
    # Model ids churn faster than this file; override without editing it,
    # e.g. BENCH_MODEL_OPENAI=gpt-5-mini.
    kwargs["model"] = os.environ.get(f"BENCH_MODEL_{name.upper()}", kwargs["model"])
    return getattr(mod, spec["cls"])(**kwargs)


# --- mock providers ------------------------------------------------------


@dataclass
class MockCall:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)


class MockModel:
    """Emits a scripted tool trajectory, ignoring the conversation.

    `strategy` receives the task dict and returns a list of MockCall, which the
    harness replays as if the model had chosen them.
    """

    def __init__(self, name: str, strategy: Callable[[dict], list[MockCall]]):
        self.name = name
        self.strategy = strategy
        self.is_mock = True

    def plan(self, task: dict) -> list[MockCall]:
        return self.strategy(task)


def _args_for(tool_name: str, task: dict) -> dict[str, Any]:
    """Correct arguments: the task's own param_checks, with coordinates taken
    from the city the task geocodes, as a model reading the geocode result would."""
    from tools import city

    checks = task.get("param_checks") or {}
    place = city((checks.get("geocode_location") or {}).get("location", "")) or city("austin")
    coords = {"lat": place["lat"], "lon": place["lon"]}
    base: dict[str, Any] = {
        "geocode_location": {"location": place["name"].title()},
        "get_weather_conditions": {**coords},
        "get_target_transit": {"target_name": "Orion Nebula", **coords},
        "get_moon_phase": {"date": "2026-01-01"},
        "get_darkness_window": {**coords, "date": "2026-01-01"},
        "get_field_of_view": {"focal_length_mm": 600, "sensor": "APS-C"},
        "save_session_plan": {"target_name": "Orion Nebula",
                              "start_time": "22:00", "end_time": "03:00"},
    }.get(tool_name, {})
    return {**base, **(checks.get(tool_name) or {})}


def _ordered(task: dict) -> list[str]:
    """Topologically order required tools so dependencies come first."""
    required = list(task["required_tools"])
    deps = task.get("dependencies") or {}
    out: list[str] = []
    remaining = set(required)
    guard = 0
    while remaining and guard < 50:
        guard += 1
        for tool_name in required:
            if tool_name not in remaining:
                continue
            needed = set(deps.get(tool_name, [])) & set(required)
            if needed <= set(out):
                out.append(tool_name)
                remaining.discard(tool_name)
    out.extend(sorted(remaining))
    return out


def perfect(task: dict) -> list[MockCall]:
    return [MockCall(t, _args_for(t, task)) for t in _ordered(task)]


def degrading(task: dict) -> list[MockCall]:
    """Correct for the first three steps, then drops the rest.

    Reproduces the failure shape the benchmark is designed to detect: shallow
    tasks pass, deep tasks fail, and the failure is mid-chain.
    """
    return [MockCall(t, _args_for(t, task)) for t in _ordered(task)[:3]]


def spurious(task: dict) -> list[MockCall]:
    """Complete, but calls an unnecessary tool as well."""
    calls = [MockCall(t, _args_for(t, task)) for t in _ordered(task)]
    extra = "get_moon_phase" if "get_moon_phase" not in task["required_tools"] else "get_field_of_view"
    calls.insert(1, MockCall(extra, _args_for(extra, task)))
    return calls


def order_violating(task: dict) -> list[MockCall]:
    """Calls every required tool, but ignores dependency order."""
    return [MockCall(t, _args_for(t, task)) for t in reversed(_ordered(task))]


MOCK_MODELS = {
    "mock_perfect": lambda: MockModel("mock_perfect", perfect),
    "mock_degrading": lambda: MockModel("mock_degrading", degrading),
    "mock_spurious": lambda: MockModel("mock_spurious", spurious),
    "mock_order": lambda: MockModel("mock_order", order_violating),
}


def load(name: str):
    if name in MOCK_MODELS:
        return MOCK_MODELS[name]()
    return load_real(name)
