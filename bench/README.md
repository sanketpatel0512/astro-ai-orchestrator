# Routing Benchmark

Measures whether a model calls the **right tool at the right point in a chain** —
not whether its final answer reads well.

Most agent evaluations score the last message. That tells you a task failed; it
does not tell you where. This harness records the full trajectory of every run
and scores each step against a reference, so the output is a picture of *where*
chains break rather than a pass rate.

## What the model sees

The seven tools the app's agent uses, defined once in [`astro_tools.py`](../astro_tools.py):
`geocode_location`, `get_weather_conditions`, `get_target_transit`,
`get_moon_phase`, `get_darkness_window`, `get_field_of_view`, `save_session_plan`.

[`tools.py`](tools.py) reuses each tool's name, description and argument schema
and swaps only the body for fixture data, so the benchmark measures routing over
exactly the interface the app exposes. If a tool is added to or removed from
`astro_tools.py` without a matching fixture, `tools.py` refuses to import.

The benchmark uses its own neutral system prompt rather than the app's. The
app's prompt supplies coordinates and prescribes a workflow, which would leave
nothing for the model to route.

## What is measured

| metric | definition |
|---|---|
| `coverage` | fraction of required tools actually called |
| `precision` | fraction of distinct calls that were required — spurious calls cost you here |
| `order_valid` | every dependency satisfied before the tool that needed it |
| `params_valid` | fraction of argument assertions that passed; a missing argument fails |
| `redundant_calls` | calls that exactly repeat an earlier one (same tool, same arguments) |
| `complete` | coverage is 1.0, order valid, every argument assertion passed, no forbidden tool called |
| `first_error_step` | 1-indexed position of the first bad call — the interesting number |

A call is *bad* if it is forbidden, spurious, out of order, redundant, or fails
an argument assertion.

**Coordinates are checked, not just order.** When a task geocodes a known city,
every `lat`/`lon` the model passes afterwards is asserted against that city
(within 0.1°). Calling `geocode_location` first and then using some other
coordinates is caught, even though the order looks right.

Reference trajectories are **partial orders**, not fixed sequences. Some steps
genuinely must precede others (you cannot fetch weather before you have
coordinates); the rest are order-free. Penalising a model for picking a
different valid order would measure style, not correctness.

Low coverage with no first error means the model *gave up*; high coverage with
an early first error means it *went wrong*. Both are reported, and runs that hit
the 12-step cap are flagged separately (`step_cap_rate`).

## Why the tools return fixtures

The variable under test is tool *selection*, and live calls would add
third-party latency, rate limits and outages as confounds while making runs
unrepeatable and more expensive. Fixtures make the benchmark free to re-run and
the results reproducible.

## Task suite

24 astrophotography tasks, chain depth 2 to 6 (five per depth, four at depth 6).
Depth is the count of required tools. Every prompt asks for exactly what its
required tools cover, so a model that does what it was asked can score complete.
Tasks carry `forbidden_tools` where a request plainly does not need a given
tool, which is how over-calling gets caught.

`harness.py` validates the suite before running and aborts on: unknown tools,
depth that disagrees with the required list, dependencies or argument checks
pointing at non-required tools, argument checks naming an argument the tool
doesn't have, a geocode check on a city with no fixture, and required/forbidden
overlap.

## Running it

```bash
pip install -r requirements.txt

# 1. verify the pipeline for free, before spending anything (~10 seconds)
python harness.py --models mock_perfect,mock_degrading,mock_spurious,mock_order --trials 1
python score.py
python report.py

# 2. a cheap real smoke test: two models, four tasks, one trial
python harness.py --models claude,openai --trials 1 --tasks T01,T06,T11,T21

# 3. the full run
export ANTHROPIC_API_KEY=... GOOGLE_API_KEY=... OPENAI_API_KEY=... GROQ_API_KEY=...
python harness.py --models claude,gemini,openai,groq --trials 3
python score.py
python report.py
```

Each `harness.py` run writes a new `runs/<timestamp>-<models>.jsonl` and refuses
to overwrite an existing file: raw trajectories are the expensive asset, so
commit them. It scores nothing. `score.py` scores the newest file by default, or
the files you pass with `--runs`; never pass two files covering the same model
and tasks, or those runs are counted twice. Because scoring is separate, the
definition of "correct" can be revised later without paying for the API calls
again.

**Models.** `providers.py` lists one model per provider. Model ids churn, so
override one without editing the file, e.g. `BENCH_MODEL_OPENAI=gpt-5-mini`.
Note that some reasoning models reject `temperature=0`.

**Trials.** Three per task, because models are not deterministic even at
temperature 0. A single trial per task will produce a number; it will not
produce a number worth publishing.

**Cost.** The full run is 24 tasks × 4 models × 3 trials = 288 runs, each
issuing roughly 3–8 model calls. Expect low tens of dollars at current frontier
pricing, less if a cheaper tier is substituted. Start with the smoke test.

## The mock providers

Four scripted models exist so the pipeline can be verified before any spend.
Running them reproduces this table exactly; if a change to the scorer stops
reproducing it, the scorer is wrong.

| mock | behaviour | completion | coverage | precision | order valid | redundant | by depth |
|---|---|---|---|---|---|---|---|
| `mock_perfect` | correct tools, order and arguments | 100% | 100% | 100% | 100% | 0% | 100% at every depth |
| `mock_spurious` | complete, plus one extra call | 96% | 100% | 82% | 100% | 21% | d3 dips to 80% |
| `mock_degrading` | correct for three steps, then stops | 42% | 78% | 100% | 100% | 0% | 100% at d2–d3, 0% at d4+ |
| `mock_order` | all required tools, order reversed | 4% | 100% | 100% | 4% | 0% | 20% at d2, 0% beyond |

- `mock_spurious` loses 4% because on T09 its extra `get_moon_phase` is
  forbidden. On the five tasks that already require both moon and field of view,
  its extra call is a repeat of `get_field_of_view` and shows up as redundant.
- `mock_order` keeps 20% at depth 2 because T04 has no dependencies, so any
  order is valid.
- `mock_degrading` has `median_first_error_step: None`: it never makes a wrong
  call, it just stops.

## Extending

- **Add a tool**: define it in `astro_tools.py` and append it to `ALL_TOOLS`,
  then add a fixture with the same name and signature in `tools.py`. The app
  and the benchmark both pick it up.
- **Add a model**: one entry in `REAL_MODELS` in `providers.py`.
- **Add tasks**: append to `tasks.yaml`; the validator rejects malformed ones.
- **Change the rubric**: edit `score.py`, re-run it over existing `runs/` files,
  then re-run the mocks and confirm the table above still reproduces.

## Reporting honestly

The suite is one domain, one phrasing per task, and a small sample. That is
enough to characterise a *shape* and not enough to rank frontier models
definitively. Fixtures mean the benchmark says nothing about handling real API
failures. Reference trajectories are one person's judgement, explicit and
inspectable in `tasks.yaml`. State the sample size wherever a number appears,
report the spread across tasks rather than a bare mean, and describe the curve
rather than declaring a winner.
