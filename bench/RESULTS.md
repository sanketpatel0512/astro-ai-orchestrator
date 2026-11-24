## Results

| model | completion | coverage | precision | order valid | d2 | d3 | d4 | d5 | d6 |
|---|---|---|---|---|---|---|---|---|---|
| claude | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| gemini | 100% | 100% | 99% | 100% | 100% | 100% | 100% | 100% | 100% |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="bench/completion_by_depth_dark.png">
  <img alt="Task completion by chain depth, one panel per model"
       src="bench/completion_by_depth.png">
</picture>

**Method.** 24 astrophotography planning tasks requiring 2–6 tool calls
each, run across 2 models at temperature 0 where the provider
allows it (Claude 5 models run at their default), 144 runs total.
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
