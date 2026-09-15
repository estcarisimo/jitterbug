# AGENTS.md — Jitterbug

Instructions for AI coding agents (Claude Code, Copilot, Codex, Cursor, ...) working in
this repository. `CLAUDE.md` includes this file. Humans: see `CONTRIBUTING.md`.

## What this project is

Jitterbug is a Python package (`jitterbug`, `src/jitterbug/`) that infers network
congestion from Round-Trip Time (RTT) time series: it computes minimum-RTT intervals,
detects change points, and classifies each period as congested or not using latency
jumps and either jitter dispersion or a Kolmogorov–Smirnov test. It implements the
method from *Jitterbug: A New Framework for Jitter-Based Congestion Inference*
(PAM 2022). Version 2.x is a rewrite of the original 1.x scripts.

It is **not** published on PyPI (the name belongs to an unrelated project); install
from GitHub. Do not add `pip install jitterbug` or a PyPI badge anywhere.

## Layout

```
src/jitterbug/
  analyzer.py            JitterbugAnalyzer: load → min-RTT intervals → change points →
                         latency jumps + jitter analysis → CongestionInferenceResult
  models/                Pydantic v2: rtt_data.py (RTTMeasurement, RTTDataset,
                         MinimumRTTDataset), analysis.py (ChangePoint, LatencyJump,
                         JitterAnalysis, CongestionInference*), config.py (JitterbugConfig
                         and per-stage configs)
  detection/             change_point_detector.py (dispatch by config.algorithm),
                         algorithms.py (RupturesDetector, BayesianChangePointDetector)
  analysis/              jitter_analyzer.py (dispersion, KS test), latency_jump_analyzer.py,
                         congestion_inference_analyzer.py
  io/                    data_loader.py (CSV, scamper JSON, InfluxDB), exporters.py
  cli/main.py            Typer CLI: `jitterbug analyze|validate|config|visualize|version`
  visualization/         plotter.py (JitterbugPlotter, matplotlib)
tests/                   pytest; test_cli.py runs the CLI on the bundled dataset
examples/network_analysis/data/raw.csv         PAM 2022 dataset (47 164 RTT samples)
examples/network_analysis/expected_results/    reference output of the paper (BCP + KS)
docs/                    plain Markdown guides
```

## Commands

```bash
uv sync --extra visualization              # environment (Python 3.10–3.13 supported)
uv sync --extra bcp                        # + Bayesian back end (git dependency)
uv run pre-commit install                  # once per clone
uv run ruff check src/ tests/ examples/ tools/
uv run ruff format src/ tests/ examples/ tools/   # CI checks with --check
uv run mypy src/jitterbug                  # advisory; do not add new errors
uv run pytest                              # ~37 tests, seconds
uv run jitterbug analyze examples/network_analysis/data/raw.csv --output /tmp/r.json
uv build                                   # sdist + wheel via uv_build
```

## Conventions

- Python 3.10+: `X | None`, `list[str]`, no `typing.Optional/List`. Ruff target `py310`.
- Line length 100. Ruff is the only linter/formatter (no black/isort/flake8).
  `[tool.ruff.lint] ignore` lists temporary exceptions with counts; remove entries as
  you fix them, never add new ones.
- `pathlib.Path` for paths; `logging` with a module-level logger, never `print`, in
  `src/` outside `cli/`.
- NumPy-style docstrings on public API. Type hints on every function.
- All settings live in Pydantic models in `models/config.py`; new options go there,
  then the CLI, then docs.
- Tests: plain functions, fixtures, `parametrize`; small synthetic series; floats via
  `numpy.testing.assert_allclose`. `tests/test_<module>.py` mirrors `src/`.
- Keep `CHANGELOG.md` current: a bullet under `[Unreleased]` for every user-visible
  change. Version lives only in `pyproject.toml` (and `CITATION.cff` at release time).
- Do not edit `uv.lock` by hand; run `uv lock` / `uv add`. CI uses `uv sync --locked`.
- Do not commit generated outputs (`*.png`, `*.html`, results JSON) outside
  `examples/network_analysis/`.

## Things that are easy to get wrong

- Only two detectors exist: `ruptures` (core dependency) and `bcp` (extra). The
  experimental `torch`, `rbeast` and `adtk` detectors were removed in 2.1; do not
  reintroduce silent fallbacks. `get_available_algorithms()` checks with
  `importlib.util.find_spec`, so it only lists what can run.
- Plotting is matplotlib only (`JitterbugPlotter`). The plotly dashboard and
  interactive modules were removed in 2.1; do not add plotly back without an explicit
  request for interactivity.
- The ruptures detector retries with a lower penalty and tags those results
  `ruptures_<model>_lowpen`.
- The `bcp` extra is a git dependency (`bayesian_changepoint_detection`), so a wheel
  carrying it cannot be uploaded to PyPI. `all` includes it on purpose for now.
- Ruff 0.16 formats fenced Python blocks inside Markdown files in the paths it is
  given; `README.md` is deliberately not in the CI paths yet.
- The `examples/` scripts and notebooks are not run in CI; `tests/test_cli.py` is the
  only thing that exercises the bundled dataset end to end.

## Pull request workflow (required)

`main` is protected by the `protect-main` ruleset: no direct pushes, PR required, the
`lint`, `test (...)` and `build` checks required, Copilot code review requested
automatically, review threads must be resolved.

**Repository policy: every PR must get a Copilot code review, and no PR is merged until
CI is green and Copilot has no unresolved findings.** The ruleset requests the review
on PRs whose base is `main`; stacked PRs need a manual request, and so does every
re-review after a push:

```bash
gh api -X POST repos/estcarisimo/jitterbug/pulls/<n>/requested_reviewers \
  -f 'reviewers[]=copilot-pull-request-reviewer[bot]'
```

The review lands as a *Comment* review a few minutes later (Copilot never approves or
blocks). Read it:

```bash
gh api repos/estcarisimo/jitterbug/pulls/<n>/reviews --jq '.[-1].body'
gh api repos/estcarisimo/jitterbug/pulls/<n>/comments \
  --jq '.[] | select(.in_reply_to_id == null) | {id, path, line, body}'
```

For each comment either **fix it** (commit + push, then re-request the review) or
**reply with the reason it does not apply** (with evidence: a passing CI log, a `ruff
rule` lookup, a reproduction on `main`) and resolve the thread via GraphQL
`resolveReviewThread`. Never resolve a thread without a fix or a written justification.
Copilot does produce false positives; it also reviews a stack one commit at a time, so
check whether a later commit already fixes what it flags. Pre-existing bugs it finds
that are outside the PR's scope go to a GitHub issue, linked from the reply.

Concretely:

1. Branch from `main` (`feat/…`, `fix/…`, `docs/…`, `chore/…`), commit, push, open the
   PR with `gh pr create`. Fill the PR template checklist honestly.
2. Wait for CI: `gh pr checks <n> --watch`. If anything is red, read the log
   (`gh run view <run-id> --log-failed`, or the raw job log via
   `gh api repos/estcarisimo/jitterbug/actions/jobs/<job-id>/logs`), fix locally,
   push, wait again.
3. Make sure a Copilot review is requested and wait for it. Address every comment as
   above.
4. Repeat 2–3 until CI is green **and** Copilot has reviewed the latest push with no
   unresolved threads.
5. Only then merge: `gh pr merge <n> --squash`. Never merge red, never merge without a
   Copilot review.
6. Stacked PRs: merge the bottom one **without** `--delete-branch`; GitHub retargets
   the next PR to `main` automatically. Deleting the base branch first closes the
   stacked PR (it can be recovered: push the old tip back to the branch name,
   `gh pr reopen`, `gh pr edit --base main`, then delete).
7. Do not use the admin bypass. If a human explicitly asks for it in an emergency,
   note it in the PR description.

Never push directly to `main`, never force-push a shared branch, never disable or
weaken a CI check to get green.

## Session continuity

The maintainer keeps a roadmap and a session log in a private Notion page. At the
start of a session, read the latest entry there (or ask for it); at the end, record
what was done, what was verified, what is pending, and the next concrete action, with
branch/commit/PR references.
