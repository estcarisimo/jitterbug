# AGENTS.md — Jitterbug

Instructions for AI coding agents (Claude Code, Codex, Cursor, ...) working in
this repository. `CLAUDE.md` includes this file. Humans: see `CONTRIBUTING.md`.

## What this project is

Jitterbug is a Python package (`jitterbug`, `src/jitterbug/`) that infers network
congestion from Round-Trip Time (RTT) time series: it computes minimum-RTT intervals,
detects change points, and classifies each period as congested or not using latency
jumps and either jitter dispersion or a Kolmogorov–Smirnov test. It implements the
method from *Jitterbug: A New Framework for Jitter-Based Congestion Inference*
(PAM 2022). Version 2.x is a rewrite of the original 1.x scripts.

The distribution is named `jitterbug-inference` (`jitterbug` on PyPI belongs to an
unrelated project); the import package and the CLI are still `jitterbug`. It is **not**
published on PyPI yet (planned once the Bayesian back end is on PyPI); install from
GitHub. Never write `pip install jitterbug`, and no PyPI badge until the first upload.

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
tests/                   pytest; test_cli.py runs the CLI on the bundled dataset;
                         test_paper_regression.py pins the paper-dataset results
examples/network_analysis/data/raw.csv         PAM 2022 dataset (47 163 RTT samples)
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
uv run mypy src/jitterbug                  # blocking in CI (disallow_untyped_defs)
uv run pytest -m "not slow"                # seconds
uv run pytest                              # + Bayesian regression (~2 min, needs --extra bcp)
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
`lint`, `test (...)` and `build` checks required, review threads must be resolved.

**Repository policy: every PR gets an independent code review from a fresh session, and
no PR is merged until CI is green and that review returns `APPROVE` on the final
commit.** "Fresh" means a reviewer with no context from the session that wrote the
change: a new AI agent session started for the review alone (a Claude Sonnet subagent
today), or a human. The brief the reviewer follows is `.github/REVIEW.md`; give it the
PR number and nothing else. This replaced GitHub Copilot review in
September 2026 (cost); the Copilot-specific steps are gone.

The loop:

1. Branch from `main` (`feat/…`, `fix/…`, `docs/…`, `chore/…`), commit, push, open the
   PR with `gh pr create`. Fill the PR template checklist honestly.
2. Wait for CI: `gh pr checks <n> --watch`. If anything is red, read the log
   (`gh run view <run-id> --log-failed`, or the raw job log via
   `gh api repos/estcarisimo/jitterbug/actions/jobs/<job-id>/logs`), fix locally,
   push, wait again.
3. Start a fresh reviewer session with `.github/REVIEW.md` and the PR number. Wait for
   its verdict, then **post the verdict in full as a PR comment**
   (`gh pr comment <n> --body-file <verdict.md>`, prefixed with the round number and
   the commit reviewed). The verdict lives on the PR, not in a session transcript.
4. For each finding either **fix it** (commit + push) or **rebut it with evidence** (a
   passing CI log, a `ruff rule` lookup, a reproduction on `main`) in a PR comment.
   Reviewers do produce false positives. Pre-existing bugs outside the PR's scope go
   to a GitHub issue, linked from the PR.
5. After any push, start **another** fresh reviewer (never reuse the previous session).
   It reads the earlier rounds (`gh pr view <n> --comments`) and reports any finding
   that was neither fixed nor validly rebutted. Repeat 2–5 until CI is green **and** the
   latest push has `VERDICT: APPROVE`.
6. Only then merge: `gh pr merge <n> --squash`. Never merge red, never merge without an
   `APPROVE` on the final commit.
7. Stacked PRs: merge the bottom one **without** `--delete-branch`; GitHub retargets
   the next PR to `main` automatically. If the next PR shows `DIRTY` because the squash
   touched files it deletes, first check that `git rev-parse origin/main^{tree}` equals
   the tree of the previous branch's tip (i.e. `main` now contains exactly what the
   branch was built on); only then update the branch with
   `git merge -s ours origin/main`, push, and let CI run.
8. Do not use the admin bypass. If a human explicitly asks for it in an emergency,
   note it in the PR description.

Never push directly to `main`, never force-push a shared branch, never disable or
weaken a CI check to get green.

## Session continuity

The maintainer keeps a roadmap and a session log in a private Notion page. At the
start of a session, read the latest entry there (or ask for it); at the end, record
what was done, what was verified, what is pending, and the next concrete action, with
branch/commit/PR references.
