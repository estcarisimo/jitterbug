# Changelog

All notable changes to Jitterbug are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Unit tests for `analysis/` (latency jumps, jitter dispersion, KS test, the congestion
  state machine) and `io/` (CSV, DataFrame, scamper JSON, format inference, mocked
  InfluxDB, validation, the JSON/CSV/summary exporters) on small synthetic series.
  Coverage 23 % → 84 %; CI now fails below 75 %.
- `tests/test_paper_regression.py`: golden counts for both detectors on the PAM 2022
  dataset (ruptures + jitter dispersion: 22 periods / 11 congested; BCP + KS test:
  34 / 14) and overlap-based agreement with the paper's reference intervals (11/15 and
  13/15 recovered, no spurious detections). A dedicated CI job runs the Bayesian half
  with the `bcp` extra; Linux resolves torch from the CPU wheel index.
- `tests/test_cli.py::test_visualize_writes_the_standard_plots`: first test of the
  visualization code path (headless, `MPLBACKEND=Agg`).
- `bcp_device` option in `change_point_detection` (default `cpu`). The Bayesian library
  picks a GPU when it sees one, and on Apple Silicon that made the paper's configuration
  take over half an hour; on CPU it takes about two minutes. The value is validated
  (`cpu`, `cuda` or `mps`), and a test with a mocked backend checks that it reaches
  both the likelihood and the detection call.
- `get_available_algorithms()` now reports only the detectors whose packages are
  installed (`importlib.util.find_spec`), with tests.
- Continuous integration on GitHub Actions: ruff lint and format checks, mypy
  (blocking since #19), `pip-audit` over the locked dependency set, tests on Python 3.10–3.13
  (Ubuntu) and 3.12 (macOS) with a CLI smoke test on the bundled PAM 2022 dataset,
  and a build job that installs the wheel in a clean environment. (#4)
- `tests/test_cli.py`: first tests for the command-line interface, run through Typer's
  `CliRunner`. (#5)
- Extras `rbeast` and `adtk` for the corresponding change point back ends, which the
  code already supported but no extra declared. `bayesian` is kept as a deprecated
  alias of `bcp`. (#5)
- Pre-commit configuration (ruff plus file hygiene hooks), Dependabot for GitHub
  Actions and Python dependencies, and a committed `uv.lock`. (#4)
- Community and policy files: `SECURITY.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`,
  `AGENTS.md`, `CITATION.cff`, `CODEOWNERS`, issue and pull request templates.

### Changed

- Code review policy: PRs are reviewed by an independent, fresh-context session
  following `.github/REVIEW.md` (today a Claude Sonnet subagent) instead of GitHub
  Copilot; merge requires `APPROVE` on the final commit. `AGENTS.md`, `CONTRIBUTING.md`
  and the PR template updated.

- Dependency floors raised to releases that support Python 3.10 (`numpy>=1.24`,
  `pandas>=2.0`, `scipy>=1.10`, `pydantic>=2.5`, `pydantic-settings>=2.1`,
  `ruptures>=1.1.9`, `typer>=0.12`, `rich>=13`); `click` and `requests` were declared
  but never imported and are no longer dependencies.
- `bandit` runs in the CI lint job (clean at the time of writing).
- mypy passes with `disallow_untyped_defs` on the whole package and is now a blocking CI
  check. Pydantic models use `model_config = ConfigDict(...)` / `SettingsConfigDict`
  and `model_dump()` instead of the deprecated `class Config` / `.dict()`; the two
  remaining test warnings are from third-party libraries.
- README rewritten in the project's canonical layout: features, quick start, usage,
  configuration, architecture, development, example output, citation. The table of
  contents, the duplicated installation sections and the `pip install jitterbug`
  instructions are gone; every command was run before being documented. The REST API,
  Docker and `visualize` command are not documented until they work (#6, #7).
- **Python 3.10 or newer is required** (was 3.8). (#4)
- All temporary ruff ignores are gone except `B008` (Typer's `Option(...)` defaults):
  `pathlib` everywhere (`Path.open`, `/`, `mkdir`, `iterdir`), `raise ... from` inside
  every `except`, collapsed nested conditions, no line over 100 characters. `tools/`
  scripts no longer patch `sys.path` to a directory that does not exist. Behaviour
  unchanged (CLI output on the bundled dataset is identical).
- `pyproject.toml` is the single source of packaging metadata; `setup.py`,
  `requirements.txt`, `requirements-new.txt` and `install_dev.sh` are gone. The build
  backend is `uv_build`. Development tools live in the `dev` dependency group, so
  `uv sync` installs them. (#5)
- `jitterbug.__version__` and the API's reported version are read from package
  metadata instead of hard-coded strings. (#5)
- Code style is enforced by ruff (line length 100) instead of black, isort and flake8;
  type annotations use the Python 3.10 syntax (`X | None`, `list[str]`). (#4)
- The Docker image installs from `pyproject.toml`. (#5)

### Fixed

- A file with an unknown extension whose content is not text raises the documented
  `ValueError("Cannot infer format ...")` instead of leaking `UnicodeDecodeError`.
- `load_from_influxdb` computed epochs as `astype(int) / 1e9`, which assumes nanosecond
  timestamps; pandas 3 parses `_time` at microsecond resolution, so every epoch was
  1000× too small. The conversion is now resolution-independent. Found by running the
  mocked InfluxDB tests in CI, where the `influx` extra is absent (a stub module now
  stands in for it).
- `verbose: true` in a configuration file had no effect from the CLI: the command
  installs a logging handler before reading the file, and the analyzer's second
  `basicConfig` was a no-op. The analyzer now sets the `jitterbug` logger level.
- The two analysis notebooks in `examples/` run again: they imported `requests` (no
  longer a dependency), read fields that do not exist on the result models
  (`start_time`, `confidence_score`, `jitter_ratio`, `ks_statistic`), and their "REST
  API" sections exercised the server removed in this release. Executed end to end with
  `nbconvert` before committing.
- Change point timestamps are timezone-aware UTC, like the measurements they come from.
  They used to be naive local time, so `start_timestamp`/`end_timestamp` in results
  files and the CLI table depended on the machine's timezone (the epochs were always
  right). The CLI table columns are now labelled `Start (UTC)` / `End (UTC)`.
- `DataLoader.validate_data()` returns plain Python numbers and booleans (it used to
  return numpy scalars, which `json.dumps` rejects); CSV datasets record `source: csv`
  and the file path in their metadata; `.jsonl` is recognised as scamper JSON.
- `jitterbug analyze` and `jitterbug visualize` no longer overwrite the `algorithm`,
  `method`, `threshold` and `output_format` values of a `--config` file with the CLI's
  own defaults; a flag now overrides the file only when it is given explicitly. A wrong
  `--algorithm` value is rejected with Pydantic's message instead of failing later.
- `jitterbug visualize` works again: it now writes the five matplotlib figures through
  `JitterbugPlotter.save_all_plots` and prints the summary. It used to abort with
  `'CongestionInference' object has no attribute 'timestamp'` (#7). The
  `--static-only`/`--interactive-only` flags are gone; `--prefix` is new. When the
  analysis yields no inferences the confidence heatmap is an empty placeholder instead
  of an `imshow` error, so the command still writes its five files.
- Time axes in every plot use matplotlib's automatic date locator and concise formatter
  instead of one labelled tick per hour, which produced an unreadable axis on multi-day
  series.
- The Bayesian detector no longer swallows exceptions and returns "no change points";
  a failure is raised as `RuntimeError` with the cause attached.
- `JitterbugAnalyzer.analyze()` returns the same `metadata` keys (`total_measurements`,
  `min_intervals`, `change_points`) on its early-return paths as on the full path. (#3)
- Test suite: eight assertions that had drifted from the code, and five tests that
  failed during the last fifteen minutes of every hour. (#3)

### Removed

- The `influx` *file* format: `--format influx` and `.flux`/`.influx` files were routed to
  a placeholder that raised `NotImplementedError`. Loading from an InfluxDB server through
  `DataLoader.load_from_influxdb()` is unchanged. Format inference now fails with a clear
  message instead of guessing `influx` for anything it does not recognise.
- Dead code in `analysis/`: `LatencyJumpAnalyzer.analyze_detailed`,
  `CongestionInferenceAnalyzer._apply_inference_logic` and `_post_process_inferences`
  implemented alternative rules that nothing called.
- **The plotly dashboard and interactive modules** (`visualization/dashboard.py`,
  `visualization/interactive.py`), the `plotly` dependency, `examples/visualization_demo.py`
  and the generated `interactive_bcp_ks/timeline.html`. They had no tests, the dashboard
  crashed on every real result (#7), and the demo called a method that does not exist.
  `docs/VERSION_COMPARISON.md`, a 1.x-versus-2.x page that described all of the removed
  components as features, is gone too.
- **The experimental `torch`, `rbeast` and `adtk` detectors** and their extras. None of
  them was evaluated in the paper; `torch` was a heuristic rather than a trained model,
  and `rbeast`/`adtk` silently fell back to an internal statistical method when their
  package was missing while still labelling the output with the back end's name. The
  detectors are now `ruptures` (default) and `bcp`. `algorithms.py` shrinks from 1 215
  to 318 lines; `examples/interactive_algorithm_selector.py` and the stale plots for the
  removed detectors are gone, and `examples/network_analysis/plots/` was regenerated
  (BCP 14/15, ruptures 11/15, unchanged).
- **The REST API server and the Docker image.** `jitterbug.api`, the `api` extra, the
  `Dockerfile`, `docker-compose.yml`, the entrypoint script, `docs/DOCKER.md` and the two
  related examples are gone. The server had not been able to start since the 2.0
  rewrite (`create_app()` raised on import and `/analyze` called a method that did not
  exist, #6), had no tests, and the Docker image's only entrypoint was that server.
  Jitterbug is a library and a CLI; wrap it in your own service if you need HTTP.
- The DockerHub push, PyPI publish and Codecov upload jobs from CI; none of them had
  working credentials. (#4)
- The `print`-based smoke scripts `test_algorithms.py` and
  `examples/test_new_implementation.py`, replaced by real tests. (#5)

## [2.0.0] - 2025-07-19

Complete rewrite of the PAM 2022 framework: Pydantic models, five change point
detection back ends (ruptures, BCP, PyTorch, Rbeast, ADTK), CSV/JSON/InfluxDB input,
Typer CLI, optional REST API and visualization modules. See the
[pull request](https://github.com/estcarisimo/jitterbug/pull/2).

## [1.0.0] - 2024-03-18

Original implementation accompanying the paper *Jitterbug: A new framework for
jitter-based congestion inference* (PAM 2022).

[Unreleased]: https://github.com/estcarisimo/jitterbug/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/estcarisimo/jitterbug/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/estcarisimo/jitterbug/releases/tag/v1.0.0
