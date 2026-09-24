# Changelog

All notable changes to Jitterbug are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Non-sequential (clustering) analysis mode: `analysis_mode: clustering` /
  `jitterbug analyze --mode clustering`. Minimum-RTT intervals are clustered by
  (minimum RTT, jitter IQR) with a Gaussian mixture (components chosen by BIC), k-means,
  or k-means with the silhouette-best k (`clustering.algorithm`, `--clustering-algorithm`);
  every cluster is compared with the lowest-latency one by latency jump and a KS test on
  raw jitter, wherever its intervals fall in time; a temporal smoothing
  (`clustering.min_period_intervals`, default 2) merges the per-interval verdicts into
  periods. Output is the usual `CongestionInferenceResult`, with a `clustering` summary
  in the metadata. On the PAM 2022 dataset the GMM recovers 15 of the 15 reference
  congestion periods; its 2 extra periods are at the two ends of the data, where the
  minimum RTT is elevated but the sequential reference cannot give a verdict (sequential
  BCP + KS: 14/15, 0 extra). New
  `clustering` extra (scikit-learn), included in `all`; guide in
  `docs/CLUSTERING_MODE.md`. The sequential mode stays the default and is unchanged.

- Documentation site built with MkDocs (Material theme): the guides in `docs/`, an API
  reference generated from the docstrings, and the README, changelog, contributing
  guide and citation included from the root files. CI builds it with `--strict` and
  validates `CITATION.cff` on every PR; `.github/workflows/pages.yml` publishes it to
  GitHub Pages from `main`. New `docs` dependency group.
- The documentation is online at <https://estcarisimo.github.io/jitterbug/>: README badge,
  the `Documentation` project URL (shown on PyPI from the next release) and the
  repository website point to it.

### Changed

- American English is now a documented convention (`AGENTS.md`, `CONTRIBUTING.md`,
  the review brief). Existing British spellings in docs, comments, the changelog and
  one test name (`test_unrecognizable_content_is_an_error`) were corrected; no public
  function, method, option or message changed.
- README: the extras table lists `jupyter`.

## [2.1.1] - 2026-09-23

### Added

- `.github/workflows/publish.yml`: publishing a GitHub release uploads the tag's sdist
  and wheel to PyPI with Trusted Publishing (OpenID Connect, no stored token), after
  checking that the version matches the tag and smoke-testing the wheel, then attaches
  both files to the release. A manual run uploads an existing tag to TestPyPI or PyPI.

### Changed

- README links and the figure use absolute GitHub URLs, so they work on the PyPI
  project page.
- First release on PyPI: `pip install jitterbug-inference`. README (with a PyPI badge),
  `docs/INSTALLATION.md`, `AGENTS.md` install instructions updated accordingly.
- `CODE_OF_CONDUCT.md`: incidents are reported privately through GitHub (maintainer
  profile, the repository's private reporting form, GitHub's *Report content*); the
  document no longer lists an email address.

## [2.1.0] - 2026-09-23

Maintenance release: the project is back to a maintainable state, with CI, a
protected `main`, tests, and only the components that work. Jitterbug is now a
library plus a CLI with two change point detectors (`ruptures` and the paper's
Bayesian `bcp`, now installed from PyPI as `bayesian-changepoint`); the REST API,
Docker image, experimental detectors and plotly dashboards described in 2.0.0 are
gone (see *Removed*). Python 3.10 or newer is required. The distribution is renamed
`jitterbug-inference`; installation is from GitHub, there is no PyPI release yet.

### Added

- CLI tests for `validate` (metrics table, `--verbose`, contract violations, missing
  file), `config --template` (YAML/JSON to stdout and round trip through a file) and the
  error paths of `analyze` (bad input, unknown method), completing the CLI coverage item
  of the roadmap.
- `docs/INPUT_FORMATS.md`: the input contract (columns, units, ordering, what is dropped
  and why) for CSV, scamper JSON, DataFrames and InfluxDB, linked from the README.
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

- The `bcp` extra now installs the Bayesian library from PyPI as
  **`bayesian-changepoint>=1.2`** (the import name `bayesian_changepoint_detection` is
  unchanged), replacing the unmaintained upstream release and the git dependency used
  during development. `git` is no longer needed to install any extra, and a wheel of
  Jitterbug no longer carries a direct URL dependency. The new release fixes the offline
  segment likelihoods and the detection performance: on the PAM 2022 dataset the
  paper's configuration (`--algorithm bcp --method ks_test`) runs in about four seconds
  end to end (under one second of detection) instead of two minutes, and yields
  28 periods / 14 congested (was 34 / 14), recovering 14 of the paper's 15 congested
  intervals (was 13) with no spurious detections. Golden values, the example plots and
  summaries, and the timing notes in the docs are updated. The detector no longer
  passes `truncate=-40`, deprecated in 1.2; the exact sum is as fast and matches the
  truncated one to 1e-12 on the paper dataset.
- The distribution is now named **`jitterbug-inference`** (`[project] name`); the
  import package and the `jitterbug` command are unchanged. `jitterbug` on PyPI belongs
  to an unrelated project, so this is the name a future PyPI release will use.
  `jitterbug.__version__` reads the metadata of the new distribution name.
- `docs/INSTALLATION.md` rewritten: install from a clone with uv or pip, or directly
  from GitHub with `pip install "jitterbug-inference[...] @ git+..."`; the extras table;
  notes on the `bcp` extra; verification commands; a troubleshooting table of symptoms
  that actually occur. It and `docs/ALGORITHM_USAGE.md` said `pip install jitterbug[...]`,
  which installs the unrelated project.

- README: `jitterbug visualize` is back in *Usage* (it was left out while broken, #7)
  and `docs/VISUALIZATION_USAGE.md` is in the documentation table; coverage figure
  refreshed.
- Build backend requirement raised to `uv_build>=0.12.14,<0.13` (Dependabot). Building
  from source with an older `uv` CLI still works: it fetches the backend from PyPI.
- `examples/README.md` and `examples/network_analysis/README.md` rewritten to match the
  directory: setup with `uv sync --extra ...`, the script and the notebooks as the two
  entry points, the dataset described (dates, sizes, what the paper found and what
  Jitterbug recovers), and the reference CSV layout explained. The example scripts no
  longer patch `sys.path`; `results/` and the benchmark outputs are ignored by git.
- Code review policy: PRs are reviewed by an independent, fresh-context session
  following `.github/REVIEW.md` (today a Claude Sonnet subagent) instead of GitHub
  Copilot; merge requires `APPROVE` on the final commit. `AGENTS.md`, `CONTRIBUTING.md`
  and the PR template updated.
- Input validation happens once, at the edge, in `DataLoader.load_from_dataframe`
  (CSV and InfluxDB go through it too): rows with a missing epoch or RTT, a non-positive
  RTT, or an RTT above `MAX_RTT_MS` (10 s) are dropped with a warning and counted in
  `metadata["dropped_rows"]`; unsorted rows are sorted (stable) with a warning instead of
  being rejected; a non-numeric cell is a `ValueError` naming the column and value. Before,
  the first bad row surfaced as a Pydantic error from inside the loading loop. The loop
  itself is vectorized: the bundled dataset loads in 0.1 s instead of 1 s.
- The scamper JSON reader applies the same RTT bounds per response: a `ping` response
  with an RTT of 0 (a timeout) or above 10 s no longer makes the whole file fail with a
  Pydantic error; it is dropped with a warning and counted in
  `metadata["dropped_responses"]`.

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
  scripts no longer patch `sys.path` to a directory that does not exist. Behavior
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
  right). The CLI table columns are now labeled `Start (UTC)` / `End (UTC)`.
- `DataLoader.validate_data()` returns plain Python numbers and booleans (it used to
  return numpy scalars, which `json.dumps` rejects); CSV datasets record `source: csv`
  and the file path in their metadata; `.jsonl` is recognized as scamper JSON.
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
  instead of one labeled tick per hour, which produced an unreadable axis on multi-day
  series.
- The Bayesian detector no longer swallows exceptions and returns "no change points";
  a failure is raised as `RuntimeError` with the cause attached.
- `JitterbugAnalyzer.analyze()` returns the same `metadata` keys (`total_measurements`,
  `min_intervals`, `change_points`) on its early-return paths as on the full path. (#3)
- Test suite: eight assertions that had drifted from the code, and five tests that
  failed during the last fifteen minutes of every hour. (#3)

### Removed

- `examples/basic_analysis.py` (synthetic two-hour series on which nothing is detected)
  and `examples/output_formats_demo.py` (600 lines, crashed on a PyYAML argument, still
  used Pydantic v1 `.dict()`). `examples/network_analysis/basic_analysis.py` is the
  script example.
- The `influx` *file* format: `--format influx` and `.flux`/`.influx` files were routed to
  a placeholder that raised `NotImplementedError`. Loading from an InfluxDB server through
  `DataLoader.load_from_influxdb()` is unchanged. Format inference now fails with a clear
  message instead of guessing `influx` for anything it does not recognize.
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
  package was missing while still labeling the output with the back end's name. The
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

[Unreleased]: https://github.com/estcarisimo/jitterbug/compare/v2.1.1...HEAD
[2.1.1]: https://github.com/estcarisimo/jitterbug/compare/v2.1.0...v2.1.1
[2.1.0]: https://github.com/estcarisimo/jitterbug/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/estcarisimo/jitterbug/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/estcarisimo/jitterbug/releases/tag/v1.0.0
