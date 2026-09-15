# Changelog

All notable changes to Jitterbug are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Continuous integration on GitHub Actions: ruff lint and format checks, mypy
  (advisory), `pip-audit` over the locked dependency set, tests on Python 3.10–3.13
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

- README rewritten in the project's canonical layout: features, quick start, usage,
  configuration, architecture, development, example output, citation. The table of
  contents, the duplicated installation sections and the `pip install jitterbug`
  instructions are gone; every command was run before being documented. The REST API,
  Docker and `visualize` command are not documented until they work (#6, #7).
- **Python 3.10 or newer is required** (was 3.8). (#4)
- All temporary ruff ignores are gone: `pathlib` everywhere (`Path.open`, `/`, `mkdir`,
  `iterdir`), `raise ... from` inside every `except`, collapsed nested conditions, no line
  over 100 characters. `tools/` scripts no longer patch `sys.path` to a directory that
  does not exist. Behaviour unchanged (CLI output on the bundled dataset is identical).
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

- `JitterbugAnalyzer.analyze()` returns the same `metadata` keys (`total_measurements`,
  `min_intervals`, `change_points`) on its early-return paths as on the full path. (#3)
- Test suite: eight assertions that had drifted from the code, and five tests that
  failed during the last fifteen minutes of every hour. (#3)

### Removed

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
