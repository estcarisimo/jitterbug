# Contributing to Jitterbug

Thanks for your interest in Jitterbug. This guide covers the development setup, the
checks every change must pass, and how a change gets from your machine into `main`.

## Development setup

Jitterbug uses [uv](https://github.com/astral-sh/uv) for everything: environments,
dependencies and building.

```bash
git clone git@github.com:estcarisimo/jitterbug.git
cd jitterbug
uv sync --extra visualization   # runtime + dev tools (the `dev` group is installed by default)
uv run pre-commit install       # git hooks: ruff lint/format, file hygiene
```

Add `--extra bcp` for the Bayesian back end used in the paper (a git dependency), or
`--extra all` for every optional back end. See `pyproject.toml` for the full list.

Run the same checks CI runs:

```bash
uv run ruff check src/ tests/ examples/ tools/
uv run ruff format --check src/ tests/ examples/ tools/
uv run mypy src/jitterbug            # blocking in CI
uv run pytest -m "not slow"          # a few seconds
uv run pytest                        # + the Bayesian regression tests (~2 min, needs --extra bcp)
uv run jitterbug analyze examples/network_analysis/data/raw.csv --output /tmp/results.json
uv build
```

`uv run pre-commit run --all-files` runs the lint and hygiene hooks on the whole tree.

## Project conventions

- Python 3.10+ code: `X | None` unions, `list[str]` generics, `match` is fine. Ruff's
  `UP` rules enforce the 3.10 target.
- Line length 100, ruff formatter, imports sorted by ruff (`I`). The temporary
  `ignore` entries in `[tool.ruff.lint]` each name a follow-up; remove an entry when
  its findings are fixed, and do not add new ones.
- `pathlib.Path` for paths, `logging` (module-level `logger = logging.getLogger(__name__)`)
  instead of `print` in library code. `print` and Rich output belong in `cli/` only.
- NumPy-style docstrings on public functions and classes.
- Configuration is Pydantic models in `src/jitterbug/models/config.py`. Add a field
  there and thread it through the CLI (`src/jitterbug/cli/main.py`) rather than adding
  ad-hoc parameters.
- Type hints on every function; mypy runs with `disallow_untyped_defs` and is blocking
  in CI.
- Tests are plain pytest functions with fixtures and `parametrize`, one file per
  module (`tests/test_<module>.py`). Use small synthetic RTT series; the bundled
  `examples/network_analysis/data/raw.csv` (47 163 rows, PAM 2022) is for smoke and
  regression tests only. Compare floats with `numpy.testing.assert_allclose`.
- `tests/test_paper_regression.py` pins the current output on the bundled dataset
  (golden counts) and checks agreement with the paper's reference intervals by overlap.
  If you change the numbers on purpose, update the goldens and explain in the changelog.
  The Bayesian half is marked `slow` and runs in its own CI job with the `bcp` extra.
- Every behaviour change gets a test under `tests/` and a line in `CHANGELOG.md` under
  `[Unreleased]`.

## Making a change

`main` is protected. Nobody pushes to it directly; every change lands through a pull
request that has passed CI **and** a Copilot code review. This is repository policy: a
PR without a Copilot review is not merged.

1. Create a branch from `main`: `git switch -c <type>/<short-name>` (`feat/`, `fix/`,
   `docs/`, `chore/`).
2. Commit in small, coherent steps. The pre-commit hooks run ruff on each commit.
3. Push and open a PR: `gh pr create --fill`. The PR template has a checklist.
4. **Iterate until green.** Two things must be true before merging:
   - CI is green: `lint`, every `test (...)` matrix leg, and `build`.
   - A Copilot code review has run on the latest push and **every comment has been
     addressed**: either fix it and push, or reply explaining why it is not applicable
     and resolve the thread. Do not merge with unresolved review threads.

   The ruleset requests Copilot automatically on PRs targeting `main`. For stacked PRs
   or if no request appears, request it yourself and re-request it after each push:

   ```bash
   gh api -X POST repos/estcarisimo/jitterbug/pulls/<n>/requested_reviewers \
     -f 'reviewers[]=copilot-pull-request-reviewer[bot]'   # or the Reviewers gear in the sidebar
   gh pr checks <n> --watch                                 # wait for CI
   gh pr view <n> --comments                                # read review comments
   ```

5. Merge (squash) once both are green. The branch is deleted automatically.

   Stacked PRs: merge the bottom of the stack **without** `--delete-branch`; GitHub
   then retargets the next PR to `main` on its own. Deleting the base branch first
   closes the stacked PR.

Repository admins can technically bypass the ruleset. Treat that as an emergency-only
escape hatch and say so in the PR when it is used.

## Releasing

1. Bump `version` in `pyproject.toml` (the only place the version lives;
   `jitterbug.__version__` reads it from package metadata).
2. Move the `[Unreleased]` section of `CHANGELOG.md` under a new `[X.Y.Z] - YYYY-MM-DD`
   heading and add the compare link at the bottom. Update `version` and
   `date-released` in `CITATION.cff`.
3. Open a PR with those changes and merge it.
4. Tag and publish a GitHub release: `git tag vX.Y.Z && git push origin vX.Y.Z`, then
   `gh release create vX.Y.Z --generate-notes`.

Jitterbug is not published on PyPI at the moment (the `jitterbug` name there belongs to
an unrelated project); installation is from GitHub. A PyPI release under a new name is
a separate, later decision.

## Reporting issues

Use the issue templates. For security problems follow `SECURITY.md` instead of opening
a public issue.
