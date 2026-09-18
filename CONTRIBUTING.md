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
uv run pytest                        # + the Bayesian regression tests (~10 s more, needs --extra bcp)
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
request that has passed CI **and** an independent code review from a fresh session
(see `.github/REVIEW.md`). This is repository policy: a PR without an `APPROVE` on its
final commit is not merged.

1. Create a branch from `main`: `git switch -c <type>/<short-name>` (`feat/`, `fix/`,
   `docs/`, `chore/`).
2. Commit in small, coherent steps. The pre-commit hooks run ruff on each commit.
3. Push and open a PR: `gh pr create --fill`. The PR template has a checklist.
4. **Iterate until green.** Two things must be true before merging:
   - CI is green: `lint`, every `test (...)` matrix leg, and `build`.
   - A reviewer with no context from your session — a new AI agent session given only
     `.github/REVIEW.md` and the PR number, or a human — has returned `VERDICT: APPROVE`
     for the **latest** push. Every finding before that is either fixed or rebutted with
     evidence in the PR; after each push, start a new reviewer session (never reuse one).

   ```bash
   gh pr checks <n> --watch          # wait for CI
   gh pr view <n> --comments         # every round's verdict is posted here in full
   ```

5. Merge (squash) once both are green. The branch is deleted automatically.

   Stacked PRs: merge the bottom of the stack **without** `--delete-branch`; GitHub
   then retargets the next PR to `main` on its own. Deleting the base branch first
   closes the stacked PR. If the retargeted PR shows as `DIRTY`, see the stacked-PR
   step in `AGENTS.md` for the safe way to update it.

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

Jitterbug is not published on PyPI yet: the `jitterbug` name there belongs to an
unrelated project, so the distribution is named `jitterbug-inference` (the import
package and the command stay `jitterbug`). The first PyPI upload waits until the
Bayesian back end (`bcp` extra, today a git dependency) is itself on PyPI; until then
installation is from GitHub.

## Reporting issues

Use the issue templates. For security problems follow `SECURITY.md` instead of opening
a public issue.
