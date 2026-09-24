# Installation

The distribution is named `jitterbug-inference` on PyPI (the name `jitterbug` there
belongs to an unrelated project); the import is `import jitterbug` and the command is
`jitterbug`.

Requirements: Python 3.10 or newer.

## From PyPI

```bash
pip install jitterbug-inference                            # core
pip install "jitterbug-inference[bcp,visualization]"       # the paper's setup plus figures
uv tool install "jitterbug-inference[bcp]"                  # the CLI in its own environment
```

Pin a version for reproducible runs: `pip install "jitterbug-inference==2.1.1"`.

## From a clone with uv (recommended)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh      # once, if you do not have uv
git clone https://github.com/estcarisimo/jitterbug.git
cd jitterbug
uv sync                                               # core: ruptures detector, CLI, library
uv run jitterbug version
```

`uv sync` creates `.venv`, installs the package in editable mode and the `dev`
dependency group (pytest, ruff, mypy). Add extras as needed:

| Extra | Installs | Enables |
| --- | --- | --- |
| `bcp` | [bayesian-changepoint](https://pypi.org/project/bayesian-changepoint/) + torch | `--algorithm bcp`, the paper's detector |
| `visualization` | matplotlib | `jitterbug visualize`, `JitterbugPlotter` |
| `clustering` | scikit-learn | `--mode clustering`, the non-sequential mode |
| `influx` | influxdb-client | `DataLoader.load_from_influxdb` |
| `jupyter` | JupyterLab, ipykernel | the notebooks in `examples/` |
| `all` | all of the above | |

```bash
uv sync --extra bcp --extra visualization    # the paper's setup plus figures
uv sync --extra all
```

Either activate the environment (`source .venv/bin/activate`) or prefix commands with
`uv run`.

## From a clone with pip

```bash
git clone https://github.com/estcarisimo/jitterbug.git
cd jitterbug
python -m venv .venv && source .venv/bin/activate
pip install -e ".[visualization]"        # extras in brackets, as usual
```

## Directly from GitHub (unreleased changes)

```bash
pip install "jitterbug-inference[visualization] @ git+https://github.com/estcarisimo/jitterbug.git"
```

Or a tag: `...jitterbug.git@v2.1.1`.

## Notes on the `bcp` extra

- It installs `bayesian-changepoint` (>= 1.2) from PyPI, imported as
  `bayesian_changepoint_detection`. The older PyPI projects `bayescd` and
  `bayesian-changepoint-detection` are unmaintained releases of the same library that
  Jitterbug does not work with. On Linux, uv resolves `torch` from the CPU-only index
  configured in `pyproject.toml` (`[tool.uv.sources]`), which avoids a multi-gigabyte CUDA download.
- The detector runs on CPU by default (`change_point_detection.bcp_device: cpu`). On
  Apple Silicon the library would otherwise pick MPS, which is an order of magnitude
  slower for series of this size. The bundled dataset takes about a second on CPU.
- `--extra bayesian` is a deprecated alias of `--extra bcp` and will go in 3.0.

## Check the installation

```bash
uv run jitterbug version
uv run jitterbug analyze examples/network_analysis/data/raw.csv --summary-only
uv run jitterbug analyze examples/network_analysis/data/raw.csv --algorithm bcp --method ks_test   # bcp extra
uv run jitterbug visualize examples/network_analysis/data/raw.csv --output-dir plots           # visualization extra
uv run python -c "from jitterbug.detection import get_available_algorithms; print(get_available_algorithms())"
```

`get_available_algorithms()` lists only the detectors whose packages are importable.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `--algorithm bcp` fails with "bayesian_changepoint_detection package is required" | The extra is not installed: `uv sync --extra bcp`. |
| `bcp` runs for many minutes on a Mac | An older configuration file sets `bcp_device: mps`; use `cpu`. |
| `jitterbug visualize` reports that matplotlib is missing | `uv sync --extra visualization`. |
| Plots fail on a server without a display | Set `MPLBACKEND=Agg` before running. |
| `--output-format parquet` fails | Parquet needs `pyarrow` (`uv pip install pyarrow`). |
| `pip install jitterbug` installed something else | That is the unrelated PyPI project; uninstall it and use one of the commands above. |

## Developer setup

See [CONTRIBUTING.md](https://github.com/estcarisimo/jitterbug/blob/main/CONTRIBUTING.md): `uv sync --extra visualization`,
`uv run pre-commit install`, then `uv run pytest -m "not slow"`.
