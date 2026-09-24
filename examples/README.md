# Examples

Everything here runs on the PAM 2022 dataset bundled in `network_analysis/data/`. Two ways
in: a script, or the notebooks.

```
examples/
├── README.md
├── config_example.yaml               # every option, with comments
├── algorithm_benchmark.py            # ruptures vs BCP on synthetic series
├── jitter_dispersion_analysis.ipynb  # notebook: the jitter-dispersion method, step by step
├── kolmogorov_smirnov_analysis.ipynb # notebook: the KS-test method, step by step
└── network_analysis/
    ├── README.md                     # the dataset and what the paper found on it
    ├── basic_analysis.py             # library usage end to end, writes results/
    ├── data/
    │   ├── raw.csv                   # 47 163 RTT samples, 15 days (input contract: docs/INPUT_FORMATS.md)
    │   └── mins.csv                  # minimum RTT per 15-minute interval (for reference; Jitterbug recomputes it)
    ├── expected_results/
    │   ├── jd_inferences.csv         # the paper's congestion periods, jitter dispersion
    │   └── kstest_inferences.csv     # the paper's congestion periods, KS test
    ├── plots/                        # figures and summaries for both detectors (tools/generate_visualizations.py)
    └── results/                      # written by basic_analysis.py, not committed
```

## Setup

From the repository root:

```bash
uv sync --extra visualization          # ruptures detector, matplotlib
uv sync --extra bcp --extra visualization   # + the Bayesian detector (bayesian-changepoint, torch)
uv sync --extra jupyter --extra visualization   # + JupyterLab for the notebooks
```

## The script

```bash
uv run python examples/network_analysis/basic_analysis.py
```

Loads `raw.csv`, runs the default pipeline (ruptures + jitter dispersion), prints the
congested periods and writes `results/analysis_results.json` and
`results/congestion_summary.csv`. The same thing from the CLI:

```bash
uv run jitterbug analyze examples/network_analysis/data/raw.csv --output results.json
uv run jitterbug analyze examples/network_analysis/data/raw.csv --algorithm bcp --method ks_test
uv run jitterbug visualize examples/network_analysis/data/raw.csv --output-dir plots
```

## The notebooks

```bash
uv run jupyter lab examples/
```

Each notebook loads the data, runs one jitter method, plots the raw RTT, the minimum-RTT
series and the detected periods, and compares the result with `expected_results/`. They
run headless too (`jupyter nbconvert --execute`), which is how they are checked before a
release.

## Comparing the detectors

```bash
uv run python examples/algorithm_benchmark.py
```

Times `ruptures` and `bcp` (if installed) on synthetic series with known change points
(step changes, gradual drift, noise, a longer "real-world-like" one) and prints detection
counts, run times and a recommendation; writes `benchmark_results.csv` and
`benchmark_report.html` next to the script (not committed).

## What to expect

With `--algorithm bcp --method ks_test` (the paper's configuration) Jitterbug finds
28 periods, 14 of them congested, recovering 14 of the paper's 15 with no spurious
detection; the default `ruptures` + jitter dispersion finds 22 periods, 11 congested
(11 of 15). `tests/test_paper_regression.py` pins both. Interval boundaries differ from
the paper's 1.x implementation by a few minutes; the counts and the overlap are what is
checked. Analysis takes a few seconds with `ruptures`.

## Related documentation

- [docs/INPUT_FORMATS.md](../docs/INPUT_FORMATS.md) — what an input file must contain
- [docs/ALGORITHM_SELECTION_GUIDE.md](../docs/ALGORITHM_SELECTION_GUIDE.md) — when to use which detector
- [docs/VISUALIZATION_USAGE.md](../docs/VISUALIZATION_USAGE.md) — the figures and `JitterbugPlotter`
