# Runtime across releases

How long each Jitterbug release takes to analyze the PAM 2022 dataset
(`examples/network_analysis/data/raw.csv`, 47 163 RTT samples, 1 440 fifteen-minute
intervals), and where the difference comes from.

**In short:** with the Bayesian detector (BCP + KS, the paper's configuration), the
current release is **5.9× faster** than 1.0 as each was released (16.6 s → 2.8 s), with
the same 28 periods and 14 congested. The gain comes from the Bayesian change point
library (`bayesian-changepoint` 1.2), not from Jitterbug's own code: 1.0's code on today's
dependencies takes 2.6 s. With ruptures + KS, 2.1 and later are **1.5× faster** than 2.0
(2.05 s → 1.32 s), and that gain is Jitterbug's: the CSV loader went from 0.73 s to 0.11 s.
Jitterbug 2.0 with the BCP release it shipped against did not finish within an hour.

## Setup

- **Machine:** Apple M1 (8 cores), macOS 15.7, Python 3.12.11 for every release, default
  thread settings (no `*_NUM_THREADS` set). Measured on 2026-09-25.
- **Isolation:** one virtual environment per release and dependency set, built with `uv`.
- **Configurations:** `bcp_ks` (BCP change points, KS test; `jitterbug analyze -a bcp -m
  ks_test`, or `-c bcp -i ks` in 1.0) and `ruptures_ks` (ruptures, KS test; 2.x only, 1.0
  has no ruptures detector), plus `bcp_jd` and `ruptures_jd` with jitter dispersion
  instead of the KS test (`-m jitter_dispersion`, `-i jd` in 1.0). Default parameters of
  each release.
- **Dependency sets:**
    - *As released*: dependencies resolved as of the release date
      (`uv pip install --exclude-newer`), with the BCP implementation the release used:
      Hildensia's numpy BCP from git for 1.0, the first PyTorch rewrite (BCP 1.0, git) for
      2.0, `bayesian-changepoint` 1.2 from PyPI for 2.1 and later.
    - *Current*: every release on the same, current dependencies (NumPy 2.5, pandas 3.0,
      PyTorch 2.14, `bayesian-changepoint` 1.2.1). This separates changes in Jitterbug's
      code from changes in its dependencies.
- **Measurements:** *CLI* is the wall-clock time of the command in a fresh process
  (interpreter start-up, imports, analysis, writing the results file), 5 runs, median.
  *Stages* is the in-process time of each analysis step (import, load, change points,
  latency jumps, jitter, inference), 3 runs, median. Before the runs, each environment
  imports Jitterbug once to compile bytecode and the dataset is read once into the page
  cache.
- **Check:** every run's results are counted. All releases give 28 periods / 14 congested
  with BCP and 22 / 11 with ruptures, with either jitter method, so the timings compare
  the same analysis.

## Results

### BCP + KS, as released

| Release | Median (s) | Min–max (s) | Peak RSS (MiB) | Speedup vs 1.0.0 | Periods / congested |
|---|---:|---:|---:|---:|---:|
| 1.0.0 | 16.62 | 16.52–16.77 | 179 | 1.00× | 28 / 14 |
| 2.0.0 | > 3 600 | – | – | < 0.005× | did not finish |
| 2.1.0 | 2.85 | 2.78–8.55 | 470 | 5.83× | 28 / 14 |
| 2.1.1 | 2.81 | 2.80–8.81 | 472 | 5.91× | 28 / 14 |
| 2.2.0 | 2.82 | 2.76–8.70 | 473 | 5.89× | 28 / 14 |
| 2.3.0 | 2.95 | 2.76–8.84 | 470 | 5.64× | 28 / 14 |

| Release | import | load | change points | latency jumps | jitter | other |
|---|---:|---:|---:|---:|---:|---:|
| 1.0.0 | 0.61 | 0.013 | **15.51** | 0.000 | 0.14 | 0.000 |
| 2.1.0 | 0.72 | 0.12 | 0.79 | 0.001 | 0.13 | 0.056 |
| 2.3.0 | 0.78 | 0.12 | 0.80 | 0.001 | 0.13 | 0.056 |

Change point detection is 93% of 1.0's runtime; with `bayesian-changepoint` 1.2 it drops
from 15.5 s to 0.8 s (19×). What is left in 2.1+ is mostly fixed cost: the analysis stages
add up to about 1.1 s, and the other 1.7 s of the 2.8 s is Python start-up and imports
(PyTorch comes with the BCP back end).

Jitterbug 2.0 installed the first PyTorch rewrite of BCP (1.0, July 2025), whose
Student-t likelihood looped in Python over every point of every segment, which makes the
offline dynamic program cubic in the number of intervals (fixed in
`bayesian-changepoint` 1.2). On the 1 440 intervals of this dataset it ran for more than
an hour, so it is reported as a lower bound.

### BCP + KS, current dependencies

| Release | Median (s) | Min–max (s) | Peak RSS (MiB) | vs 1.0.0 | Periods / congested |
|---|---:|---:|---:|---:|---:|
| 1.0.0 | 2.61 | 2.43–2.66 | 410 | 1.00× | 28 / 14 |
| 2.0.0 | 3.39 | 3.37–9.47 | 474 | 0.77× | 28 / 14 |
| 2.1.0 | 2.75 | 2.73–8.85 | 471 | 0.95× | 28 / 14 |
| 2.1.1 | 2.89 | 2.73–8.85 | 473 | 0.90× | 28 / 14 |
| 2.2.0 | 2.73 | 2.70–8.72 | 471 | 0.96× | 28 / 14 |
| 2.3.0 | 2.79 | 2.72–8.52 | 469 | 0.94× | 28 / 14 |

On the same dependencies, Jitterbug's own code is not faster on this path: change point
detection takes the same 0.8 s in every release, and 2.x spends about 0.1 s more than 1.0
loading the data into its validated data models (0.12 s against 0.01 s). 2.0 is 0.8 s
slower because of its CSV loader (next section).

### ruptures + KS, as released

| Release | Median (s) | Min–max (s) | Peak RSS (MiB) | Speedup vs 2.0.0 | Periods / congested |
|---|---:|---:|---:|---:|---:|
| 2.0.0 | 2.05 | 2.01–3.17 | 285 | 1.00× | 22 / 11 |
| 2.1.0 | 1.34 | 1.30–2.49 | 262 | 1.53× | 22 / 11 |
| 2.1.1 | 1.33 | 1.31–2.49 | 261 | 1.54× | 22 / 11 |
| 2.2.0 | 1.34 | 1.30–2.47 | 259 | 1.53× | 22 / 11 |
| 2.3.0 | 1.32 | 1.31–2.44 | 256 | 1.55× | 22 / 11 |

| Release | import | load | change points | latency jumps | jitter | other |
|---|---:|---:|---:|---:|---:|---:|
| 2.0.0 | 0.62 | **0.73** | 0.058 | 0.001 | 0.12 | 0.060 |
| 2.1.0 | 0.69 | 0.11 | 0.056 | 0.001 | 0.12 | 0.056 |
| 2.3.0 | 0.71 | 0.11 | 0.055 | 0.001 | 0.12 | 0.056 |

The whole difference is loading the CSV (0.73 s → 0.11 s), a Jitterbug change in 2.1; the
current-dependency runs show the same 1.5× (1.91 s → 1.31 s). ruptures itself takes 0.06 s,
so with this detector the runtime is dominated by start-up and imports.

### Jitter dispersion instead of the KS test

The same picture with the other jitter method. The 2.x runs are 0.1–0.3 s shorter than
with the KS test: jitter dispersion takes 0.03 s where the KS test takes 0.13 s. The v2.0.0 row with BCP is
left out: its BCP did not finish with the KS test either, and the jitter method runs after
change point detection.

| Configuration | Release | Median (s) | Min–max (s) | Speedup | Periods / congested |
|---|---|---:|---:|---:|---:|
| BCP + JD, as released | 1.0.0 | 16.11 | 16.07–16.25 | 1.00× | 28 / 14 |
| | 2.1.0 | 2.73 | 2.60–3.19 | 5.90× | 28 / 14 |
| | 2.3.0 | 2.64 | 2.60–2.93 | 6.11× | 28 / 14 |
| BCP + JD, current dependencies | 1.0.0 | 2.38 | 2.31–2.55 | 1.00× | 28 / 14 |
| | 2.0.0 | 3.27 | 3.23–3.78 | 0.73× | 28 / 14 |
| | 2.3.0 | 2.60 | 2.58–3.09 | 0.91× | 28 / 14 |
| ruptures + JD, as released | 2.0.0 | 1.94 | 1.91–1.96 | 1.00× | 22 / 11 |
| | 2.1.0 | 1.20 | 1.18–1.25 | 1.61× | 22 / 11 |
| | 2.3.0 | 1.21 | 1.19–1.25 | 1.60× | 22 / 11 |

2.1.1 and 2.2.0 are within about 0.1 s of 2.1.0 and 2.3.0 in every configuration; all rows are
in the raw data.

## Limitations

- One machine, one dataset. The BCP cost grows faster than linearly with the number of
  intervals, so longer series widen the gap between BCP implementations.
- The first run in each PyTorch environment is about 6 s slower (the maximum column of the
  KS tables, which ran first) while macOS loads PyTorch's libraries from disk; the medians
  are not affected, and the jitter dispersion runs, done later, do not show it.
- Default thread settings, as a user would run it. PyTorch may use several cores; the
  numpy BCP of 1.0 is single-threaded.
- "As released" resolves dependencies by date on today's Python 3.12, not the exact
  environment of 2022 (1.0's `requirements.txt` pins NumPy 1.19, which has no wheels for
  Python 3.12 on Apple silicon). The versions used are listed in `environment.json`.

## Reproducing

The script and the raw measurements are in the repository:

```bash
uv run python tools/benchmark_versions.py setup --workdir /tmp/jb-bench --deps era current
uv run python tools/benchmark_versions.py run --workdir /tmp/jb-bench --deps era current \
    --repeats 5 --stage-repeats 3 --timeout 3600
uv run python tools/benchmark_versions.py report --workdir /tmp/jb-bench --deps era current
```

`examples/network_analysis/benchmarks/` holds `runtime_by_release.csv` (every run and
stage; the script writes it as `results.csv` in the work directory) and `environment.json`
(machine and package versions) from the measurements above.
