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
Jitterbug 2.0, with the first PyTorch release of the library, was the slow point: 118 s on
the CPU, with a different result (34 periods), and more than an hour on the Apple GPU (MPS),
which that library release picked by default; the current release is **about 40× faster**
than 2.0 on the CPU. [What changed in the Bayesian library](#what-changed-in-the-bayesian-library)
lists the work behind the BCP numbers.

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
- **Device:** every BCP run is on the CPU, except the row marked MPS. Jitterbug 2.0 did
  not choose a device, and BCP 1.0 then picked Apple's MPS when PyTorch reported it; its
  CPU rows come from `run --hide-mps`, which makes PyTorch report no MPS device. Since 2.1
  Jitterbug passes `bcp_device` (default `cpu`), and `bayesian-changepoint` 1.2 defaults to
  the CPU as well.
- **Check:** every run's results are counted. All releases give 28 periods / 14 congested
  with BCP and 22 / 11 with ruptures, with either jitter method, so the timings compare
  the same analysis. The exception is 2.0 with its own BCP release (34 periods), below.

## Results

### BCP + KS, as released

| Release | Median (s) | Min–max (s) | Peak RSS (MiB) | Speedup vs 1.0.0 | Periods / congested |
|---|---:|---:|---:|---:|---:|
| 1.0.0 | 16.62 | 16.52–16.77 | 179 | 1.00× | 28 / 14 |
| 2.0.0, CPU | 118 | 111–141 | 372 | 0.14× | 34 / 14 |
| 2.0.0, MPS (its default) | > 3 600 | – | – | < 0.005× | did not finish |
| 2.1.0 | 2.85 | 2.78–8.55 | 470 | 5.83× | 28 / 14 |
| 2.1.1 | 2.81 | 2.80–8.81 | 472 | 5.91× | 28 / 14 |
| 2.2.0 | 2.82 | 2.76–8.70 | 473 | 5.89× | 28 / 14 |
| 2.3.0 | 2.95 | 2.76–8.84 | 470 | 5.64× | 28 / 14 |

| Release | import | load | change points | latency jumps | jitter | other |
|---|---:|---:|---:|---:|---:|---:|
| 1.0.0 | 0.61 | 0.013 | **15.51** | 0.000 | 0.14 | 0.000 |
| 2.0.0, CPU | 0.76 | 0.80 | **110** | 0.001 | 0.11 | 0.062 |
| 2.1.0 | 0.72 | 0.12 | 0.79 | 0.001 | 0.13 | 0.056 |
| 2.3.0 | 0.78 | 0.12 | 0.80 | 0.001 | 0.13 | 0.056 |

Change point detection is 93% of 1.0's runtime; with `bayesian-changepoint` 1.2 it drops
from 15.5 s to 0.8 s (19×). What is left in 2.1+ is mostly fixed cost: the analysis stages
add up to about 1.1 s, and the other 1.7 s of the 2.8 s is Python start-up and imports
(PyTorch comes with the BCP back end).

Jitterbug 2.0 installed the first PyTorch rewrite of BCP (1.0, July 2025), whose
Student-t likelihood looped in Python over every point of every segment, which makes the
offline dynamic program cubic in the number of intervals, and ran in float32. On the CPU
its change point detection takes 110 s, seven times the NumPy original, and returns 34
periods instead of 28. That release also picked Apple's MPS device when available, and
there the same run did not finish within an hour (reported as a lower bound). Both were
fixed in the library (next section and [below](#what-changed-in-the-bayesian-library)).

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
with the KS test: jitter dispersion takes 0.03 s where the KS test takes 0.13 s. 2.0 with
its own BCP release was measured on the CPU only, since on MPS it had already run past an
hour with the KS test, and change point detection comes before the jitter method.

| Configuration | Release | Median (s) | Min–max (s) | Speedup | Periods / congested |
|---|---|---:|---:|---:|---:|
| BCP + JD, as released | 1.0.0 | 16.11 | 16.07–16.25 | 1.00× | 28 / 14 |
| | 2.0.0, CPU | 112 | 109–121 | 0.14× | 34 / 15 |
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

## What changed in the Bayesian library

The BCP speedup, and better BCP results, come from work on the Bayesian change point
library itself,
[hildensia/bayesian_changepoint_detection](https://github.com/hildensia/bayesian_changepoint_detection)
(PyPI: [`bayesian-changepoint`](https://pypi.org/project/bayesian-changepoint/), docs:
[estcarisimo.github.io/bayesian_changepoint_detection](https://estcarisimo.github.io/bayesian_changepoint_detection/)),
which Jitterbug uses through its `bcp` extra. Jitterbug uses the offline detector
(Fearnhead 2006) with the Student-t likelihood on the 1 440 minimum-RTT intervals.

| BCP version | Used by | BCP + KS on the paper dataset | Periods / congested |
|---|---|---|---:|
| 0.4, NumPy original | Jitterbug 1.0 | change points 15.5 s, CLI 16.6 s | 28 / 14 |
| 1.0.0, first PyTorch rewrite (July 2025) | Jitterbug 2.0, and `main` until [#31](https://github.com/estcarisimo/jitterbug/pull/31) (September 2026) | change points 110 s, CLI 118 s on the CPU; more than an hour on MPS | 34 / 14 |
| 1.2 (September 2026) | Jitterbug 2.1 and later | change points 0.8 s, CLI 2.8 s | 28 / 14 |

From 1.0.0 to 1.2 the change point stage went from 110 s to 0.8 s (**about 140×**) on the same
data and CPU. Until September 2026 Jitterbug's lock file pinned a commit one day after the
1.0.0 tag with the same detection code (`2e8b4f6`); the regression test pinned 34 / 14
then. Against the paper's reference output (`examples/network_analysis/expected_results/`,
15 congested intervals), 1.2 recovers 14 with no spurious detections, where 1.0 recovered
13 (Jitterbug [#31](https://github.com/estcarisimo/jitterbug/pull/31)). These are the
changes in the library that did it:

| BCP pull request | What it changed | Effect on Jitterbug |
|---|---|---|
| [#50](https://github.com/hildensia/bayesian_changepoint_detection/pull/50) (fixes [#47](https://github.com/hildensia/bayesian_changepoint_detection/issues/47)) | Exact closed-form Normal-Gamma segment likelihood from prefix sums, instead of a Python loop over every point of every segment (O(n³) scalar operations) under an approximate predictive; float64 recursion, falling back to the CPU on MPS. 70–155× faster on 250–1 000 points, and it finds changes the old code missed. | CLI from about 136 s to about 8 s when Jitterbug adopted it ([#31](https://github.com/estcarisimo/jitterbug/pull/31)); 34 → 28 periods, 13 → 14 of 15 reference intervals recovered; no more MPS runs. |
| [#76](https://github.com/hildensia/bayesian_changepoint_detection/pull/76) | Full sum over segment ends; `truncate` deprecated, since the inherited truncation rule could drop the dominant term. | Jitterbug stopped passing `truncate=-40`: same 28 change points, probabilities within 1e-12, same speed. |
| [#96](https://github.com/hildensia/bayesian_changepoint_detection/pull/96) (closes [#55](https://github.com/hildensia/bayesian_changepoint_detection/issues/55)) | Segment statistics exact for data far from zero (errors of up to 44 nats before). | RTTs around 60 ms are data far from zero. |
| [#100](https://github.com/hildensia/bayesian_changepoint_detection/pull/100) | Inputs keep float64 precision instead of being cast to float32. | Minimum RTTs reach the detector unrounded. |
| [#103](https://github.com/hildensia/bayesian_changepoint_detection/pull/103) | CPU by default, accelerators opt-in (the offline detector cannot use Apple's MPS, which lacks float64). | The default Jitterbug 2.0 relied on, now CPU; Jitterbug also passes `bcp_device: cpu` since 2.1. |
| [#106](https://github.com/hildensia/bayesian_changepoint_detection/pull/106) | Changepoint table built in O(J T²) instead of O(T³), with an early stop; up to 19× faster. | Part of the 0.8 s change point stage. |
| [#105](https://github.com/hildensia/bayesian_changepoint_detection/pull/105), [#107](https://github.com/hildensia/bayesian_changepoint_detection/pull/107) | Timings across BCP releases, and detection quality on the Turing Change Point Dataset. | The library-side counterpart of this page. |

The library's own benchmark
([Performance](https://github.com/hildensia/bayesian_changepoint_detection#-performance))
measures the offline Student-t detector on 1 000 synthetic points at 0.53 s for 1.2.0,
24 s for the NumPy original and 108 s for 1.0.0, which also misses changes there; on the
1 440 RTT intervals here, the same ordering: 0.8 s, 15.5 s and 110 s.

## Limitations

- The MPS result is specific to Apple silicon; on a machine with CUDA, BCP 1.0 would have
  picked that GPU instead, which was not measured.
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
uv run python tools/benchmark_versions.py run --workdir /tmp/jb-bench --deps era \
    --releases v2.0.0 --configs bcp_ks bcp_jd --hide-mps --timeout 3600   # the CPU rows
uv run python tools/benchmark_versions.py report --workdir /tmp/jb-bench
```

`examples/network_analysis/benchmarks/` holds `runtime_by_release.csv` (every run and
stage; the script writes it as `results.csv` in the work directory) and `environment.json`
(machine and package versions) from the measurements above.
