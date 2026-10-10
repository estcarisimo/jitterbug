# Online mode

Jitterbug's sequential pipeline is retrospective twice over: the offline Bayesian
detector sees the whole series, and the verdict for a period needs the *next* change
point to close it. The online mode (`jitterbug.streaming`, commands `jitterbug stream`
and `jitterbug replay`) runs the same decision rule one RTT sample at a time. Its default
back end needs the `bcp` extra; the sliding-window back end runs with any detector. The
results on the paper dataset are pinned in `tests/test_paper_regression.py`
(`TestOnlineReplay`, `TestOnlineDispersionReplay`, `TestSlidingWindowReplay`); the numbers
below come from
`jitterbug replay` and the tools described at the end.

## How it works

Two back ends share the same events and the same decision rule; `streaming.backend`
picks one. The default, `bocpd`, is the incremental detector described here. The other,
`window`, reruns the offline sequential pipeline on a trailing window every few bins and
is described under [Back ends compared](#back-ends-compared).

1. **Causal binning.** A minimum-RTT bin (15 min by default) closes when a sample from a
   later bin arrives.
2. **Online change point detection.** Each closed bin feeds
   `bayesian_changepoint_detection.streaming.OnlineChangepointDetector` (Adams & MacKay
   2007), from the same `bcp` extra the offline detector uses. A change point is
   declared when the most probable run length drops (`decision: map`, the default). Two
   fixed-delay rules are available for comparison: `lag` thresholds the posterior that a
   segment started exactly `lag` bins ago, and `window` thresholds the posterior that it
   started within the last `lag` bins.
3. **Two-stage verdicts.** The stream start opens the baseline period. When a change
   point opens a new period, a *provisional* verdict (latency jump against the previous
   period plus the configured jitter test on the observations so far) is emitted once the
   open period holds `min_period_samples` jitter samples and, with dispersion, 12
   dispersion values (see below). When the next change point closes the period, the
   *final* verdict is emitted with the whole period: with the KS test it is exactly what
   the sequential pipeline computes; with dispersion it uses the causal series described
   below, the offline one delayed by 6 bins. The congestion state carries over between
   final verdicts as in the sequential mode.

Both jitter methods work online, and `jitter_analysis.method` picks one as in the
sequential mode (the config default is `jitter_dispersion`; the results below say which
method each row used). The KS test uses consecutive RTT differences and is causal as it
stands. Jitter dispersion uses the trailing version of the offline filters: the same
moving IQR and moving average, each window ending at the bin it describes instead of
being centered on it. The values are the offline ones delayed by 6 bins (1.5 h at the
default orders), so the threshold keeps its meaning, but a dispersion value reflects the
new period alone only after that delay. A provisional verdict with dispersion therefore
waits until the open period holds 12 dispersion values (the delay plus one averaging
window, 3 h); with 6 values, 11 of 31 provisional verdicts flipped on the paper dataset,
with 12 none.

## Command line

`jitterbug stream` reads `epoch,rtt` lines (seconds, milliseconds; a header line is
skipped, so is any line that does not parse) from a file or standard input and prints one
JSON object per event: change points, provisional verdicts and final verdicts. `--follow`
keeps reading a file as it grows (a half-written line is held until its newline arrives);
`--events verdicts` or `--events change-points` filters the output. Samples must arrive in
time order; an older sample is dropped. When the input ends (or on Ctrl-C with
`--follow`), the open bin is closed and a summary line goes to standard error, so
`stream FILE` and `replay FILE` emit the same events.

```bash
my-probe | jitterbug stream --events verdicts
jitterbug stream rtts.csv --follow --output events.jsonl
```

`jitterbug replay` runs a recorded dataset (any input format `analyze` accepts) through
the same pipeline in time order, prints a summary and, with `--reference`, scores the
final verdicts against a `starts,ends,congestion` CSV such as the paper's
`expected_results`.

```bash
jitterbug replay examples/network_analysis/data/raw.csv --method ks_test \
  --reference examples/network_analysis/expected_results/kstest_inferences.csv \
  --output events.json
```

Both commands take `--config` and the most common knobs as flags: `--backend`,
`--method` (`jitter_dispersion` or `ks_test`), `--decision`, `--hazard-lambda`,
`--min-period-samples`, `--min-time-elapsed`.

## Configuration

The online settings live in the `streaming` section of `JitterbugConfig`
(`StreamingConfig`). The bin width, latency jump threshold, significance level and
device come from the shared sections, so the online and sequential modes make the same
decision on the same data.

```yaml
data_processing:
  minimum_interval_minutes: 15     # bin width, shared with the sequential mode
latency_jump:
  threshold: 0.5                   # ms, shared
jitter_analysis:
  method: jitter_dispersion        # jitter_dispersion | ks_test, shared
  threshold: 0.25                  # dispersion rise, ms, shared
  significance_level: 0.05         # KS test, shared
streaming:
  backend: bocpd                   # bocpd (incremental) | window (offline rerun)
  decision: map                    # map | lag | window   (bocpd)
  hazard_lambda: 50                # expected run length, in bins   (bocpd)
  max_run_length: 1000             # run lengths kept by the detector   (bocpd)
  window_hours: 72                 # trailing window   (window)
  rerun_every_bins: 1              # closed bins between reruns   (window)
  stable_runs: 2                   # identical reruns before emitting   (window)
  min_time_elapsed: 3600           # seconds between change points
  min_period_samples: 100          # jitter samples before a provisional verdict
  min_ks_statistic: 0.0            # effect-size guard for the KS test
```

## Python API

```python
from jitterbug.models import JitterbugConfig
from jitterbug.streaming import OnlineJitterbug

online = OnlineJitterbug(JitterbugConfig())
for epoch, rtt in stream:  # seconds, milliseconds
    for event in online.push(epoch, rtt):
        if event.kind == "verdict":
            print(event.stage, event.start_epoch, event.is_congested)
```

`jitterbug.streaming.replay(dataset, config)` and `score(events, reference)` are what
`jitterbug replay` calls.

## Results on the paper dataset

Replaying `examples/network_analysis/data/raw.csv` in timestamp order with the default
online settings (MAP rule, expected run length 50 bins, at least 1 h between change
points, 100 jitter samples before a provisional verdict) and the KS test, scored with the
metric of `tests/test_paper_regression.py` against the paper's KS reference:

| | Offline pipeline (BCP + KS) | Offline pipeline on growing prefixes | Online mode |
|---|---|---|---|
| Periods / congested | 28 / 14 | same boundaries at every cutoff | 31 / 15 |
| Reference congested periods recovered | 14 of 15 | 14 of 15 | 14 of 15 |
| Spurious congested periods | 0 | 0 | 0 |
| Congestion onset first reported | n/a | 1.0 h median, 8.5 h max (2.25 h without the first period) | 15 min median, 90 min max |
| Return to baseline first reported | n/a | 2.0 h median, 13.75 h max | 2.25 h median, 4.25 h max |
| Verdict changes | n/a | 3.9 % for periods under 6 h old, 2.4 % at 6-12 h, 0.3 % at 12-24 h, none older | 0 of 31 provisional verdicts flipped |

Delays are measured from the boundary to the moment it is first reported: the sample
that triggers the change point online, or the first hourly cutoff whose prefix run places
a boundary within one bin of it. *Onset* is the change point that opens a congested
period, *return* the one that closes it. `jitterbug replay` and the tools print
these numbers (`onset_delay_min_*` and `return_delay_min_*` in the replay, the "First
appearance" block in the prefix experiment). The provisional verdict arrived 11 h before the final
one on average and never disagreed with it.

Two findings from the parameter sweep:

- The fixed-delay rules recover 5 to 12 (`lag`) or 5 to 15 (`window`) of the 15
  reference periods depending on the setting, but every one of the 45 settings tried
  produces 3 to 6 spurious congested periods, against 0 for every MAP setting, and
  they place 5 to 12 (`lag`) or 5 to 14 (`window`) of the 30 reference boundaries
  within 30 min, against 20 for MAP. The posterior mass of a change spreads over
  several run lengths, so thresholding one run length, or a short window of them,
  either misses boundaries or fires on noise.
- The MAP rule refines a boundary a few bins after first reporting it. With 30 min
  between change points those refinements become duplicate boundaries and tiny periods
  (38 change points instead of 33); 1 h absorbs them. Thirty jitter samples make the
  provisional KS p-value noisy (2 flips); 100 samples, about three bins, give none.

Caveats: one path over 15 days, and the metric is agreement with the offline method, not
ground truth. The stream's first period is taken as the baseline.

## Back ends compared

The `window` back end is the obvious alternative to a new detector: every `rerun_every_bins`
closed bins, run the sequential pipeline (the configured detector and jitter method, so it
also works with `ruptures` and with jitter dispersion) on the trailing `window_hours` of
samples. A change point is emitted once the detector has placed one within two bins of the
same spot in `stable_runs` consecutive reruns; change points come out in time order and are
never retracted. Final verdicts form a contiguous chain whose boundaries are the emitted
change points: the next one is emitted once the offline pipeline has judged the period at
the last emitted boundary identically (same verdict, boundaries within two bins) in
`stable_runs` consecutive reruns and the change point that closes it has been emitted. If
the detector moves a boundary by a bin or two between reruns, the emitted one stands and the
verdict of the offline period that contains it is used; if the window moves past a period
before it stabilizes, that period is skipped with a warning and the chain resumes at the
next change point. Because change points come out in time order, one that only stabilizes
after a later one was emitted is dropped; the chain then waits for the next emitted
boundary, up to a window length, before it skips. The window start opens the baseline
period, as the stream start does in the incremental back end; once the window has moved past
the stream start, anything touching its left edge is ignored. The open period gets a
provisional verdict from the same two-period rule with the configured jitter method
(trailing filters for dispersion; `jitter_method` on each event says which). On every event,
`n_prev` and `n_curr` count the observations the jitter test used: raw jitter samples for
the KS test, dispersion values for dispersion. The prefix experiment above showed why this
works: the offline boundaries never move and verdicts rarely flip.

BCP on the paper dataset, each row scored against the paper's reference for its jitter
method (`jitterbug replay` on the full series; offline, BCP + KS gives 28 periods / 14
congested, 14 of 15 recovered, 0 spurious, and BCP + dispersion the same counts against
its own reference):

| Back end | Jitter method | Periods / congested | Reference periods recovered | Spurious | Boundaries within 30 min | Onset delay, median / max | Return delay, median / max | Provisional flips | Replay wall time |
|---|---|---|---|---|---|---|---|---|---|
| Incremental Bayesian (`bocpd`, MAP rule) | KS test | 31 / 15 | 14 of 15 | 0 | 20 of 30 | 15 min / 90 min | 135 min / 255 min | 0 of 31 | 2 s |
| Sliding window, rerun every bin (`window`) | KS test | 31 / 14 | 14 of 15 | 0 | 30 of 30 | 45 min / 120 min | 113 min / 180 min | 2 of 31 | 160 s |
| Sliding window, rerun every 4 bins | KS test | 30 / 15 | 14 of 15 | 1 | 30 of 30 | 120 min / 225 min | 180 min / 360 min | 1 of 30 | 41 s |
| Incremental Bayesian (`bocpd`, MAP rule) | dispersion (causal) | 31 / 15 | 14 of 15 | 0 | 20 of 30 | 15 min / 90 min | 135 min / 255 min | 0 of 31 | 1 s |
| Sliding window, rerun every 4 bins | dispersion (causal) | 30 / 15 | 14 of 15 | 1 | 30 of 30 | 120 min / 225 min | 180 min / 360 min | 1 of 30 | 34 s |

Both back ends recover the same 14 of 15 reference periods, with either jitter method. The incremental detector
reports onsets within one bin at the median (90 min at worst), sooner than the sliding
window, whose delay is bounded below by `stable_runs × rerun_every_bins` bins. The sliding window places every
reference boundary within 30 min, because it sees the whole window when it decides, but
where the detector moves a boundary between reruns it can produce a short spurious period
next to a real one (the 4-bin row has one of 1.75 h). Rerunning every bin costs about
0.1 s per bin on this data (BCP on 72 h), cheap for one path and not for thousands; the
incremental detector costs a fraction of a millisecond per bin. The table is produced by
`tools/compare_online_backends.py`; the 4-bin window row is pinned in
`tests/test_paper_regression.py` (`TestSlidingWindowReplay`).

## Reproducing

```bash
uv sync --extra bcp
uv run jitterbug replay examples/network_analysis/data/raw.csv --method ks_test \
  --reference examples/network_analysis/expected_results/kstest_inferences.csv
uv run jitterbug replay examples/network_analysis/data/raw.csv \
  --reference examples/network_analysis/expected_results/jd_inferences.csv   # dispersion
uv run jitterbug replay examples/network_analysis/data/raw.csv --backend window --method ks_test \
  --reference examples/network_analysis/expected_results/kstest_inferences.csv
uv run python tools/compare_online_backends.py   # the back ends table, about 4 min
uv run python tools/replay_online.py --sweep       # decision rule x lag x hazard x threshold
uv run python tools/replay_online.py --min-time-elapsed 1800   # the 38-change-point figure
uv run python tools/replay_online.py --min-period-samples 30   # the 2-flip figure
uv run python tools/prefix_experiment.py --step-hours 1 --start-hours 24 --output prefix.json
uv run python tools/prefix_experiment.py --from-json prefix.json   # re-score without rerunning
```

`tools/prefix_experiment.py` runs the offline pipeline on growing prefixes of the dataset
(about 20 min for hourly cutoffs with BCP) and reports how often, and how long after the
fact, its verdicts change, plus the onset and return delays defined above; that is the
"rerun on a sliding window" baseline in the table.
