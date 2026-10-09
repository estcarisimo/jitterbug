# Online mode

Jitterbug's sequential pipeline is retrospective twice over: the offline Bayesian
detector sees the whole series, and the verdict for a period needs the *next* change
point to close it. The online mode (`jitterbug.streaming`, commands `jitterbug stream`
and `jitterbug replay`) runs the same decision rule one RTT sample at a time. It needs
the `bcp` extra. Its results on the paper dataset are pinned in
`tests/test_paper_regression.py` (`TestOnlineReplay`); the numbers below come from
`jitterbug replay` and the tools described at the end.

## How it works

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
   period plus the Kolmogorov-Smirnov test on the jitter so far) is emitted once the open
   period holds `min_period_samples` jitter samples. When the next change point closes
   the period, the *final* verdict is emitted with the whole period, which is exactly what
   the sequential pipeline computes. The congestion state carries over between final
   verdicts as in the sequential mode.

Only the KS method is supported online: it uses consecutive RTT differences and is
causal. The jitter-dispersion filters are centered windows and look ahead.

## Command line

`jitterbug stream` reads `epoch,rtt` lines (seconds, milliseconds; a header line is
skipped, so is any line that does not parse) from a file or standard input and prints one
JSON object per event: change points, provisional verdicts and final verdicts. `--follow`
keeps reading a file as it grows (a half-written line is held until its newline arrives);
`--events verdicts` or `--events change-points` filters the output. Samples must arrive in
time order; an older sample is dropped. When the input ends without `--follow`, the open
bin is closed, so `stream FILE` and `replay FILE` emit the same events.

```bash
my-probe | jitterbug stream --events verdicts
jitterbug stream rtts.csv --follow --output events.jsonl
```

`jitterbug replay` runs a recorded dataset (any input format `analyze` accepts) through
the same pipeline in time order, prints a summary and, with `--reference`, scores the
final verdicts against a `starts,ends,congestion` CSV such as the paper's
`expected_results`.

```bash
jitterbug replay examples/network_analysis/data/raw.csv \
  --reference examples/network_analysis/expected_results/kstest_inferences.csv \
  --output events.json
```

Both commands take `--config` and the most common knobs as flags: `--decision`,
`--hazard-lambda`, `--min-period-samples`, `--min-time-elapsed`.

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
  significance_level: 0.05         # shared
streaming:
  decision: map                    # map | lag | window
  hazard_lambda: 50                # expected run length, in bins
  min_time_elapsed: 3600           # seconds between change points
  min_period_samples: 100          # jitter samples before a provisional verdict
  max_run_length: 1000             # run lengths kept by the detector
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
settings (MAP rule, expected run length 50 bins, at least 1 h between change points,
100 jitter samples before a provisional verdict), scored with the metric of
`tests/test_paper_regression.py` against the paper's KS reference:

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

## Reproducing

```bash
uv sync --extra bcp
uv run jitterbug replay examples/network_analysis/data/raw.csv \
  --reference examples/network_analysis/expected_results/kstest_inferences.csv
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
