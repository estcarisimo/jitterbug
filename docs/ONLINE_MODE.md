# Online mode (prototype)

Jitterbug's sequential pipeline is retrospective twice over: the offline Bayesian
detector sees the whole series, and the verdict for a period needs the *next* change
point to close it. `jitterbug.streaming` is a prototype that runs the same decision rule
one RTT sample at a time. It is not wired into the CLI or `JitterbugConfig` yet, and its
results are not pinned by the regression tests; the numbers below come from the replay
harness described at the end.

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

```python
from jitterbug.streaming import OnlineJitterbug, StreamingConfig

online = OnlineJitterbug(StreamingConfig())
for epoch, rtt in stream:  # seconds, milliseconds
    for event in online.push(epoch, rtt):
        if event.kind == "verdict":
            print(event.stage, event.start_epoch, event.is_congested)
```

## Results on the paper dataset

Replaying `examples/network_analysis/data/raw.csv` in timestamp order with the default
settings (MAP rule, expected run length 50 bins, at least 1 h between change points,
100 jitter samples before a provisional verdict), scored with the metric of
`tests/test_paper_regression.py` against the paper's KS reference:

| | Offline pipeline (BCP + KS) | Offline pipeline on growing prefixes | Online prototype |
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
period, *return* the one that closes it. Both tools print these numbers
(`onset_delay_min_*` and `return_delay_min_*` in the replay, the "First appearance"
block in the prefix experiment). The provisional verdict arrived 11 h before the final
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
uv run python tools/replay_online.py                # defaults; add --events out.json for every event
uv run python tools/replay_online.py --sweep       # decision rule x lag x hazard x threshold
uv run python tools/prefix_experiment.py --step-hours 1 --start-hours 24 --output prefix.json
uv run python tools/prefix_experiment.py --from-json prefix.json   # re-score without rerunning
```

`tools/prefix_experiment.py` runs the offline pipeline on growing prefixes of the dataset
(about 20 min for hourly cutoffs with BCP) and reports how often, and how long after the
fact, its verdicts change, plus the onset and return delays defined above; that is the
"rerun on a sliding window" baseline in the table.
