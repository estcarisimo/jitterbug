# Non-sequential (clustering) mode

The method of the PAM 2022 paper is **sequential**: change point detection splits the
minimum-RTT series into periods, and each period is compared with the one right before
it. The **clustering** mode drops the time order. Every minimum-RTT interval is described
by its latency and its jitter, the intervals are clustered, and each cluster is compared
with the uncongested baseline, whether or not their intervals are adjacent in time. A
quiet Tuesday morning and a quiet Sunday night end up in the same cluster, and so do the
evening peaks of different days.

It needs the `clustering` extra (scikit-learn):

```bash
pip install "jitterbug-inference[clustering]"      # or: uv sync --extra clustering
```

## How it works

1. **Intervals and features.** The raw RTTs are binned exactly like the sequential mode's
   minimum-RTT intervals (`data_processing.minimum_interval_minutes`, 15 min by default).
   Each interval gets two features: its **minimum RTT** and the **interquartile range of
   its jitter samples** (jitter = difference of consecutive RTTs). Intervals with fewer
   than two jitter samples are skipped. The features are standardized.
2. **Clustering** with one of three algorithms (`clustering.algorithm`):

    | Algorithm | Number of clusters |
    | --- | --- |
    | `gmm` (default) | Gaussian mixture, 1..`max_clusters` components, chosen by BIC. One component means "no structure". |
    | `kmeans` | k-means with `n_clusters` clusters (default 2). |
    | `kmeans_silhouette` | k-means, 2..`max_clusters` clusters, chosen by silhouette score. |

3. **Baseline.** Clusters are numbered by increasing median minimum RTT; cluster `0`,
   the lowest, is the baseline.
4. **Verdict per cluster**, the paper's two signals against the baseline:
    - *latency jump*: the cluster's median minimum RTT exceeds the baseline's by more
      than `latency_jump.threshold` (0.5 ms), **and**
    - *jitter change*: a two-sample Kolmogorov–Smirnov test between the pooled raw jitter
      samples of the cluster and of the baseline is significant at
      `jitter_analysis.significance_level` (0.05).
5. **Periods.** Each interval takes its cluster's verdict. A temporal smoothing
   (`clustering.min_period_intervals`, default 2) first fills gaps of up to that many
   non-congested intervals inside congestion, then drops congested runs of up to that
   many intervals. Consecutive intervals with the same verdict become one period, so the
   output has the same shape as the sequential mode (`CongestionInferenceResult`), and
   exporters and plots work unchanged. A gap in the data always ends a period.

## Usage

```bash
jitterbug analyze examples/network_analysis/data/raw.csv --mode clustering
jitterbug analyze rtts.csv --mode clustering --clustering-algorithm kmeans --output results.json
jitterbug visualize rtts.csv --mode clustering --output-dir plots
```

```yaml
analysis_mode: clustering        # sequential (default) | clustering
clustering:
  algorithm: gmm                 # gmm | kmeans | kmeans_silhouette
  n_clusters: 2                  # kmeans only
  max_clusters: 6                # gmm and kmeans_silhouette
  min_period_intervals: 2        # temporal smoothing; 0 = raw per-interval verdicts
  random_state: 0
```

```python
from jitterbug import JitterbugAnalyzer, JitterbugConfig

config = JitterbugConfig(analysis_mode="clustering")
config.clustering.algorithm = "gmm"
result = JitterbugAnalyzer(config).analyze_from_file("rtts.csv")

for cluster in result.metadata["clustering"]["clusters"]:
    print(cluster["cluster"], cluster["size"], cluster["latency_jump"], cluster["is_congested"])
```

`result.metadata["clustering"]` holds the algorithm, the number of clusters, the
model-selection scores (BIC or silhouette per number of clusters) and, for every cluster,
its size, median minimum RTT, median jitter IQR, latency jump, KS statistic and p-value,
and verdict. Each period's `latency_jump` and `jitter_analysis` carry the statistics of
the cluster that describes it, and the `confidence` of a congested period is the share of
its intervals that were congested before smoothing.

## Results on the PAM 2022 dataset

Against the paper's 15 reference congestion periods (`expected_results/kstest_inferences.csv`),
with the same overlap criterion as the sequential regression tests:

| Configuration | Periods | Congested | Recovered | Spurious |
| --- | --- | --- | --- | --- |
| `gmm` (BIC → 5 clusters), smoothing 2 | 31 | 16 | **15/15** | 2 |
| `kmeans` (2 clusters), smoothing 2 | 31 | 16 | 14/15 | 2 |
| `kmeans_silhouette` (→ 2 clusters), smoothing 2 | 31 | 16 | 14/15 | 2 |
| `gmm`, no smoothing | 61 | 31 | 15/15 | 17 |
| `kmeans`, no smoothing | 101 | 51 | 9/15 | 8 |
| *Sequential, BCP + KS test (the paper)* | *28* | *14* | *14/15* | *0* |

Each clustering run takes under a second. The first three rows are pinned in
`tests/test_paper_regression.py`. The two spurious periods are at the edges of the data;
see *Limitations*.

## Limitations

- **The KS test rarely discriminates.** Clusters pool thousands of jitter samples, so
  the test is significant for almost any difference: on the paper dataset every
  non-baseline cluster has p < 1e-49, including one only 0.01 ms above the baseline.
  In practice the latency jump decides; the KS statistic in the metadata is a better
  indicator of how different a cluster's jitter really is.
- **The latency threshold is small.** 0.5 ms suits the sequential comparison of adjacent
  periods; here a cluster 0.86 ms above the baseline (114 intervals on the paper dataset)
  counts as congested, next to the clear 20+ ms clusters. Raise
  `latency_jump.threshold` if short, shallow episodes are not of interest.
- **Time is ignored until the smoothing step.** Without smoothing a single noisy interval
  can split a congestion period or create a spurious one, which is why the default
  window is two intervals (30 minutes).
- **BIC can over-split.** The features are not Gaussian, so the mixture may use several
  components for one regime (5 on the paper dataset). This is harmless for the verdict
  as long as the extra components fall on the same side of the latency threshold.
- **"Spurious" periods at the edges of the data.** Both periods counted as spurious on
  the paper dataset are at its ends: the first 4.5 h (from 12-01 16:00, when the data
  starts) and the last 8.5 h (until 12-16 16:00, when it ends). The minimum RTT is
  elevated (20–45 ms over a ~9 ms floor) in both, but the sequential reference cannot
  label them, because its first period has nothing before it to compare with and its last
  one has no closing change point. Not being tied to adjacent periods, the clustering
  mode labels them. They are counted as spurious above only because the reference has no
  verdict there.
- The sequential mode remains the reference method of the paper and the default.
