# Input formats

What Jitterbug expects from an RTT time series, how it is read, and what happens to
rows that do not fit. The rules are enforced once, when the data is loaded
(`DataLoader.load_from_dataframe` for CSV, DataFrames and InfluxDB; the scamper reader
applies the same bounds per response), so the analysis code can assume a clean series.

## The contract

One row per RTT sample, from one source to one destination.

| Column | Required | Type | Meaning |
| --- | --- | --- | --- |
| `epoch` | yes | number | Unix time in **seconds** (UTC). Fractions are fine. |
| `values` | yes | number | Round-trip time in **milliseconds**. `rtt_value`, `rtt` and `latency` are accepted as synonyms (first match wins, in that order). |
| `source` | no | string | Probe or source address; kept as metadata on each sample. |
| `destination` | no | string | Target address; kept as metadata on each sample. |

Other columns are ignored (their names are recorded in `dataset.metadata["original_columns"]`).

Rules, applied in this order:

1. **Missing column or non-numeric value → error.** A missing `epoch` or RTT column, or a
   cell such as `"12:00"` or `"timeout"` in either, raises `ValueError` naming the column
   and the first offending value. Fix the file; Jitterbug does not guess.
2. **Missing values are dropped.** A row whose `epoch` or RTT is empty (NaN, `None`, or an
   empty string) is skipped.
3. **Out-of-range RTTs are dropped.** RTT must be `> 0` and `<= 10 000` ms
   (`jitterbug.models.MAX_RTT_MS`). Zero or negative values usually encode a timeout or a
   probe error; values above ten seconds are treated the same way.
4. **Rows are sorted by `epoch`** if they are not already, with a stable sort (ties keep
   their file order).
5. **Duplicate epochs are kept.** Several samples at the same second are normal for
   high-rate probing; `jitterbug validate` flags them (and `DataLoader.validate_data`
   reports `unique_timestamps`).
6. **No rows left → error.**

Every drop and the sort are logged at `WARNING` level and counted in the dataset
metadata:

```python
dataset.metadata["dropped_rows"]    # {"missing": 2, "non_positive": 0, "too_large": 1}
dataset.metadata["sorted_on_load"]  # True if the rows were reordered
dataset.metadata["rtt_column"]      # which synonym was used
```

Sampling does **not** have to be regular. The first analysis stage takes the minimum RTT
per `minimum_interval_minutes` window (15 minutes by default), and change points are
detected on that series; gaps and bursts in the raw sampling only affect how many
samples fall in each window. `jitterbug validate` prints the average interval and the
largest gap so you can judge whether the window is appropriate.

## CSV

```csv
epoch,values
1512144010.0,63.86
1512144010.0,66.52
1512144020.0,85.2
```

Read with `pandas.read_csv` defaults: a header row, comma separated, `.` decimal point.
The bundled `examples/network_analysis/data/raw.csv` (47 163 rows, PAM 2022) is in this
form.

```bash
jitterbug validate rtts.csv --verbose
jitterbug analyze rtts.csv --output results.json
```

## scamper JSON

One JSON object per line, as written by `scamper` with `-O json` (also `.jsonl`). Only
`ping` records are used; each response with an `rtt` and a `tx` time becomes one sample,
with `src`/`dst` as `source`/`destination`:

```json
{"type":"ping","src":"192.168.1.1","dst":"8.8.8.8","responses":[{"rtt":1.712,"tx":{"sec":1752855461,"usec":719258}}]}
```

Records of other types, responses without an `rtt` (timeouts), and lines that are not
valid JSON are skipped (invalid lines with a warning). Responses whose `rtt` is outside
`(0, 10 000]` ms are dropped with a warning and counted in
`metadata["dropped_responses"]`. Samples are sorted by time. A file with no usable
response is an error.

## pandas DataFrame

```python
from jitterbug import JitterbugAnalyzer, JitterbugConfig

results = JitterbugAnalyzer(JitterbugConfig()).analyze_from_dataframe(df)
```

The frame follows the contract above. CSV and InfluxDB results go through this same
path, so anything that can be turned into a frame with `epoch` and `values` columns
(Parquet, a database query, a live probe) can be analyzed.

## InfluxDB

With the `influx` extra, `DataLoader.load_from_influxdb(url, token, org, bucket, query)`
runs a Flux query and maps `_time` to `epoch` (seconds, UTC, rounded to microseconds)
and `_value` (or `rtt`, `latency`, `values`) to the RTT column; the result then follows
the DataFrame contract. The token is used only for the connection and is never written to
configuration, logs or results.

## Format detection

`load_from_file` picks the reader from the extension (`.csv`; `.json`/`.jsonl`). For any
other extension it looks at the first line: a line starting with `{` is scamper JSON, a
line containing a comma is a CSV header. Anything else (including binary content) is a
`ValueError`; pass `file_format="csv"` or `"json"` explicitly to override.
