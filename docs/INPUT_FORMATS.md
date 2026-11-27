# Input formats

What Jitterbug expects from an RTT time series, how it is read, and what happens to
rows that do not fit.

Loading has three steps:

1. A **reader** turns a file into a table of raw rows. There is one reader per format
   (CSV, TSV, JSON lines, scamper, RIPE Atlas, Parquet, Arrow, ping and fping output),
   chosen by extension or content.
2. The **normalizer** (`jitterbug.io.normalize`) applies the `input` settings to those
   rows: column mapping, then units, then row policies. Files, pandas DataFrames and
   InfluxDB results all go through it, so the rules below are enforced in one place.
3. The result is an `RTTDataset`: Unix-second times, millisecond RTTs, in time order.

Everything is configured in the `input` section of the configuration file
(`InputConfig`). The defaults reproduce the contract of earlier versions exactly, so an
`epoch,values` CSV needs no settings at all.

## Columns

One row per RTT sample, from one source to one destination.

| Setting | Default | Meaning |
| --- | --- | --- |
| `time_column` | `epoch` | Sample time, in `time_unit`. |
| `rtt_column` | none | Round-trip time, in `rtt_unit`. When unset, the first of `values`, `rtt_value`, `rtt` and `latency` that exists is used. |
| `source_column` | `source` | Optional; kept as metadata on each sample. |
| `destination_column` | `destination` | Optional; kept as metadata on each sample. |

Other columns are ignored (their names are recorded in `dataset.metadata["original_columns"]`).
A missing time or RTT column is an error that lists the columns that do exist.

## Units

| Setting | Values | Default |
| --- | --- | --- |
| `time_unit` | `s`, `ms`, `us`, `ns` (Unix epoch), `iso`, `auto` | `s` |
| `rtt_unit` | `ms`, `us`, `s`, `ns` | `ms` |

Times become Unix seconds and RTTs become milliseconds. `iso` reads ISO 8601 strings
such as `2024-01-01T00:00:00Z`, `2024-01-01T02:00:00+02:00` or `2024-01-01 00:00:00.25`;
values without an offset are **UTC**, values with one are converted to UTC. A datetime
column of a DataFrame is accepted the same way.

`auto` infers the time unit:

- datetime values, or strings that are not numbers, are `iso`;
- numbers are judged by the median of their absolute values: below 1e11 is seconds,
  below 1e14 milliseconds, below 1e17 microseconds, anything larger nanoseconds.

The thresholds work because the units are a factor of 1000 apart: Unix seconds stay
below 1e11 until the year 5138, while Unix milliseconds have been above 1e11 since 1973.
Small relative times (seconds since the start of a capture) also come out as seconds.
The inferred unit is logged at `INFO` level and stored in `metadata["time_unit"]`, with
`metadata["time_unit_inferred"]` set to `true`. If you know the unit, set it: `auto` is a
guess, and a file of millisecond offsets since the start of a capture (small numbers)
would be read as seconds.

A value that cannot be read in its unit (`"12:00"` in a numeric column, `"yesterday"`
with `iso`, numbers with `iso`) is an error naming the column and the first offending
value. Jitterbug does not guess per row. So is a time that is infinite or falls outside the years
1–9999 once converted (usually a sign of the wrong `time_unit`).

## Row policies

Applied in this order, after the units are converted:

| Step | Setting | Default | Alternatives |
| --- | --- | --- | --- |
| Missing time or RTT (NaN, `None`, empty string) | – | dropped | – |
| RTT `<= 0` or `> 10 000` ms (`MAX_RTT_MS`) | `out_of_range` | `drop` | `error` |
| Rows out of time order | `unsorted` | `sort` (stable: ties keep file order) | `error` |
| Rows sharing a timestamp | `duplicates` | `keep` | `first`, `min`, `error` |

- Zero or negative RTTs usually encode a timeout or a probe error; values above ten
  seconds are treated the same way.
- `duplicates: first` keeps the first row of each timestamp in file order, `min` the
  one with the lowest RTT (the first on ties). Several samples at the same second are
  normal for high-rate probing, which is why they are kept by default;
  `jitterbug validate` flags them.
- No rows left is an error.
- **Nothing is imputed or resampled.** Sampling does not have to be regular: the first
  analysis stage takes the minimum RTT per `minimum_interval_minutes` window (15 minutes
  by default), so gaps and bursts only change how many samples fall in each window.
  `jitterbug validate` prints the average interval and the largest gap.

Every drop, the sort and any removed duplicates are logged at `WARNING` level and
recorded in the dataset metadata:

```python
dataset.metadata["dropped_rows"]        # {"missing": 2, "non_positive": 0, "too_large": 1}
dataset.metadata["sorted_on_load"]      # True if the rows were reordered
dataset.metadata["duplicate_rows"]      # rows sharing a timestamp with an earlier row
dataset.metadata["duplicates_removed"]  # rows removed by duplicates: first | min
dataset.metadata["rtt_column"]          # the RTT column that was used
dataset.metadata["time_unit"]           # the time unit that was applied
```

## Configuration

```yaml
input:
  time_column: ts
  time_unit: ms          # s | ms | us | ns | iso | auto
  rtt_column: rtt_us
  rtt_unit: us           # ms | us | s | ns
  delimiter: null        # CSV separator; null detects , tab or ;
  duplicates: keep       # keep | first | min | error
  out_of_range: drop     # drop | error
  unsorted: sort         # sort | error
```

The column and unit settings are also command-line flags of `analyze`, `validate`,
`visualize` and `replay`; a flag overrides the configuration file:

```bash
jitterbug analyze rtts.tsv --time-column ts --time-unit ms --rtt-column rtt_us --rtt-unit us
jitterbug validate export.csv --time-column timestamp --time-unit iso
jitterbug validate rtts.csv --config config.yaml    # validate reads the input section too
```

In Python, pass the settings through `JitterbugConfig(input=InputConfig(...))` or
`DataLoader(InputConfig(...))`.

## Several series in one file

A file often holds more than one series: pings from one source to several destinations,
several probes, several devices. Jitterbug's analysis assumes **one** path, so these
must be separated. Two settings select series on the raw rows, before the row policies
run:

| Setting | Flag | Meaning |
| --- | --- | --- |
| `group_by` | `--group-by` (repeatable or comma-separated) | Columns whose values identify a series; each distinct combination is analyzed on its own. |
| `series` | `--series NAME=VALUE` (repeatable) | Keep only the rows where the column equals the value. |

In both, `source` and `destination` name the mapped `source_column` and
`destination_column` (for scamper, `src` and `dst`); any other name is a raw input
column, such as a probe id: `prb_id` and `msm_id` for RIPE Atlas, a Parquet or Arrow
column, or `destination` for the hosts of a ping or fping log. Values are compared as text; a numeric column with gaps
(read as floats) is written without the `.0` (`probe=1234`), and a number given as a
filter value (`probe: 1234` in YAML) is read as the same text.

```yaml
input:
  group_by: [source, destination]
  series:
    destination: 8.8.8.8
```

```bash
jitterbug analyze pings.json --group-by destination --output results.json
jitterbug analyze rtts.csv --group-by probe_id --output results.csv --output-format csv
jitterbug analyze pings.json --series destination=8.8.8.8          # one series, usual output
jitterbug validate pings.json --series source=10.0.0.1 --series destination=8.8.8.8
```

- **Filters first, then groups, then the row policies per series.** Duplicates,
  sorting and drops are counted and applied within each series, so two destinations
  probed at the same second are not duplicates of each other. `time_unit: auto` is
  inferred once from all the selected rows.
- **A filter that matches nothing** is an error that lists the values the column has.
  A missing key column is an error that lists the columns.
- **A series that cannot be analyzed** (every row dropped, fewer than two
  `minimum_interval_minutes` intervals, an analysis error) is reported as skipped with
  its reason; the other series are analyzed as usual. A row policy set to `error` that
  any series violates fails the whole run.
- **One-series commands.** `validate`, `visualize` and `replay` take `--series` but not
  `--group-by`. If the configuration sets `group_by` and the rows hold several series,
  they exit with an error that lists the keys; add `--series` to pick one.
- **Mixed input.** With neither setting, a file whose rows hold more than one path is
  still analyzed as one series, as in earlier versions, but a `WARNING` names how many
  paths there are and the `--group-by` that separates them, and
  `metadata["source_destination_pairs"]` counts the distinct (source, destination)
  pairs. A path is a distinct (source, destination) pair, or for RIPE Atlas a (`prb_id`,
  `msm_id`, `destination`) combination.

### Grouped output

Without `group_by` (with or without `series`), the output files are exactly those of a
single-series run. With `group_by`, the console prints one row per series (samples,
periods, congested periods, congestion ratio, or why it was skipped), and the files are:

**JSON**: one document. Each `result` is what a single-series run on that series'
rows writes; a skipped series has a `skipped_reason` instead.

```json
{
  "group_by": ["destination"],
  "series": [
    {"key": {"destination": "8.8.8.8"}, "result": {"inferences": [...], "metadata": {...}}},
    {"key": {"destination": "9.9.9.9"}, "skipped_reason": "1 minimum-RTT interval(s) of 15 min, need at least 2"}
  ],
  "metadata": {"series_count": 2, "analyzed": 1, "skipped": 1, "series_filter": {}}
}
```

Series appear in order of first appearance in the file; series whose rows were all
dropped while loading come last. A missing key value is `null`.

**CSV and Parquet**: every period of every analyzed series, with the key columns first:

```csv
destination,starts,ends,congestion
8.8.8.8,1512216010.0,1512243010.0,True
8.8.8.8,1512243010.0,1512301509.0,False
1.1.1.1,1512207010.0,1512220509.0,True
```

Skipped series and series without periods have no rows. A key column may not be named
`starts`, `ends` or `congestion`.

In Python:

```python
from jitterbug import JitterbugAnalyzer, JitterbugConfig
from jitterbug.io import DataLoader
from jitterbug.models import InputConfig

config = JitterbugConfig(input=InputConfig(group_by=["destination"]))
analyzer = JitterbugAnalyzer(config)
results = analyzer.analyze_series_from_file("pings.json")   # or analyze_series_from_dataframe
for entry in results.series:
    print(entry.key, entry.skipped_reason or len(entry.result.inferences))
analyzer.save_series_results(results, "results.json")

datasets = DataLoader(config.input).load_series("pings.json")  # {key: RTTDataset}
```

A key is a tuple of `(name, value)` pairs in `group_by` order (`SeriesKey`), e.g.
`(("destination", "8.8.8.8"),)`; `entry.key` and `dataset.metadata["series_key"]` hold it as
a `{name: value}` mapping. The single-series methods (`load_from_file`,
`analyze_from_file`, `analyze_from_dataframe`) keep their return types; they apply
`series` and refuse input that `group_by` splits into several series.

## Inspecting a file

`jitterbug inspect` is a dry run of loading: it reads the file with the same reader,
mapping, units and policies as `analyze`, and reports what would happen, without
analyzing anything.

```bash
jitterbug inspect pings.json
jitterbug inspect rtts.csv --time-column ts --time-unit auto --group-by probe_id
jitterbug inspect rtts.csv --config config.yaml --json > report.json
```

It shows:

- the detected format and compression, and reader details (CSV separator, invalid JSON
  lines);
- the columns found and the mapping in effect: time and RTT columns and units,
  including the unit `time_unit: auto` inferred and the RTT synonym that was picked, and
  the settings a format fixes (scamper);
- row counts: read, selected by `series`, missing, non-positive, too large, valid; and
  for each row policy its setting, the rows it concerns and its effect (dropped,
  sorted, kept, or an error);
- the series: under `group_by`, or the distinct (source, destination) pairs when
  `group_by` is empty (with a reminder that they would be analyzed as one series), each
  with its sample count, time range, median interval and largest gap between distinct
  timestamps, and duplicate timestamps.

It takes the input flags of `analyze` (`--format`, `--time-column`, `--rtt-column`,
`--time-unit`, `--rtt-unit`, `--group-by`, `--series`) and `--config` (only the `input`
section is used). `--json` prints the report as JSON (`jitterbug.io.inspection.inspect_file`
returns the same dictionary). It exits with status 1 when loading would fail (a missing
column, an unreadable value, a policy set to `error` that the data violates), after
printing the report with the reason.

**`inspect` or `validate`?** Use `inspect` first, on a file you do not know: it explains
how the file will be read and which series it holds, and it works when loading would
fail. Use `validate` on the one series you are about to analyze: it loads it for real and
reports its data quality (ordering, duplicates, intervals, gaps, RTT statistics and
outliers).

## Formats

| Format | Extensions | Detected from content | Notes |
| --- | --- | --- | --- |
| `csv` | `.csv` | header with `,` or `;` | Separator detected from the header |
| `tsv` | `.tsv`, `.tab` | header with a tab | Tab-separated |
| `json` | `.json`, `.jsonl`, `.ndjson` | line starting with `{`, `[{` or a lone `[` | JSON lines or one JSON array: scamper output, RIPE Atlas results or generic JSONL, decided by content |
| `scamper` | – | – | Forces the scamper reader |
| `jsonl` | – | – | Forces the generic JSONL reader |
| `atlas` | – | – | Forces the RIPE Atlas reader |
| `parquet` | `.parquet`, `.pq` | magic bytes `PAR1` | Needs the `parquet` extra |
| `arrow` | `.arrow`, `.feather`, `.ipc`, `.arrows` | magic bytes (`ARROW1`, `FEA1`, stream marker) | Arrow IPC file or stream, Feather; needs the `parquet` extra |
| `ping` | – | `PING host ...` header or a reply line | `ping -D` output |
| `fping` | – | an fping reply or timeout line | `fping -D` output |

`--format` (`file_format=` in Python) overrides detection. Without it:

1. the extension decides (ignoring a compression suffix);
2. otherwise the first bytes of the file: Parquet and Arrow are recognized by their magic
   bytes, also inside a `.gz`, `.bz2`, `.xz` or `.zst` file (which is then rejected, see
   below), and binary content is never decoded as text;
3. otherwise the first non-empty line, tried in this order: a `PING host ...` header or a
   ping reply line is `ping`; an fping reply, timeout or duplicate line (`host : [0],
   ...`) is `fping`; a comma or semicolon makes it CSV, a tab TSV (unless the line looks
   like JSON); a line starting with `{`, `[{` or a lone `[` is JSON. A ping line such as
   `[1700000000.1] 64 bytes from ...` starts with a bracket but is not JSON.

Anything else is an error. `.txt` and `.log` files have no registered extension, so they
are always detected by content.

### CSV and TSV

```csv
epoch,values
1512144010.0,63.86
1512144010.0,66.52
1512144020.0,85.2
```

A header row is required. For `csv` the separator is the one of comma, tab and semicolon
that occurs most often in the header (comma on ties, and for a one-column header); `tsv`
always uses a tab. Set `input.delimiter` to force another one-character separator. The
decimal point must be `.`: a semicolon-separated file with decimal commas (`20,5`) gives
a non-numeric error. The bundled `examples/network_analysis/data/raw.csv` (47 163 rows,
PAM 2022) is a comma-separated file in the default schema.

### Generic JSON lines

One flat JSON object per line; the top-level fields are the columns, mapped through the
`input` settings like CSV columns. Records may have different fields (absent ones are
missing values); lines that are not valid JSON are skipped with a warning and counted in
`metadata["invalid_lines"]`; a line that is valid JSON but not an object is an error.
A file holding one JSON array of such objects (`[{...}, {...}]`, on one line or many) is
read the same way.

```json
{"ts": "2024-01-01T00:00:00Z", "rtt": 20.5, "source": "probe-1"}
{"ts": "2024-01-01T00:01:00Z", "rtt": 21.0, "source": "probe-1"}
```

```bash
jitterbug analyze rtts.jsonl --time-column ts --time-unit iso
```

### scamper JSON

One JSON object per line, as written by `scamper -O json` or `sc_warts2json`. Only
`ping` records are used; each response with an `rtt` (milliseconds) and a `tx` time
becomes one sample, with `src`/`dst` as `source`/`destination`:

```json
{"type":"ping","src":"192.168.1.1","dst":"8.8.8.8","responses":[{"rtt":1.712,"tx":{"sec":1752855461,"usec":719258}}]}
```

Records of other types, responses without an `rtt` (timeouts), and lines that are not
valid JSON are skipped (invalid lines with a warning). scamper's schema is fixed, so the
column and unit settings do not apply to it; the row policies do. Responses whose `rtt` is
outside `(0, 10 000]` ms are dropped and counted in `metadata["dropped_responses"]`. A
file with no usable response is an error.

**scamper, RIPE Atlas or generic JSONL?** The `json` format reads the records (one per
line, or the elements of a JSON array) and decides, in this order:

1. **scamper** when at least one record is a scamper ping (`"type": "ping"` with a
   `responses` field) or one of scamper's bookkeeping records (`"type"` `list`,
   `cycle-def`, `cycle-start` or `cycle-stop`);
2. **RIPE Atlas** when at least one record has the `prb_id`, `msm_id` and `result` fields
   that every Atlas result carries, whatever its type;
3. **generic JSONL** otherwise.

A `"type": "ping"` field on its own is not enough, since scamper, RIPE Atlas and other
tools all write it. Use `--format scamper`, `--format atlas` or `--format jsonl` to skip
the check. A file whose first character is `[` is parsed as one JSON array (so a
pretty-printed array spanning many lines works); its elements are the records.

### RIPE Atlas ping results

What the RIPE Atlas API returns for a ping measurement
(`https://atlas.ripe.net/api/v2/measurements/<id>/results/`), a JSON array, or the same
results one per line:

```json
[{"fw":5080,"type":"ping","prb_id":6001,"msm_id":1001,"timestamp":1700000000,"from":"193.0.0.1","src_addr":"192.168.1.10","dst_addr":"8.8.8.8","dst_name":"dns.google","result":[{"rtt":12.5},{"x":"*"},{"rtt":12.7}]}]
```

```bash
curl -s "https://atlas.ripe.net/api/v2/measurements/1001/results/?probe_ids=6001&start=1700000000&stop=1700086400&format=json" > atlas.json
jitterbug analyze atlas.json
```

Only `"type": "ping"` results are used; other measurement types (and records that are not
Atlas results) are skipped and counted in `metadata["skipped_records"]`. Every entry of a
result's `result` list with an `rtt` (milliseconds) becomes one sample at the result's
`timestamp` (Unix seconds), so the packets of one result share a timestamp; they are kept
by default, and `duplicates: min` keeps the lowest RTT of each result. The source is
`from` (the probe's public address), or `src_addr` when `from` is empty; the destination
is `dst_addr`, or `dst_name`. `prb_id`, `msm_id` and `src_addr` are kept as columns of the
raw frame, so they work as `group_by` and `series` keys. Results of several probes,
measurements or destinations in one file are one series unless you separate them, which
mixes different paths; a warning says so and `metadata["series"]` counts the (probe,
measurement, destination) combinations. Analyze each path with
`--group-by prb_id,msm_id,destination` (or just `--group-by prb_id` for one measurement),
or keep one with `--series prb_id=6001`; `jitterbug inspect atlas.json` lists the paths
(see [Several series in one file](#several-series-in-one-file)).

Entries without an `rtt` are skipped and counted: `{"x": "*"}` in `metadata["timeouts"]`,
`{"error": ...}` (and anything else) in `metadata["errors"]`; duplicate replies (`"dup"`)
in `metadata["duplicate_replies"]`. Replies outside `(0, 10 000]` ms are dropped and
counted in `metadata["dropped_responses"]`, as for scamper. The schema is fixed, so the
column and unit settings do not apply; the row policies do. A file with no ping result,
or with no reply at all, is an error.

### Parquet and Arrow

Columnar files are read with pyarrow, which comes with the `parquet` extra
(`pip install "jitterbug-inference[parquet]"`); without it, reading one raises an
`ImportError` that names the extra. `parquet` reads Apache Parquet; `arrow` reads the Arrow
IPC file format (Feather version 2), Feather version 1 and the Arrow IPC stream format.

The file's columns are the raw columns, mapped through the `input` settings like CSV
columns. A time column with a timestamp type is read as datetimes whatever `time_unit` says
(naive timestamps are UTC, zoned ones are converted), so a frame saved with
`df.to_parquet()` loads without `--time-unit`. A numeric time column follows `time_unit`
as usual.

```python
df.to_parquet("rtts.parquet")              # epoch: datetime64, values: float
```

```bash
jitterbug analyze rtts.parquet
jitterbug analyze export.feather --time-column ts --rtt-column rtt_us --rtt-unit us
```

Both formats compress their own data, so `rtts.parquet.gz` and the like are rejected with
an error that says to drop the suffix; so is a `data.gz` whose content is Parquet or Arrow.

### ping and fping output

The text that `ping` and `fping` print, one reply per line. Each reply line needs a Unix
timestamp in brackets, which `ping -D` (Linux iputils) and `fping -D` print:

```text
PING dns.google (8.8.8.8) 56(84) bytes of data.
[1700000000.100000] 64 bytes from dns.google (8.8.8.8): icmp_seq=1 ttl=117 time=12.3 ms
[1700000001.100000] 64 bytes from dns.google (8.8.8.8): icmp_seq=2 ttl=117 time=11.9 ms
```

```text
[1700000000.10000] 8.8.8.8 : [0], 64 bytes, 12.3 ms (12.3 avg, 0% loss)
[1700000001.10000] 8.8.8.8 : [1], timed out (12.3 avg, 50% loss)
```

```bash
ping -D -i 5 example.net > ping.txt         # stop with Ctrl-C
fping -D -l -p 5000 example.net > fping.log
jitterbug analyze ping.txt
jitterbug analyze fping.log
```

Supported line shapes:

- `ping`: `64 bytes from HOST: icmp_seq=N ttl=T time=R ms`, as printed by Linux iputils,
  BSD and macOS (whose `icmp_seq` starts at 0), and BusyBox (`seq=N`); `HOST` may be
  `name (address)`, in which case the address is the destination. IPv6 addresses work.
- `fping`: `HOST : [N], B bytes, R ms (...)`, as printed per probe by `fping -l`, `-c` and
  `-C`.

**One path per file.** Jitterbug analyzes one RTT series from one source to one
destination. A file with replies from several destinations (`fping host1 host2`) is read
as one series unless you separate them, which mixes the paths' latencies; a warning says
so and `metadata["destinations"]` lists them. Analyze each host with
`jitterbug analyze fping.log --group-by destination`, or keep one with
`--series destination=8.8.8.8`. `jitterbug stream` reads one series per process; it warns
when its input mixes destinations, so filter the lines first there (for example
`grep ' 8.8.8.8 '`).

Each timestamped reply becomes one sample: the bracketed time (Unix seconds), `time=` (or
the `ms` value of fping) in milliseconds, and the host as `destination`; the sequence
number is kept as a `seq` column. Duplicate replies (`(DUP!)`, `duplicate for [N]`),
timeouts (`no answer yet`, `Request timeout`, `timed out`) and ICMP errors are skipped and
counted in `metadata["duplicate_replies"]`, `metadata["timeouts"]` and
`metadata["errors"]` (timeouts are only visible when ping prints them, e.g. with
`ping -O`). Headers, statistics and other lines are ignored.

**Timestamps are required.** A file whose reply lines have none is an error that says to
rerun with `-D`; in a file that has some, the reply lines without one are skipped with a
warning and counted in `metadata["untimestamped_replies"]`. Without timestamps the samples
cannot be placed in time, and the time the file was read would be wrong for any recorded
output. BSD and macOS `ping` cannot print Unix timestamps (their `-D` sets the
don't-fragment bit); prefix the lines with moreutils' `ts` instead, which gives the same
shape: `ping -i 5 example.net | ts '[%.s]' > ping.txt`.

The schema is fixed, so the column and unit settings do not apply to these formats; the
row policies do. `jitterbug stream --format ping|fping` reads the same lines live (see
[Online mode](ONLINE_MODE.md)). Windows `ping` output is not supported.

## Compressed files

A file whose name ends in `.gz`, `.bz2`, `.xz` or `.zst` is decompressed while it is
read, and the extension before it decides the format (`rtts.csv.gz` is CSV,
`pings.jsonl.zst` is JSON lines; `data.xz` falls back to the first line). gzip, bzip2
and xz need nothing beyond the standard library. Results work the same way:
`--output results.json.gz` or `results.csv.zst` writes the usual JSON or CSV
compressed.

```bash
zstd rtts.csv                                   # -> rtts.csv.zst
jitterbug analyze rtts.csv.zst --output results.json.zst
zstd -dc results.json.zst | head                # or zstdcat
```

On the bundled PAM 2022 dataset `raw.csv` shrinks from 880 KB to 207 KB with Zstandard.
Its codec is the standard library's `compression.zstd` on Python 3.14 and later; on
older versions install the `zstd` extra (`pip install "jitterbug-inference[zstd]"`),
otherwise a `.zst` path raises an `ImportError` that says so. Parquet output compresses
its own columns, so `results.parquet.gz` and the like are rejected, and so are compressed
Parquet and Arrow inputs.

## pandas DataFrame

```python
from jitterbug import JitterbugAnalyzer, JitterbugConfig
from jitterbug.models import InputConfig

config = JitterbugConfig(input=InputConfig(time_column="ts", time_unit="iso"))
results = JitterbugAnalyzer(config).analyze_from_dataframe(df)
```

The frame goes through the same normalizer as a file, so anything that can be turned
into a frame (a database query, a live probe) can be analyzed.

## InfluxDB

With the `influx` extra, `DataLoader.load_from_influxdb(url, token, org, bucket, query)`
runs a Flux query and maps `_time` to `epoch` (seconds, UTC, rounded to microseconds)
and `_value` (or `rtt`, `latency`, `values`) to the RTT column. The column names and the
time unit are fixed by this mapping; `rtt_unit` and the row policies apply. The token is
used only for the connection and is never written to configuration, logs or results.

## Adding a format

Readers live in `jitterbug.io.readers`. A reader is a function
`read(path, input_config) -> RawData` that returns the raw rows as a DataFrame (plus
optional metadata), registered with `register_reader(Reader(name=..., label=...,
read=..., extensions=(...), sniff=..., magic=(...)))`. `sniff` looks at the first
non-empty line of a text file, `magic` lists the leading bytes of a binary format;
sniffers run in registration order. The normalizer does the rest.
