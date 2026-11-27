# Data sources

Jitterbug can read RTT samples straight from a database query instead of a file. The
query result is a table like any CSV: it goes through the same column mapping, units,
row policies, `--series` filters and `--group-by` split (see
[Input formats](INPUT_FORMATS.md)), so nothing assumes a schema. You write the query,
and map its columns with `--time-column`, `--rtt-column` and friends, or the `input`
section of the configuration file.

| Kind | Database | Query language | Extra | Secret from |
| --- | --- | --- | --- | --- |
| `clickhouse` | ClickHouse (HTTP interface) | SQL | `clickhouse` ([clickhouse-connect](https://pypi.org/project/clickhouse-connect/)) | `CLICKHOUSE_PASSWORD` |
| `influxdb2` | InfluxDB 2 | Flux | `influx` ([influxdb-client](https://pypi.org/project/influxdb-client/)) | `INFLUXDB_V2_TOKEN` |
| `influxdb3` | InfluxDB 3 (Core, Enterprise, Cloud) | SQL or InfluxQL | `influxdb3` ([influxdb3-python](https://pypi.org/project/influxdb3-python/)) | `INFLUX_TOKEN` |
| `duckdb` | DuckDB, in memory or a database file | SQL | `duckdb` ([duckdb](https://pypi.org/project/duckdb/)) | none |

```bash
pip install 'jitterbug-inference[clickhouse]'     # or [influx], [influxdb3], [duckdb]
```

Every command that takes an input file (`analyze`, `validate`, `inspect`, `visualize`,
`replay`) also takes a source, in one of two ways:

- a **URI** in place of the file, with `--query` or `--query-file`:

  ```bash
  export CLICKHOUSE_PASSWORD=...           # never on the command line
  jitterbug analyze clickhouse://db.example:8123/netmon \
      --query "SELECT ts, rtt_ms, dst FROM pings WHERE ts > now() - INTERVAL 7 DAY" \
      --time-column ts --rtt-column rtt_ms --group-by dst -o results.json
  ```

- a **profile** from the configuration file, with `--source NAME`:

  ```bash
  jitterbug analyze --source netmon --config jitterbug.yaml -o results.json
  ```

`jitterbug stream` does not read databases, and nothing polls a database for new rows:
each command runs its query once.

## URIs

| URI | Meaning |
| --- | --- |
| `clickhouse://[user@]host[:port][/database]` | ClickHouse over HTTP; the client's default port (8123) when none is given |
| `clickhouses://[user@]host[:port][/database]` | The same over HTTPS (8443 by default) |
| `influxdb2://host[:port]?org=ORG` | InfluxDB 2, Flux; port 8086 by default. The bucket is named in the Flux query |
| `influxdb2s://host[:port]?org=ORG` | The same over HTTPS (443 by default) |
| `influxdb3://host[:port]/database[?language=influxql]` | InfluxDB 3, SQL by default; port 8181 by default |
| `influxdb3s://host[:port]/database` | The same over HTTPS (443 by default) |
| `duckdb:` (or `duckdb://`, `duckdb::memory:`) | DuckDB, in-memory database |
| `duckdb:///absolute/path/rtts.duckdb` | DuckDB database file, opened read-only |
| `duckdb:relative/path/rtts.duckdb` | The same, relative to the working directory |
| `duckdb:rtts.duckdb?read_only=false` | Open the file for writing (the query may create tables) |

`influxdb://` is refused as ambiguous: say `influxdb2://` or `influxdb3://`. A URI never
holds a password or token: `clickhouse://user:password@host`, `?password=` or `?token=`
are rejected with an error that names the environment variable to use instead, and the
error does not repeat the value.

## Profiles

Named sources live under `sources:` in the configuration file, next to `input`:

```yaml
sources:
  netmon:
    kind: clickhouse
    host: db.example
    port: 8123
    database: netmon
    user: jitterbug_ro             # or CLICKHOUSE_USER; the password is never here
    query_file: queries/pings.sql
  lab:
    kind: influxdb3
    host: influx.lab
    database: probes
    token_env: LAB_INFLUX_TOKEN    # read the token from this variable instead
    query: >
      SELECT time, rtt, dst FROM ping WHERE time > now() - INTERVAL '1 day'
  archive:
    kind: duckdb
    path: /data/rtts.duckdb
    query: SELECT epoch, rtt FROM pings WHERE probe = 'p1'

input:
  time_column: ts
  rtt_column: rtt_ms
```

| Setting | Kinds | Meaning |
| --- | --- | --- |
| `kind` | all | `clickhouse`, `influxdb2`, `influxdb3` or `duckdb` |
| `query` / `query_file` | all | The query, or a file holding it (read when the query runs; a relative path is relative to the working directory, not the configuration file). At most one; `--query`/`--query-file` replace it |
| `host`, `port`, `secure` | ClickHouse, InfluxDB | Server address (no scheme, no user) and whether to use HTTPS. `host` defaults to `localhost` |
| `database` | ClickHouse, InfluxDB 3 | Database (required for InfluxDB 3) |
| `user` | ClickHouse | User; `CLICKHOUSE_USER`, then the server's `default` user, when unset |
| `org` | InfluxDB 2 | Organization |
| `language` | InfluxDB 3 | `sql` (default) or `influxql` |
| `path`, `read_only` | DuckDB | Database file (in memory when unset) and whether to open it read-only (default `true`) |
| `password_env` | ClickHouse | Environment variable holding the password (default `CLICKHOUSE_PASSWORD`) |
| `token_env` | InfluxDB | Environment variable holding the token (default `INFLUXDB_V2_TOKEN` or `INFLUX_TOKEN`) |

A setting that does not apply to the kind (`host` on a DuckDB source, `org` on
ClickHouse) is an error rather than silently ignored, and so is any setting named like a
secret (`password`, `token`, `api_key`, `secret`, ...).

## Credentials

Passwords and tokens come only from environment variables, read when the query runs:

| Kind | Variable | Notes |
| --- | --- | --- |
| `clickhouse` | `CLICKHOUSE_PASSWORD` (user: `CLICKHOUSE_USER`) | The names `clickhouse-client` uses. Unset means no password |
| `influxdb2` | `INFLUXDB_V2_TOKEN` | The name `influxdb-client` uses |
| `influxdb3` | `INFLUX_TOKEN` | The name `influxdb3-python` uses. Required by the client; any value works for a server started without authentication |
| `duckdb` | none | Credentials of attached databases and object storage go through DuckDB itself; see the recipes below |

`password_env` and `token_env` name another variable, for instance to keep two servers'
tokens apart. Jitterbug never writes a secret anywhere: configuration dumps and result
files hold the variable's name at most; errors from the client libraries are redacted
(the secret, URL user information, `password=`/`token=` pairs and `Authorization`
headers become `***`) and not chained to the original exception; and the commands
install a logging filter that redacts what the client libraries log themselves.

What *is* recorded, in the dataset metadata (`RTTDataset.metadata`) and the
`jitterbug inspect` report: the source kind, its connection details (`host`, `port`,
`secure`, `database`, `org`, `language`, `path`, `query_file`) and the query text. Keep
secrets out of the query text itself (see the DuckDB recipes for how).

## Columns and time

The query decides the columns, so map them as you would for a CSV. Two conveniences:

- A **datetime** time column (ClickHouse `DateTime`/`DateTime64`, DuckDB `TIMESTAMP` and
  `TIMESTAMPTZ`, InfluxDB times) is read as datetimes whatever `time_unit` says; naive
  datetimes are taken as UTC. A numeric column still follows `time_unit`
  (`s` by default, or `auto`).
- **InfluxDB defaults.** While `time_column` is left at `epoch` and the result has no
  `epoch` column, the time is taken from `_time` (Flux) or `time` (InfluxDB 3). For
  Flux, `_value` is the RTT column while `rtt_column` is unset. A plain Flux query
  therefore needs no mapping; pivoted results do. With several Flux tables (one per
  series), they are concatenated, and `--group-by` on a tag column splits them again.

Check what a query returns before analyzing it; `inspect` runs the query and reports
the source, the query, the columns, the mapping in effect, the row policies and the
series found:

```bash
jitterbug inspect influxdb3://influx.lab/probes \
    --query "SELECT time, rtt, dst FROM ping LIMIT 1000" --group-by dst
```

## DuckDB recipes

DuckDB is the universal adapter: with an in-memory database (`duckdb:`), the query can
read files and other databases directly. Statements separated by `;` run in order and
the last one returns the rows. Extensions (`sqlite`, `postgres`, `httpfs`) are
downloaded by DuckDB the first time they are loaded, so that step needs network access.

**Many Parquet or CSV files at once** (globs, hive partitions, a union of formats):

```bash
jitterbug analyze duckdb: --query "
  SELECT epoch, rtt_ms AS rtt, probe
  FROM read_parquet('archive/year=*/month=*/*.parquet', hive_partitioning = true)
  WHERE year = 2025
" --group-by probe -o results.json

jitterbug inspect duckdb: --query "SELECT * FROM read_csv('logs/*.csv.gz')"
```

**A SQLite or PostgreSQL database**, attached read-only:

```sql
-- sqlite.sql
INSTALL sqlite; LOAD sqlite;
ATTACH 'measurements.sqlite' AS m (TYPE sqlite, READ_ONLY);
SELECT ts, rtt FROM m.pings WHERE target = '192.0.2.1';
```

```sql
-- postgres.sql: the password comes from libpq's PGPASSWORD (or ~/.pgpass), not the query
INSTALL postgres; LOAD postgres;
ATTACH 'host=db.example dbname=netmon user=reader' AS pg (TYPE postgres, READ_ONLY);
SELECT ts, rtt_ms, dst FROM pg.public.pings WHERE ts > now() - INTERVAL 7 DAY;
```

```bash
jitterbug analyze duckdb: --query-file postgres.sql --time-column ts --rtt-column rtt_ms
```

**Object storage** (S3, GCS, R2, HTTP) through `httpfs`, with credentials found by
DuckDB's credential chain (environment, `~/.aws`, instance roles), never written in the
query:

```sql
INSTALL httpfs; LOAD httpfs; INSTALL aws; LOAD aws;
CREATE SECRET (TYPE s3, PROVIDER credential_chain);
SELECT epoch, rtt FROM read_parquet('s3://my-bucket/rtts/2025-*.parquet');
```

A DuckDB query runs with your permissions: it can read any file you can. A database
file is opened read-only unless `read_only` is false, and never created by a read-only
source.

## Python

```python
from jitterbug import JitterbugAnalyzer
from jitterbug.io import DataLoader
from jitterbug.io.sources import parse_source_uri
from jitterbug.models import InputConfig, JitterbugConfig, SourceConfig

source = SourceConfig(
    kind="clickhouse", host="db.example", database="netmon",
    query="SELECT ts, rtt_ms FROM pings WHERE dst = '8.8.8.8'",
)  # password from CLICKHOUSE_PASSWORD
config = JitterbugConfig(input=InputConfig(time_column="ts", rtt_column="rtt_ms"))
results = JitterbugAnalyzer(config).analyze_from_source(source)

source = parse_source_uri("duckdb:").model_copy(
    update={"query": "SELECT * FROM 'rtts/*.parquet'"}
)
datasets = DataLoader(InputConfig(group_by=["dst"])).load_series_from_source(source)
```

`DataLoader.load_from_source`, `load_series_from_source` and `read_raw_source`, and
`JitterbugAnalyzer.analyze_from_source` and `analyze_series_from_source` mirror their
file counterparts. `DataLoader.load_from_influxdb(url, token, org, bucket, query)`
still works for existing code; it now uses the same path (and redaction). To keep
client libraries' own log records clean in your application, add
`jitterbug.io.sources.RedactingFilter` to your handlers (or call
`install_redacting_filter()`).

## Adding a back end

Back ends live in `jitterbug.io.sources`. A back end is a function
`fetch(source, query, secret) -> DataFrame` that imports its client lazily (through
`import_client`, so a missing library names the extra), runs the query and closes the
connection, registered with `register_source(QuerySource(kind=..., label=...,
fetch=...))`. Add the kind to `SourceConfig` and, if it takes a URI, to
`jitterbug.io.sources.uri`. Errors, redaction, datetime handling and normalization are
handled by the caller.
