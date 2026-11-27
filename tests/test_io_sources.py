"""Database sources: DuckDB for real, ClickHouse and InfluxDB through fake clients."""

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from numpy.testing import assert_allclose
from typer.testing import CliRunner

from jitterbug.cli.main import app
from jitterbug.io import DataLoader
from jitterbug.io.inspection import inspect_source
from jitterbug.io.sources import (
    RedactingFilter,
    SourceError,
    available_sources,
    describe_source,
    parse_source_uri,
    read_secret,
    redact,
)
from jitterbug.io.sources import clickhouse as clickhouse_source
from jitterbug.io.sources import influxdb as influxdb_source
from jitterbug.models import JitterbugConfig, SourceConfig
from jitterbug.models.config import InputConfig

runner = CliRunner()

T0 = 1_700_000_000
SECRET = "s3cr3t-Pa55w0rd-xyz"


def flat(output: str) -> str:
    """Console output with Rich's line wrapping undone."""
    return " ".join(output.split())


def _frame(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame({"epoch": T0 + 60.0 * np.arange(n), "rtt": 20.0 + np.arange(n)})


# ---------------------------------------------------------------------------- registry


def test_every_kind_is_registered() -> None:
    assert available_sources() == ["clickhouse", "influxdb2", "influxdb3", "duckdb"]


# --------------------------------------------------------------------------------- URIs


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        (
            "clickhouse://db.example:8123/net",
            {"host": "db.example", "port": 8123, "database": "net"},
        ),
        ("clickhouses://ro@db.example", {"host": "db.example", "secure": True, "user": "ro"}),
        ("clickhouse://localhost", {"host": "localhost", "database": None}),
        ("influxdb2://influx:8086?org=lab", {"host": "influx", "port": 8086, "org": "lab"}),
        ("influxdb2s://cloud.example?org=lab", {"secure": True, "port": None}),
        ("influxdb3://influx:8181/probes", {"database": "probes", "language": "sql"}),
        ("influxdb3://influx/probes?language=influxql", {"language": "influxql"}),
        ("duckdb:", {"path": None}),
        ("duckdb://", {"path": None}),
        ("duckdb::memory:", {"path": None}),
        ("duckdb:///var/lib/rtts.duckdb", {"path": "/var/lib/rtts.duckdb", "read_only": True}),
        (
            "duckdb:data/rtts.duckdb?read_only=false",
            {"path": "data/rtts.duckdb", "read_only": False},
        ),
    ],
)
def test_uri_forms(uri: str, expected: dict[str, Any]) -> None:
    source = parse_source_uri(uri)
    for name, value in expected.items():
        assert getattr(source, name) == value, name


@pytest.mark.parametrize(
    "uri",
    [
        f"clickhouse://default:{SECRET}@db:8123/net",
        f"clickhouse://:{SECRET}@db",
        f"clickhouse://db?password={SECRET}",
        f"influxdb2://influx?org=lab&token={SECRET}",
        f"influxdb3://influx/probes?Token={SECRET}",
    ],
)
def test_uri_credentials_are_rejected_without_echoing_them(uri: str) -> None:
    with pytest.raises(ValueError, match="environment variable") as info:
        parse_source_uri(uri)
    assert SECRET not in str(info.value)


@pytest.mark.parametrize(
    ("uri", "message"),
    [
        ("influxdb://influx:8086", "ambiguous"),
        ("influxdb2://influx/bucket?org=lab", "no path"),
        (f"influxdb3://{SECRET}@influx/probes", "takes no user"),
        ("influxdb3://influx", "needs a database"),
        ("clickhouse://db?compress=1", "Unknown parameter"),
        ("duckdb://host/file.db", "takes no host"),
        ("duckdb:x.db?read_only=maybe", "true or false"),
    ],
)
def test_bad_uris(uri: str, message: str) -> None:
    with pytest.raises(ValueError, match=message) as info:
        parse_source_uri(uri)
    assert SECRET not in str(info.value)


# ------------------------------------------------------------------------ configuration


@pytest.mark.parametrize("key", ["password", "token", "Token", "api_key", "secret"])
def test_profile_secrets_are_rejected(key: str) -> None:
    with pytest.raises(ValueError, match="environment variable") as info:
        JitterbugConfig(sources={"db": {"kind": "clickhouse", "host": "h", key: SECRET}})
    assert SECRET not in str(info.value)


def test_host_with_credentials_is_rejected() -> None:
    with pytest.raises(ValueError) as info:
        SourceConfig(kind="clickhouse", host=f"user:{SECRET}@db")
    assert SECRET not in str(info.value)


def test_settings_of_another_kind_are_rejected() -> None:
    with pytest.raises(ValueError, match="do\\(es\\) not apply to a duckdb source"):
        SourceConfig(kind="duckdb", host="db")
    with pytest.raises(ValueError, match="either query or query_file"):
        SourceConfig(kind="duckdb", query="SELECT 1", query_file=Path("q.sql"))


def test_config_dump_omits_empty_sources_and_keeps_profiles() -> None:
    assert "sources" not in JitterbugConfig().model_dump()
    config = JitterbugConfig(sources={"lab": {"kind": "duckdb", "query": "SELECT 1"}})
    dumped = config.model_dump()
    assert dumped["sources"]["lab"]["kind"] == "duckdb"
    assert JitterbugConfig.model_validate(dumped) == config


def test_secret_env_defaults_and_overrides() -> None:
    assert SourceConfig(kind="clickhouse").secret_env() == "CLICKHOUSE_PASSWORD"
    assert SourceConfig(kind="influxdb2").secret_env() == "INFLUXDB_V2_TOKEN"
    assert SourceConfig(kind="influxdb3", database="d").secret_env() == "INFLUX_TOKEN"
    assert SourceConfig(kind="duckdb").secret_env() is None
    custom = SourceConfig(kind="clickhouse", password_env="LAB_CH_PASSWORD")
    assert custom.secret_env() == "LAB_CH_PASSWORD"
    with pytest.raises(ValueError):
        SourceConfig(kind="clickhouse", password_env="not a variable name")


# ---------------------------------------------------------------------------- redaction


def test_redact_scrubs_known_secrets_and_credential_patterns() -> None:
    text = (
        f"connect to http://default:{SECRET}@db:8123 failed; password={SECRET}; "
        f"headers {{'Authorization': 'Token abcdefgh12345678'}}; token: other-secret-value"
    )
    cleaned = redact(text, [SECRET, None])
    assert SECRET not in cleaned
    assert "abcdefgh12345678" not in cleaned
    assert "other-secret-value" not in cleaned
    assert "http://default:***@db:8123" in cleaned


# ------------------------------------------------------------------------------- DuckDB


@pytest.fixture
def duckdb() -> Any:
    return pytest.importorskip("duckdb")


def _duckdb(query: str, **settings: Any) -> SourceConfig:
    return SourceConfig(kind="duckdb", query=query, **settings)


VALUES_QUERY = (
    "SELECT * FROM (VALUES (1700000120, 22.0), (1700000000, 20.0), (1700000060, 21.0)) "
    "AS t(epoch, rtt)"
)


@pytest.mark.usefixtures("duckdb")
class TestDuckDB:
    def test_in_memory_query(self) -> None:
        dataset = DataLoader().load_from_source(_duckdb(VALUES_QUERY))
        epochs, rtts = dataset.to_arrays()
        assert_allclose(epochs, [T0, T0 + 60, T0 + 120])  # sorted on load, as for files
        assert_allclose(rtts, [20.0, 21.0, 22.0])
        assert dataset.metadata["source"] == "duckdb"
        assert dataset.metadata["source_details"] == {"kind": "duckdb", "path": ":memory:"}
        assert dataset.metadata["query"] == VALUES_QUERY
        assert dataset.metadata["sorted_on_load"] is True

    def test_timestamps_load_without_time_unit(self) -> None:
        query = (
            "SELECT TIMESTAMP '2024-01-01 00:00:00' + INTERVAL (i) MINUTE AS ts, "
            "TIMESTAMPTZ '2024-01-01 00:00:00+00' + INTERVAL (i) MINUTE AS tz, "
            "20.0 + i AS latency_us FROM range(3) AS r(i)"
        )
        for column in ("ts", "tz"):
            config = InputConfig(time_column=column, rtt_column="latency_us", rtt_unit="us")
            dataset = DataLoader(config).load_from_source(_duckdb(query))
            epochs, rtts = dataset.to_arrays()
            start = datetime(2024, 1, 1, tzinfo=UTC).timestamp()
            assert_allclose(epochs, [start, start + 60, start + 120])
            assert_allclose(rtts, [0.020, 0.021, 0.022])
            assert dataset.metadata["time_unit"] == "iso"

    def test_csv_and_parquet_files(self, tmp_path: Path, duckdb: Any) -> None:
        for day in (1, 2):
            frame = _frame(3).assign(epoch=lambda f, d=day: f["epoch"] + 86400 * d)
            frame.to_csv(tmp_path / f"day{day}.csv", index=False)
        connection = duckdb.connect()
        connection.execute(
            f"COPY (SELECT * FROM read_csv('{tmp_path}/day*.csv')) "
            f"TO '{tmp_path}/all.parquet' (FORMAT parquet)"
        )
        connection.close()
        csv = DataLoader().load_from_source(
            _duckdb(f"SELECT * FROM read_csv('{tmp_path}/day*.csv') ORDER BY epoch")
        )
        parquet = DataLoader().load_from_source(
            _duckdb(f"SELECT epoch, rtt FROM '{tmp_path}/*.parquet'")
        )
        assert len(csv) == len(parquet) == 6
        assert_allclose(csv.to_arrays()[0], parquet.to_arrays()[0])

    def test_database_file_is_read_only_by_default(self, tmp_path: Path, duckdb: Any) -> None:
        path = tmp_path / "rtts.duckdb"
        connection = duckdb.connect(str(path))
        connection.register("frame_view", _frame(5))
        connection.execute("CREATE TABLE rtts AS SELECT * FROM frame_view")
        connection.close()

        dataset = DataLoader().load_from_source(_duckdb("SELECT * FROM rtts", path=str(path)))
        assert len(dataset) == 5
        assert dataset.metadata["source_details"] == {
            "kind": "duckdb",
            "path": str(path),
            "read_only": True,
        }
        with pytest.raises(SourceError, match="DuckDB query failed: .*read-only"):
            DataLoader().load_from_source(
                _duckdb("CREATE TABLE x AS SELECT 1 AS epoch; SELECT 1", path=str(path))
            )
        missing = tmp_path / "missing.duckdb"
        with pytest.raises(SourceError):
            DataLoader().load_from_source(_duckdb("SELECT 1", path=str(missing)))
        assert not missing.exists()  # read-only never creates a database file

    def test_several_statements_and_query_file(self, tmp_path: Path) -> None:
        query_file = tmp_path / "rtts.sql"
        query_file.write_text(
            "CREATE TABLE t AS SELECT * FROM range(4) AS r(i);\n"
            f"SELECT {T0} + 60 * i AS epoch, 30.0 AS rtt FROM t;\n"
        )
        source = SourceConfig(kind="duckdb", query_file=query_file)
        dataset = DataLoader().load_from_source(source)
        assert len(dataset) == 4
        assert dataset.metadata["source_details"]["query_file"] == str(query_file)
        assert dataset.metadata["query"].startswith("CREATE TABLE t")

    def test_group_by_and_series_apply_to_query_results(self) -> None:
        query = (
            f"SELECT {T0} + 60 * (i // 2) AS epoch, 20.0 + i AS rtt, "
            "CASE WHEN i % 2 = 0 THEN 'a' ELSE 'b' END AS dst FROM range(8) AS r(i)"
        )
        grouped = DataLoader(InputConfig(group_by=["dst"])).load_series_from_source(_duckdb(query))
        assert list(grouped) == [(("dst", "a"),), (("dst", "b"),)]
        assert all(len(d) == 4 for d in grouped.values())
        assert grouped[(("dst", "b"),)].metadata["series_key"] == {"dst": "b"}
        one = DataLoader(InputConfig(series={"dst": "b"})).load_from_source(_duckdb(query))
        assert_allclose(one.to_arrays()[1], [21.0, 23.0, 25.0, 27.0])
        with pytest.raises(ValueError, match="holds 2 series"):
            DataLoader(InputConfig(group_by=["dst"])).load_from_source(_duckdb(query))

    def test_mapping_errors_name_the_query_result(self) -> None:
        with pytest.raises(ValueError, match="duckdb query result: Input has no 'epoch'"):
            DataLoader().load_from_source(_duckdb("SELECT 1 AS t, 2.0 AS rtt"))

    def test_inspect_reports_kind_query_and_columns(self) -> None:
        report = inspect_source(_duckdb(VALUES_QUERY), InputConfig())
        assert report["source"] == {"kind": "duckdb", "path": ":memory:"}
        assert report["query"] == VALUES_QUERY
        assert report["columns"] == ["epoch", "rtt"]
        assert report["rows"]["valid"] == 3
        assert report["errors"] == []
        json.dumps(report)


def test_missing_query_is_an_error() -> None:
    with pytest.raises(ValueError, match="needs a query"):
        DataLoader().load_from_source(SourceConfig(kind="duckdb"))


@pytest.mark.parametrize(
    ("module", "kind", "extra"),
    [
        ("clickhouse_connect", "clickhouse", "clickhouse"),
        ("influxdb_client", "influxdb2", "influx"),
        ("influxdb_client_3", "influxdb3", "influxdb3"),
        ("duckdb", "duckdb", "duckdb"),
    ],
)
def test_missing_client_library_names_the_extra(
    monkeypatch: pytest.MonkeyPatch, module: str, kind: str, extra: str
) -> None:
    monkeypatch.setitem(sys.modules, module, None)  # makes the import fail
    monkeypatch.setenv("INFLUX_TOKEN", "token-for-the-import-test")  # checked before the import
    settings = {"database": "d"} if kind == "influxdb3" else {}
    source = SourceConfig(kind=kind, query="SELECT 1", **settings)  # type: ignore[arg-type]
    with pytest.raises(ImportError, match=rf"jitterbug-inference\[{extra}\]"):
        DataLoader().load_from_source(source)


# ------------------------------------------------------------------- fake server clients


class FakeClient:
    """Records how it was built and queried; returns ``result`` or raises ``error``."""

    def __init__(self, result: Any = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.kwargs: dict[str, Any] = {}
        self.queries: list[Any] = []
        self.closed = False

    def build(self, **kwargs: Any) -> "FakeClient":
        self.kwargs = kwargs
        return self

    def _answer(self, *args: Any, **kwargs: Any) -> Any:
        self.queries.append(args or kwargs)
        if self.error is not None:
            raise self.error
        return self.result

    # clickhouse-connect
    def query_df(self, query: str) -> Any:
        return self._answer(query)

    # influxdb-client (v2)
    def query_api(self) -> "FakeClient":
        return self

    def query_data_frame(self, query: str) -> Any:
        return self._answer(query)

    # influxdb3-python
    def query(self, **kwargs: Any) -> Any:
        return self._answer(**kwargs)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def clickhouse(monkeypatch: pytest.MonkeyPatch) -> FakeClient:
    """A fake ClickHouse at the client-construction seam; DateTime and DateTime64
    columns come back as datetimes, as ``query_df`` returns them."""
    fake = FakeClient(
        pd.DataFrame(
            {
                "ts": pd.to_datetime([T0, T0 + 60, T0 + 120], unit="s").astype("datetime64[s]"),
                "ts64": pd.to_datetime([T0, T0 + 60, T0 + 120], unit="s", utc=True).astype(
                    "datetime64[ms, UTC]"
                ),
                "rtt_ms": [20.0, 21.0, 22.0],
                "dst": ["8.8.8.8"] * 3,
            }
        )
    )
    monkeypatch.setattr(clickhouse_source, "create_client", fake.build)
    monkeypatch.delenv("CLICKHOUSE_PASSWORD", raising=False)
    monkeypatch.delenv("CLICKHOUSE_USER", raising=False)
    return fake


QUERY = "SELECT ts, ts64, rtt_ms, dst FROM net.rtts WHERE dst = '8.8.8.8'"


class TestClickHouse:
    def test_query_with_datetime_columns(
        self, clickhouse: FakeClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CLICKHOUSE_PASSWORD", SECRET)
        source = parse_source_uri("clickhouse://ro@db.example:8123/net").model_copy(
            update={"query": QUERY}
        )
        for column in ("ts", "ts64"):
            config = InputConfig(time_column=column, rtt_column="rtt_ms")
            dataset = DataLoader(config).load_from_source(source)
            assert_allclose(dataset.to_arrays()[0], [T0, T0 + 60, T0 + 120])
            assert dataset.metadata["source_details"] == {
                "kind": "clickhouse",
                "host": "db.example",
                "port": 8123,
                "secure": False,
                "database": "net",
            }
            assert SECRET not in json.dumps(dataset.metadata, default=str)
        assert clickhouse.kwargs == {
            "host": "db.example",
            "port": 8123,
            "secure": False,
            "database": "net",
            "username": "ro",
            "password": SECRET,
        }
        assert clickhouse.queries[-1] == (QUERY,)
        assert clickhouse.closed

    def test_user_from_environment_and_no_password(
        self, clickhouse: FakeClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CLICKHOUSE_USER", "reader")
        source = SourceConfig(kind="clickhouse", query=QUERY)
        DataLoader(InputConfig(time_column="ts", rtt_column="rtt_ms")).load_from_source(source)
        assert clickhouse.kwargs == {"host": "localhost", "secure": False, "username": "reader"}

    def test_password_env_names_another_variable(
        self, clickhouse: FakeClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LAB_CH_PASSWORD", SECRET)
        source = SourceConfig(kind="clickhouse", password_env="LAB_CH_PASSWORD", query=QUERY)
        DataLoader(InputConfig(time_column="ts", rtt_column="rtt_ms")).load_from_source(source)
        assert clickhouse.kwargs["password"] == SECRET

    def test_errors_are_redacted_and_not_chained(
        self,
        clickhouse: FakeClient,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setenv("CLICKHOUSE_PASSWORD", SECRET)
        clickhouse.error = ConnectionError(
            f"HTTPDriver for http://ro:{SECRET}@db:8123 received ClickHouse error code 516: "
            f"Authentication failed: password={SECRET} is incorrect"
        )
        source = SourceConfig(kind="clickhouse", host="db", user="ro", query=QUERY)
        with caplog.at_level(logging.DEBUG), pytest.raises(SourceError) as info:
            DataLoader().load_from_source(source)
        message = str(info.value)
        assert message.startswith("ClickHouse query failed: ConnectionError:")
        assert "Authentication failed" in message
        assert SECRET not in message
        assert info.value.__context__ is None and info.value.__cause__ is None
        assert SECRET not in caplog.text
        assert clickhouse.closed

    def test_cli_error_output_is_redacted(
        self, clickhouse: FakeClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CLICKHOUSE_PASSWORD", SECRET)
        clickhouse.error = RuntimeError(f"auth failed for password {SECRET}")
        result = runner.invoke(app, ["validate", "clickhouse://db/net", "--query", QUERY])
        assert result.exit_code == 1
        assert "ClickHouse query failed" in flat(result.output)
        assert SECRET not in result.output

    def test_cli_analyze_records_no_secret(
        self, clickhouse: FakeClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("CLICKHOUSE_PASSWORD", SECRET)
        out = tmp_path / "r.json"
        result = runner.invoke(
            app,
            [
                "analyze",
                "clickhouse://db/net",
                "--query",
                QUERY,
                "--time-column",
                "ts",
                "--rtt-column",
                "rtt_ms",
                "-o",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        assert SECRET not in result.output
        assert SECRET not in out.read_text()


@pytest.fixture
def influx2(monkeypatch: pytest.MonkeyPatch) -> FakeClient:
    """A fake InfluxDB 2: two Flux tables with ``_time``, ``_value`` and a tag."""
    tables = [
        pd.DataFrame(
            {
                "result": "_result",
                "table": t,
                "_time": pd.to_datetime([T0 + 60 * i for i in rows], unit="s", utc=True),
                "_value": [20.0 + i for i in rows],
                "_field": "rtt",
                "dst": dst,
            }
        )
        for t, (rows, dst) in enumerate([((0, 1, 2), "a"), ((0, 1), "b")])
    ]
    fake = FakeClient(tables)
    monkeypatch.setattr(influxdb_source, "create_v2_client", fake.build)
    monkeypatch.delenv("INFLUXDB_V2_TOKEN", raising=False)
    return fake


FLUX = 'from(bucket: "probes") |> range(start: -1d) |> filter(fn: (r) => r._field == "rtt")'


class TestInfluxDB2:
    def test_flux_defaults_and_token_from_environment(
        self, influx2: FakeClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("INFLUXDB_V2_TOKEN", SECRET)
        source = parse_source_uri("influxdb2://influx?org=lab").model_copy(update={"query": FLUX})
        datasets = DataLoader(InputConfig(group_by=["dst"])).load_series_from_source(source)
        assert influx2.kwargs == {"url": "http://influx:8086", "token": SECRET, "org": "lab"}
        assert influx2.queries == [(FLUX,)]
        assert influx2.closed
        assert {k: len(d) for k, d in datasets.items()} == {(("dst", "a"),): 3, (("dst", "b"),): 2}
        assert_allclose(datasets[(("dst", "a"),)].to_arrays()[1], [20.0, 21.0, 22.0])
        assert datasets[(("dst", "a"),)].metadata["rtt_column"] == "_value"
        assert datasets[(("dst", "a"),)].metadata["time_column"] == "_time"

    def test_https_url_and_explicit_mapping(self, influx2: FakeClient) -> None:
        source = SourceConfig(kind="influxdb2", host="cloud.example", secure=True, query=FLUX)
        config = InputConfig(time_column="_time", rtt_column="table", series={"dst": "b"})
        dataset = DataLoader(config).load_from_source(source)
        assert influx2.kwargs["url"] == "https://cloud.example"
        assert influx2.kwargs["token"] is None
        assert dataset.metadata["rtt_column"] == "table"

    def test_load_from_influxdb_keeps_working(self, influx2: FakeClient) -> None:
        dataset = DataLoader().load_from_influxdb(
            url="http://influx:8086", token=SECRET, org="lab", bucket="probes", query=FLUX
        )
        assert len(dataset) == 5
        assert influx2.kwargs == {"url": "http://influx:8086", "token": SECRET, "org": "lab"}
        assert SECRET not in json.dumps(dataset.metadata, default=str)

    def test_load_from_influxdb_redacts_errors(self, influx2: FakeClient) -> None:
        influx2.error = RuntimeError(f"401 Unauthorized: header Authorization: Token {SECRET}")
        with pytest.raises(SourceError) as info:
            DataLoader().load_from_influxdb(
                url="http://influx:8086", token=SECRET, org="lab", bucket="b", query=FLUX
            )
        assert SECRET not in str(info.value)


@pytest.fixture
def influx3(monkeypatch: pytest.MonkeyPatch) -> FakeClient:
    """A fake InfluxDB 3: SQL results name the timestamp ``time`` (naive UTC)."""
    fake = FakeClient(
        pd.DataFrame(
            {
                "time": pd.to_datetime([T0, T0 + 60, T0 + 120], unit="s"),
                "rtt": [20.0, 21.0, 22.0],
                "dst": ["a", "a", "a"],
            }
        )
    )
    monkeypatch.setattr(influxdb_source, "create_v3_client", fake.build)
    monkeypatch.delenv("INFLUX_TOKEN", raising=False)
    return fake


class TestInfluxDB3:
    def test_sql_query_with_time_default(
        self, influx3: FakeClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("INFLUX_TOKEN", SECRET)
        query = "SELECT time, rtt, dst FROM ping WHERE time > now() - INTERVAL '1 day'"
        source = parse_source_uri("influxdb3://influx/probes").model_copy(update={"query": query})
        dataset = DataLoader().load_from_source(source)
        assert influx3.kwargs == {
            "host": "http://influx:8181",
            "token": SECRET,
            "database": "probes",
        }
        assert influx3.queries == [{"query": query, "language": "sql", "mode": "pandas"}]
        assert influx3.closed
        assert_allclose(dataset.to_arrays()[0], [T0, T0 + 60, T0 + 120])
        assert dataset.metadata["source_details"]["database"] == "probes"

    def test_influxql_and_token_env(
        self, influx3: FakeClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LAB_TOKEN", SECRET)
        source = SourceConfig(
            kind="influxdb3",
            host="::1",
            port=9999,
            database="probes",
            language="influxql",
            token_env="LAB_TOKEN",
            query="SELECT rtt FROM ping",
        )
        DataLoader().load_from_source(source)
        assert influx3.kwargs["host"] == "http://[::1]:9999"
        assert influx3.kwargs["token"] == SECRET
        assert influx3.queries[0]["language"] == "influxql"


def test_describe_source_never_includes_credentials_settings() -> None:
    source = SourceConfig(kind="clickhouse", host="db", user="ro", password_env="X")
    assert describe_source(source) == {"kind": "clickhouse", "host": "db", "secure": False}


# ---------------------------------------------------------------------------------- CLI


@pytest.mark.usefixtures("duckdb")
class TestCLI:
    def test_inspect_duckdb_uri_as_json(self) -> None:
        result = runner.invoke(app, ["inspect", "duckdb:", "--query", VALUES_QUERY, "--json"])
        assert result.exit_code == 0, result.output
        report = json.loads(result.output)
        assert report["source"]["kind"] == "duckdb"
        assert report["query"] == VALUES_QUERY
        assert report["columns"] == ["epoch", "rtt"]

    def test_inspect_human_readable(self) -> None:
        result = runner.invoke(app, ["inspect", "duckdb:", "--query", VALUES_QUERY])
        assert result.exit_code == 0, result.output
        text = flat(result.output)
        assert "duckdb source (path: :memory:)" in text
        assert "Columns: epoch, rtt" in text
        assert "VALUES (1700000120, 22.0)" in text

    def test_source_profile_from_config(self, tmp_path: Path) -> None:
        query_file = tmp_path / "q.sql"
        query_file.write_text(VALUES_QUERY)
        config = tmp_path / "jitterbug.yaml"
        config.write_text(
            yaml.safe_dump({"sources": {"lab": {"kind": "duckdb", "query_file": str(query_file)}}})
        )
        result = runner.invoke(app, ["validate", "--source", "lab", "--config", str(config)])
        assert result.exit_code == 0, result.output
        assert "Total Measurements" in result.output or "3" in result.output
        # --query replaces the profile's query
        result = runner.invoke(
            app,
            ["inspect", "--source", "lab", "-c", str(config), "--json", "--query", "SELECT 1 AS x"],
        )
        assert json.loads(result.output)["columns"] == ["x"]

    def test_replay_and_analyze_accept_a_source(self, tmp_path: Path) -> None:
        query = f"SELECT {T0} + 15 * i AS epoch, 20.0 + (i % 7) AS rtt FROM range(2000) r(i)"
        result = runner.invoke(app, ["analyze", "duckdb:", "--query", query, "--summary-only"])
        assert result.exit_code == 0, result.output
        result = runner.invoke(app, ["replay", "duckdb:", "--query", query])
        assert result.exit_code == 0, result.output


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["validate", "duckdb:"], "needs a query"),
        (["validate", "--source", "nope"], "no source named 'nope'"),
        (["validate", "duckdb:", "--source", "x", "--query", "SELECT 1"], "not both"),
        (["validate", "duckdb:", "--query", "SELECT 1", "--format", "csv"], "--format applies"),
        (["validate", "influxdb://h", "--query", "x"], "ambiguous"),
        (["validate"], "missing INPUT"),
        (["validate", "duckdb:", "--query", "q", "--query-file", "q.sql"], "not both"),
        (["validate", "duckdb:", "--query-file", "/nonexistent/q.sql"], "does not exist"),
        (["validate", "missing.csv"], "does not exist"),
    ],
)
def test_cli_usage_errors(args: list[str], message: str) -> None:
    result = runner.invoke(app, args)
    assert result.exit_code == 2
    assert message in flat(result.output)


def test_cli_query_with_a_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "rtts.csv"
    _frame().to_csv(path, index=False)
    result = runner.invoke(app, ["validate", str(path), "--query", "SELECT 1"])
    assert result.exit_code == 2
    assert "apply to a database source" in flat(result.output)


def test_cli_uri_with_password_is_rejected_without_echo() -> None:
    result = runner.invoke(
        app, ["analyze", f"clickhouse://default:{SECRET}@db/net", "--query", "SELECT 1"]
    )
    assert result.exit_code == 2
    assert "CLICKHOUSE_PASSWORD" in flat(result.output)
    assert SECRET not in result.output


def test_cli_config_with_secret_is_rejected_without_echo(tmp_path: Path) -> None:
    config = tmp_path / "c.yaml"
    config.write_text(
        yaml.safe_dump({"sources": {"db": {"kind": "clickhouse", "password": SECRET}}})
    )
    result = runner.invoke(app, ["validate", "--source", "db", "-c", str(config)])
    assert result.exit_code == 1
    assert "CLICKHOUSE_PASSWORD" in flat(result.output)
    assert SECRET not in result.output


def test_redacting_filter_scrubs_client_library_logs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A client library logging the secret (message or traceback) is redacted by the
    filter the CLI installs on its handlers."""
    monkeypatch.setenv("CLICKHOUSE_PASSWORD", SECRET)
    read_secret(SourceConfig(kind="clickhouse"))
    caplog.handler.addFilter(RedactingFilter())
    library = logging.getLogger("clickhouse_connect.driver")
    try:
        raise ConnectionError(f"basic auth ro:{SECRET} refused")
    except ConnectionError:
        library.warning("retrying with %s", SECRET, exc_info=True)
    assert "retrying with ***" in caplog.text
    assert "ConnectionError" in caplog.text
    assert SECRET not in caplog.text


def test_influxdb3_without_token_says_which_variable(influx3: FakeClient) -> None:
    source = SourceConfig(kind="influxdb3", database="probes", query="SELECT 1")
    with pytest.raises(SourceError, match="INFLUX_TOKEN"):
        DataLoader().load_from_source(source)
