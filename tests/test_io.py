"""Unit tests for ``jitterbug.io``: loaders, format inference, validation and exporters."""

import json
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from jitterbug.io import DataLoader, ResultExporter
from jitterbug.models import CongestionInferenceResult, RTTDataset

from .conftest import jitter, jump, make_measurements

# --------------------------------------------------------------------------- loaders


@pytest.fixture
def loader() -> DataLoader:
    return DataLoader()


@pytest.fixture
def csv_file(tmp_path: Path) -> Path:
    path = tmp_path / "rtts.csv"
    path.write_text("epoch,values\n1700000000.0,20.5\n1700000060.0,21.0\n1700000120.0,19.5\n")
    return path


@pytest.fixture
def scamper_file(tmp_path: Path) -> Path:
    """Two ping records; the second one is out of order and has one response without rtt."""
    records = [
        {
            "type": "ping",
            "src": "10.0.0.1",
            "dst": "8.8.8.8",
            "responses": [{"rtt": 12.5, "tx": {"sec": 1700000100, "usec": 500000}}],
        },
        {"type": "cycle-start", "id": 1},  # not a ping: ignored
        {
            "type": "ping",
            "src": "10.0.0.1",
            "dst": "8.8.8.8",
            "responses": [
                {"rtt": 11.0, "tx": {"sec": 1700000000, "usec": 0}},
                {"tx": {"sec": 1700000001, "usec": 0}},  # timeout: no rtt
            ],
        },
    ]
    path = tmp_path / "pings.json"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\nnot json at all\n")
    return path


class TestCSV:
    def test_load(self, loader: DataLoader, csv_file: Path) -> None:
        ds = loader.load_from_file(csv_file)
        assert len(ds) == 3
        assert_allclose([m.rtt_value for m in ds.measurements], [20.5, 21.0, 19.5])
        assert ds.measurements[0].epoch == 1700000000.0
        assert ds.measurements[0].timestamp == datetime.fromtimestamp(1700000000, tz=timezone.utc)
        assert ds.metadata["source"] == "csv"
        assert ds.metadata["file_path"] == str(csv_file)

    def test_explicit_format_wins_over_extension(self, loader: DataLoader, tmp_path: Path) -> None:
        path = tmp_path / "data.txt"
        path.write_text("epoch,rtt\n1700000000,5\n1700000060,6\n")
        ds = loader.load_from_file(path, file_format="csv")
        assert len(ds) == 2

    def test_missing_columns(self, loader: DataLoader, tmp_path: Path) -> None:
        path = tmp_path / "bad.csv"
        path.write_text("time,ms\n1,2\n")
        with pytest.raises(ValueError, match="epoch"):
            loader.load_from_file(path)


class TestDataFrame:
    @pytest.mark.parametrize("column", ["values", "rtt_value", "rtt", "latency"])
    def test_accepted_value_columns(self, loader: DataLoader, column: str) -> None:
        df = pd.DataFrame({"epoch": [1.0, 2.0], column: [10.0, 11.0]})
        ds = loader.load_from_dataframe(df)
        assert [m.rtt_value for m in ds.measurements] == [10.0, 11.0]

    def test_source_and_destination_are_kept(self, loader: DataLoader) -> None:
        df = pd.DataFrame({"epoch": [1.0], "values": [10.0], "source": ["a"], "destination": ["b"]})
        m = loader.load_from_dataframe(df).measurements[0]
        assert (m.source, m.destination) == ("a", "b")

    def test_rejects_unknown_value_column(self, loader: DataLoader) -> None:
        with pytest.raises(ValueError, match="RTT column named one of 'values'"):
            loader.load_from_dataframe(pd.DataFrame({"epoch": [1.0], "ms": [1.0]}))

    def test_rejects_missing_epoch_column(self, loader: DataLoader) -> None:
        with pytest.raises(ValueError, match="'epoch' column"):
            loader.load_from_dataframe(pd.DataFrame({"time": [1.0], "values": [1.0]}))

    # --- the input contract, validated once at the edge (docs/INPUT_FORMATS.md)

    def test_missing_values_are_dropped_and_counted(
        self, loader: DataLoader, caplog: pytest.LogCaptureFixture
    ) -> None:
        df = pd.DataFrame({"epoch": [1.0, 2.0, np.nan, 4.0], "values": [10.0, np.nan, 12.0, 13.0]})
        with caplog.at_level("WARNING", logger="jitterbug"):
            ds = loader.load_from_dataframe(df)
        assert [m.epoch for m in ds.measurements] == [1.0, 4.0]
        assert ds.metadata["dropped_rows"] == {"missing": 2, "non_positive": 0, "too_large": 0}
        assert ds.metadata["total_rows"] == 4
        assert "Dropping 2 row(s) with missing RTT" in caplog.text

    def test_out_of_range_rtts_are_dropped(self, loader: DataLoader) -> None:
        df = pd.DataFrame({"epoch": [1.0, 2.0, 3.0, 4.0], "values": [10.0, 0.0, -5.0, 20_000.0]})
        ds = loader.load_from_dataframe(df)
        assert [m.rtt_value for m in ds.measurements] == [10.0]
        assert ds.metadata["dropped_rows"] == {"missing": 0, "non_positive": 2, "too_large": 1}

    def test_unsorted_rows_are_sorted_stably(
        self, loader: DataLoader, caplog: pytest.LogCaptureFixture
    ) -> None:
        df = pd.DataFrame(
            {
                "epoch": [3.0, 1.0, 2.0, 2.0],
                "values": [30.0, 10.0, 20.0, 21.0],
                "source": list("abcd"),
            }
        )
        with caplog.at_level("WARNING", logger="jitterbug"):
            ds = loader.load_from_dataframe(df)
        assert [m.epoch for m in ds.measurements] == [1.0, 2.0, 2.0, 3.0]
        assert [m.source for m in ds.measurements] == ["b", "c", "d", "a"]  # ties keep order
        assert ds.metadata["sorted_on_load"] is True
        assert "not in time order" in caplog.text

    def test_sorted_input_is_not_flagged(self, loader: DataLoader) -> None:
        ds = loader.load_from_dataframe(pd.DataFrame({"epoch": [1.0, 2.0], "values": [1.0, 2.0]}))
        assert ds.metadata["sorted_on_load"] is False
        assert ds.metadata["dropped_rows"] == {"missing": 0, "non_positive": 0, "too_large": 0}

    @pytest.mark.parametrize("column", ["epoch", "values"])
    def test_non_numeric_values_are_an_error(self, loader: DataLoader, column: str) -> None:
        df = pd.DataFrame({"epoch": [1.0, 2.0], "values": [10.0, 11.0]}).astype(object)
        df.loc[1, column] = "12:00"
        with pytest.raises(ValueError, match=f"Column '{column}' has 1 non-numeric value"):
            loader.load_from_dataframe(df)

    def test_all_rows_invalid_is_an_error(self, loader: DataLoader) -> None:
        with pytest.raises(ValueError, match="No valid RTT rows"):
            loader.load_from_dataframe(pd.DataFrame({"epoch": [1.0, 2.0], "values": [0.0, np.nan]}))

    def test_missing_source_cells_become_none(self, loader: DataLoader) -> None:
        df = pd.DataFrame({"epoch": [1.0, 2.0], "values": [1.0, 2.0], "source": ["a", None]})
        assert [m.source for m in loader.load_from_dataframe(df).measurements] == ["a", None]


class TestScamperJSON:
    def test_load_sorts_and_skips_junk(self, loader: DataLoader, scamper_file: Path) -> None:
        ds = loader.load_from_file(scamper_file)
        assert len(ds) == 2  # timeout response and non-ping record ignored, bad line skipped
        assert [m.rtt_value for m in ds.measurements] == [11.0, 12.5]  # sorted by time
        assert ds.measurements[1].epoch == 1700000100.5
        assert ds.measurements[0].source == "10.0.0.1"
        assert ds.metadata["format"] == "scamper_warts"

    def test_out_of_range_responses_are_dropped(self, loader: DataLoader, tmp_path: Path) -> None:
        path = tmp_path / "ping.json"
        responses = [
            {"rtt": 0.0, "tx": {"sec": 1700000000, "usec": 0}},  # timeout encoded as 0
            {"rtt": 12.0, "tx": {"sec": 1700000001, "usec": 0}},
            {"rtt": 60000.0, "tx": {"sec": 1700000002, "usec": 0}},  # a minute: bad sample
        ]
        record = {"type": "ping", "src": "a", "dst": "b", "responses": responses}
        path.write_text(json.dumps(record))
        ds = loader.load_from_file(path)
        assert [m.rtt_value for m in ds.measurements] == [12.0]
        assert ds.metadata["dropped_responses"] == 2

    def test_no_measurements_is_an_error(self, loader: DataLoader, tmp_path: Path) -> None:
        path = tmp_path / "empty.json"
        path.write_text('{"type": "cycle-start"}\n')
        with pytest.raises(ValueError, match="No valid RTT measurements"):
            loader.load_from_file(path)


class TestFormatInference:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [("a.csv", "csv"), ("a.json", "json"), ("a.jsonl", "json")],
    )
    def test_from_extension(self, loader: DataLoader, tmp_path: Path, name: str, expected: str):
        path = tmp_path / name
        path.write_text("")
        assert loader._infer_format(path) == expected

    def test_from_content_when_extension_is_unknown(self, loader: DataLoader, tmp_path: Path):
        js = tmp_path / "data.log"
        js.write_text('{"type": "ping"}\n')
        assert loader._infer_format(js) == "json"
        csv = tmp_path / "data.dat"
        csv.write_text("epoch,values\n")
        assert loader._infer_format(csv) == "csv"

    def test_unrecognisable_content_is_an_error(self, loader: DataLoader, tmp_path: Path):
        path = tmp_path / "data.bin"
        path.write_text("just some words\n")
        with pytest.raises(ValueError, match="Cannot infer format"):
            loader.load_from_file(path)

    def test_binary_content_is_an_error(self, loader: DataLoader, tmp_path: Path):
        """A non-UTF-8 file must follow the same error path, not leak UnicodeDecodeError."""
        path = tmp_path / "data.bin"
        path.write_bytes(b"\xff\xfe\x00\x01binary")
        with pytest.raises(ValueError, match="Cannot infer format"):
            loader.load_from_file(path)

    def test_unknown_format_is_rejected(self, loader: DataLoader, tmp_path: Path) -> None:
        path = tmp_path / "data.csv"
        path.write_text("epoch,values\n1,2\n")
        with pytest.raises(ValueError, match="Unsupported"):
            loader.load_from_file(path, file_format="xml")


@pytest.fixture
def influxdb_client(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """The real package when installed, otherwise a stub so the loader's import succeeds.

    The tests patch ``InfluxDBClient`` either way, so they exercise the loader's mapping
    and clean-up logic even in CI, where the ``influx`` extra is not installed.
    """
    try:
        import influxdb_client
    except ImportError:
        influxdb_client = types.ModuleType("influxdb_client")
        influxdb_client.InfluxDBClient = MagicMock()  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "influxdb_client", influxdb_client)
    return influxdb_client


@pytest.mark.usefixtures("influxdb_client")
class TestInfluxDB:
    def test_query_result_is_mapped_to_a_dataset(self, loader: DataLoader) -> None:
        frame = pd.DataFrame(
            {
                "_time": pd.to_datetime(["2024-01-01T00:00:00Z", "2024-01-01T00:01:00Z"]),
                "_value": [30.0, 31.0],
            }
        )
        client = MagicMock()
        client.query_api.return_value.query_data_frame.return_value = frame
        with patch("influxdb_client.InfluxDBClient", return_value=client) as ctor:
            ds = loader.load_from_influxdb(
                url="http://influx:8086", token="t", org="o", bucket="b", query="q"
            )
        ctor.assert_called_once_with(url="http://influx:8086", token="t", org="o")
        client.query_api.return_value.query_data_frame.assert_called_once_with("q")
        client.close.assert_called_once()
        assert [m.rtt_value for m in ds.measurements] == [30.0, 31.0]
        assert ds.measurements[0].epoch == datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()

    def test_epoch_is_exact_at_microsecond_resolution(self, loader: DataLoader) -> None:
        """Regression: `astype(int) / 1e9` assumed nanoseconds and was 1000x off when
        pandas parsed `_time` at microsecond resolution (the pandas 3 default)."""
        times = pd.Series(pd.to_datetime(["2024-01-01T00:00:00.250Z"])).astype(
            "datetime64[us, UTC]"
        )
        frame = pd.DataFrame({"_time": times, "_value": [1.0]})
        client = MagicMock()
        client.query_api.return_value.query_data_frame.return_value = frame
        with patch("influxdb_client.InfluxDBClient", return_value=client):
            ds = loader.load_from_influxdb(url="u", token="t", org="o", bucket="b", query="q")
        assert ds.measurements[0].epoch == 1704067200.25

    def test_multiple_tables_are_concatenated(self, loader: DataLoader) -> None:
        t1 = pd.DataFrame({"_time": pd.to_datetime(["2024-01-01T00:00:00Z"]), "_value": [1.0]})
        t2 = pd.DataFrame({"_time": pd.to_datetime(["2024-01-01T00:01:00Z"]), "_value": [2.0]})
        client = MagicMock()
        client.query_api.return_value.query_data_frame.return_value = [t1, t2]
        with patch("influxdb_client.InfluxDBClient", return_value=client):
            ds = loader.load_from_influxdb(url="u", token="t", org="o", bucket="b", query="q")
        assert len(ds) == 2

    def test_missing_value_column(self, loader: DataLoader) -> None:
        frame = pd.DataFrame({"_time": pd.to_datetime(["2024-01-01T00:00:00Z"]), "x": [1.0]})
        client = MagicMock()
        client.query_api.return_value.query_data_frame.return_value = frame
        with (
            patch("influxdb_client.InfluxDBClient", return_value=client),
            pytest.raises(ValueError, match="No RTT value column"),
        ):
            loader.load_from_influxdb(url="u", token="t", org="o", bucket="b", query="q")
        client.close.assert_called_once()  # closed even on failure


class TestValidateData:
    def test_metrics(self, loader: DataLoader, rng: np.random.Generator) -> None:
        values = np.concatenate([20.0 + rng.normal(0, 0.1, 99), [200.0]])  # one outlier
        report = loader.validate_data(RTTDataset(measurements=make_measurements(values)))
        assert report["valid"] is True
        m = report["metrics"]
        assert m["total_measurements"] == 100
        assert m["time_ordered"] is True
        assert m["has_duplicates"] is False
        assert m["rtt_statistics"]["outliers"] == 1
        json.dumps(report)  # every value is a plain Python type
        assert_allclose(m["average_interval_seconds"], 60.0)
        assert_allclose(m["duration_seconds"], 99 * 60.0)

    def test_duplicates_are_reported(self, loader: DataLoader) -> None:
        ms = make_measurements(np.array([1.0, 2.0, 3.0]))
        ms[1] = ms[0].model_copy(update={"rtt_value": 2.0})  # same epoch twice
        report = loader.validate_data(RTTDataset(measurements=ms))
        assert report["metrics"]["has_duplicates"] is True


# ------------------------------------------------------------------------- exporters


@pytest.fixture
def results() -> CongestionInferenceResult:
    from jitterbug.models import CongestionInference

    periods = [(0.0, 600.0, True), (600.0, 1200.0, False), (1200.0, 1500.0, True)]
    inferences = [
        CongestionInference(
            start_timestamp=datetime.fromtimestamp(s, tz=timezone.utc),
            end_timestamp=datetime.fromtimestamp(e, tz=timezone.utc),
            start_epoch=s,
            end_epoch=e,
            is_congested=c,
            confidence=0.9 if c else 0.0,
            latency_jump=jump(s, e, c),
            jitter_analysis=jitter(s, e, c),
        )
        for s, e, c in periods
    ]
    return CongestionInferenceResult(inferences=inferences, metadata={"change_points": 4})


class TestExporters:
    def test_json_round_trip(self, results: CongestionInferenceResult, tmp_path: Path) -> None:
        out = tmp_path / "r.json"
        ResultExporter().export_to_json(results, out)
        data = json.loads(out.read_text())
        assert len(data["inferences"]) == 3
        assert data["inferences"][0]["is_congested"] is True
        assert data["metadata"]["change_points"] == 4
        # datetimes are serialised as ISO strings
        datetime.fromisoformat(data["inferences"][0]["start_timestamp"])

    def test_csv_has_the_v1_columns(self, results: CongestionInferenceResult, tmp_path: Path):
        out = tmp_path / "r.csv"
        ResultExporter().export_to_csv(results, out)
        df = pd.read_csv(out)
        assert list(df.columns)[:3] == ["starts", "ends", "congestion"]
        assert df["congestion"].tolist() == [True, False, True]
        assert_allclose(df["starts"], [0.0, 600.0, 1200.0])

    def test_summary(self, results: CongestionInferenceResult, tmp_path: Path) -> None:
        out = tmp_path / "summary.json"
        ResultExporter().export_summary(results, out)
        summary = json.loads(out.read_text())
        assert summary["total_periods"] == 3
        assert summary["congested_periods"] == 2
        assert_allclose(summary["total_congestion_duration"], 900.0)
        assert summary["time_range"] == {"start": 0.0, "end": 1500.0}

    def test_parquet_requires_an_engine(self, results: CongestionInferenceResult, tmp_path: Path):
        pytest.importorskip("pyarrow")
        out = tmp_path / "r.parquet"
        ResultExporter().export_to_parquet(results, out)
        assert len(pd.read_parquet(out)) == 3

    def test_analyzer_save_results_dispatches_on_format(
        self, results: CongestionInferenceResult, tmp_path: Path
    ) -> None:
        from jitterbug import JitterbugAnalyzer, JitterbugConfig

        analyzer = JitterbugAnalyzer(JitterbugConfig())
        analyzer.save_results(results, tmp_path / "a.csv", "csv")
        analyzer.save_results(results, tmp_path / "b.json", "json")
        assert (tmp_path / "a.csv").read_text().startswith("starts,ends,congestion")
        assert json.loads((tmp_path / "b.json").read_text())["metadata"]["change_points"] == 4
