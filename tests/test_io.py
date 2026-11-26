"""Unit tests for ``jitterbug.io``: loaders, format inference, validation and exporters."""

import importlib
import importlib.util
import json
import sys
import types
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from jitterbug.io import DataLoader, ResultExporter, compression
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
        assert ds.measurements[0].timestamp == datetime.fromtimestamp(1700000000, tz=UTC)
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

    def test_empty_strings_count_as_missing(self, loader: DataLoader) -> None:
        """A frame built by hand (not read_csv) may carry "" for a blank cell."""
        df = pd.DataFrame({"epoch": [1.0, 2.0, 3.0], "values": [10.0, "", " "]}, dtype=object)
        ds = loader.load_from_dataframe(df)
        assert [m.rtt_value for m in ds.measurements] == [10.0]
        assert ds.metadata["dropped_rows"]["missing"] == 2

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

    def test_unrecognizable_content_is_an_error(self, loader: DataLoader, tmp_path: Path):
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
        assert ds.measurements[0].epoch == datetime(2024, 1, 1, tzinfo=UTC).timestamp()

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
            start_timestamp=datetime.fromtimestamp(s, tz=UTC),
            end_timestamp=datetime.fromtimestamp(e, tz=UTC),
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
        # datetimes are serialized as ISO strings
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


# ----------------------------------------------------------------------- compression

ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
ZSTD_BACKENDS = [
    pytest.param(
        name,
        marks=pytest.mark.skipif(
            importlib.util.find_spec(name.split(".")[0]) is None
            or (name == "compression.zstd" and sys.version_info < (3, 14)),
            reason=f"{name} not available",
        ),
    )
    for name in ("zstandard", "compression.zstd")
]


@pytest.fixture(params=ZSTD_BACKENDS)
def zstd_backend(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """Run the test once per available Zstandard codec."""
    module = importlib.import_module(request.param)
    monkeypatch.setattr(compression, "_zstd_module", lambda: module)
    return str(request.param)


@pytest.fixture
def no_zstd(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("compression.zstd", "zstandard"):
        monkeypatch.setitem(sys.modules, name, None)


def _write_zst(path: Path, text: str) -> Path:
    with compression.open_text(path, "w", newline="") as f:
        f.write(text)
    assert path.read_bytes()[:4] == ZSTD_MAGIC
    return path


class TestCompression:
    @pytest.mark.parametrize(
        ("name", "zstd", "inner"),
        [
            ("r.json.zst", True, ".json"),
            ("R.CSV.ZST", True, ".csv"),
            ("r.csv", False, ".csv"),
            ("r.zst", True, ""),
            ("r", False, ""),
        ],
    )
    def test_suffixes(self, name: str, zstd: bool, inner: str) -> None:
        assert compression.is_zstd(name) is zstd
        assert compression.inner_suffix(name) == inner

    @pytest.mark.parametrize(
        ("export", "name"),
        [("export_to_json", "r.json"), ("export_to_csv", "r.csv"), ("export_summary", "s.json")],
    )
    def test_exporters_write_the_same_content_compressed(
        self,
        zstd_backend: str,
        results: CongestionInferenceResult,
        tmp_path: Path,
        export: str,
        name: str,
    ) -> None:
        plain, packed = tmp_path / name, tmp_path / f"{name}.zst"
        getattr(ResultExporter(), export)(results, plain)
        getattr(ResultExporter(), export)(results, packed)
        assert packed.read_bytes()[:4] == ZSTD_MAGIC
        with compression.open_text(packed, newline="") as f:
            assert f.read() == plain.read_text(encoding="utf-8")

    @pytest.mark.parametrize("fmt", ["json", "csv"])
    def test_analyzer_save_results_compresses(
        self, zstd_backend: str, results: CongestionInferenceResult, tmp_path: Path, fmt: str
    ) -> None:
        from jitterbug import JitterbugAnalyzer, JitterbugConfig

        analyzer = JitterbugAnalyzer(JitterbugConfig())
        plain, packed = tmp_path / f"r.{fmt}", tmp_path / f"r.{fmt}.zst"
        analyzer.save_results(results, plain, fmt)
        analyzer.save_results(results, packed, fmt)
        with compression.open_text(packed, newline="") as f:
            assert f.read() == plain.read_text(encoding="utf-8")

    def test_parquet_with_zst_suffix_is_rejected(
        self, results: CongestionInferenceResult, tmp_path: Path
    ) -> None:
        from jitterbug import JitterbugAnalyzer, JitterbugConfig

        out = tmp_path / "r.parquet.zst"
        with pytest.raises(ValueError, match="already compressed"):
            ResultExporter().export_to_parquet(results, out)
        with pytest.raises(ValueError, match="already compressed"):
            JitterbugAnalyzer(JitterbugConfig()).save_results(results, out, "parquet")
        assert not out.exists()

    def test_compressed_csv_loads_like_the_plain_one(
        self, zstd_backend: str, loader: DataLoader, csv_file: Path, tmp_path: Path
    ) -> None:
        packed = _write_zst(tmp_path / "rtts.csv.zst", csv_file.read_text())
        plain_ds, packed_ds = loader.load_from_file(csv_file), loader.load_from_file(packed)
        assert_allclose(packed_ds.to_arrays()[1], plain_ds.to_arrays()[1])
        assert packed_ds.metadata["file_path"] == str(packed)

    def test_compressed_scamper_json_loads_like_the_plain_one(
        self, zstd_backend: str, loader: DataLoader, scamper_file: Path, tmp_path: Path
    ) -> None:
        packed = _write_zst(tmp_path / "pings.jsonl.zst", scamper_file.read_text())
        assert_allclose(
            loader.load_from_file(packed).to_arrays()[1],
            loader.load_from_file(scamper_file).to_arrays()[1],
        )

    @pytest.mark.parametrize(
        ("name", "text", "expected"),
        [
            ("rtts.csv.zst", "epoch,values\n1,2\n", "csv"),
            ("pings.json.zst", '{"type": "ping"}\n', "json"),
            ("data.zst", '{"type": "ping"}\n', "json"),
            ("data.txt.zst", "epoch,values\n1,2\n", "csv"),
        ],
    )
    def test_format_inference_ignores_the_zst_suffix(
        self,
        zstd_backend: str,
        loader: DataLoader,
        tmp_path: Path,
        name: str,
        text: str,
        expected: str,
    ) -> None:
        assert loader._infer_format(_write_zst(tmp_path / name, text)) == expected

    def test_missing_codec_raises_with_the_install_hint(
        self,
        no_zstd: None,
        loader: DataLoader,
        results: CongestionInferenceResult,
        tmp_path: Path,
    ) -> None:
        with pytest.raises(ImportError, match=r"jitterbug-inference\[zstd\]"):
            ResultExporter().export_to_json(results, tmp_path / "r.json.zst")
        packed = tmp_path / "rtts.csv.zst"
        packed.write_bytes(ZSTD_MAGIC + b"\x00" * 8)
        with pytest.raises(ImportError, match=r"jitterbug-inference\[zstd\]"):
            loader.load_from_file(packed)

    def test_plain_files_do_not_need_a_codec(
        self, no_zstd: None, results: CongestionInferenceResult, tmp_path: Path
    ) -> None:
        ResultExporter().export_to_json(results, tmp_path / "r.json")
        assert json.loads((tmp_path / "r.json").read_text())["metadata"]["change_points"] == 4

    def test_invalid_mode_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unsupported mode"):
            compression.open_text(tmp_path / "r.json", "a")


# ------------------------------------------------------- input mapping, units, policies

BASE_EPOCHS = [1700000000.0, 1700000060.5, 1700000120.0]


def _load(df: pd.DataFrame, **settings: object) -> RTTDataset:
    from jitterbug.models import InputConfig

    return DataLoader(InputConfig(**settings)).load_from_dataframe(df)  # type: ignore[arg-type]


class TestInputMapping:
    def test_custom_column_names(self) -> None:
        df = pd.DataFrame(
            {"ts": BASE_EPOCHS, "delay": [5.0, 6.0, 7.0], "probe": list("abc"), "dst": list("xyz")}
        )
        ds = _load(
            df,
            time_column="ts",
            rtt_column="delay",
            source_column="probe",
            destination_column="dst",
        )
        assert [m.rtt_value for m in ds.measurements] == [5.0, 6.0, 7.0]
        assert [(m.source, m.destination) for m in ds.measurements][0] == ("a", "x")
        assert ds.metadata["time_column"] == "ts"
        assert ds.metadata["rtt_column"] == "delay"

    def test_explicit_rtt_column_wins_over_synonyms(self) -> None:
        df = pd.DataFrame({"epoch": [1.0], "values": [1.0], "other": [9.0]})
        assert _load(df, rtt_column="other").measurements[0].rtt_value == 9.0

    def test_missing_configured_columns_are_errors(self) -> None:
        df = pd.DataFrame({"epoch": [1.0], "values": [1.0]})
        with pytest.raises(ValueError, match=r"no 'delay' column \(rtt_column\)"):
            _load(df, rtt_column="delay")
        with pytest.raises(ValueError, match=r"no 'ts' column \(time_column\)"):
            _load(df, time_column="ts")

    def test_default_metadata_records_the_units(self) -> None:
        ds = _load(pd.DataFrame({"epoch": BASE_EPOCHS, "values": [1.0, 2.0, 3.0]}))
        assert ds.metadata["time_unit"] == "s"
        assert ds.metadata["time_unit_inferred"] is False
        assert ds.metadata["rtt_unit"] == "ms"

    @pytest.mark.parametrize(("unit", "scale"), [("s", 1), ("ms", 1e3), ("us", 1e6), ("ns", 1e9)])
    def test_numeric_time_units(self, unit: str, scale: float) -> None:
        df = pd.DataFrame({"epoch": [e * scale for e in BASE_EPOCHS], "values": [1.0, 2.0, 3.0]})
        ds = _load(df, time_unit=unit)
        assert_allclose([m.epoch for m in ds.measurements], BASE_EPOCHS, rtol=0, atol=1e-6)
        assert ds.metadata["time_unit"] == unit

    def test_integer_nanoseconds_keep_microsecond_precision(self) -> None:
        df = pd.DataFrame({"epoch": [1_700_000_000_123_456_000], "values": [1.0]})
        assert _load(df, time_unit="ns").measurements[0].epoch == 1700000000.123456

    def test_iso_datetimes(self) -> None:
        df = pd.DataFrame(
            {
                "time": [
                    "2024-01-01T00:00:00Z",
                    "2024-01-01T02:00:01+02:00",  # 00:00:01 UTC
                    "2024-01-01 00:00:02.250",  # naive: UTC
                ],
                "rtt": [1.0, 2.0, 3.0],
            }
        )
        ds = _load(df, time_column="time", time_unit="iso")
        start = datetime(2024, 1, 1, tzinfo=UTC).timestamp()
        assert [m.epoch for m in ds.measurements] == [start, start + 1, start + 2.25]
        assert ds.measurements[0].timestamp == datetime(2024, 1, 1, tzinfo=UTC)

    def test_iso_accepts_a_datetime_column(self) -> None:
        times = pd.to_datetime(["2024-01-01T00:00:00", "2024-01-01T00:01:00"])  # naive
        ds = _load(pd.DataFrame({"epoch": times, "values": [1.0, 2.0]}), time_unit="iso")
        assert ds.measurements[1].epoch == datetime(2024, 1, 1, 0, 1, tzinfo=UTC).timestamp()

    def test_datetime_column_with_a_numeric_unit_is_an_error(self) -> None:
        times = pd.to_datetime(["2024-01-01T00:00:00Z"])
        with pytest.raises(ValueError, match="holds datetimes, not s numbers"):
            _load(pd.DataFrame({"epoch": times, "values": [1.0]}))

    def test_iso_blank_cells_are_missing(self) -> None:
        df = pd.DataFrame({"epoch": ["2024-01-01T00:00:00Z", ""], "values": [1.0, 2.0]})
        ds = _load(df, time_unit="iso")
        assert len(ds) == 1
        assert ds.metadata["dropped_rows"]["missing"] == 1

    def test_iso_rejects_garbage_and_numbers(self) -> None:
        bad = pd.DataFrame({"epoch": ["2024-01-01T00:00:00Z", "yesterday"], "values": [1.0, 2.0]})
        with pytest.raises(ValueError, match="1 value.*not ISO 8601.*'yesterday'"):
            _load(bad, time_unit="iso")
        numbers = pd.DataFrame({"epoch": BASE_EPOCHS, "values": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="holds numbers, not datetimes"):
            _load(numbers, time_unit="iso")

    def test_datetime_strings_with_a_numeric_unit_get_a_hint(self) -> None:
        df = pd.DataFrame({"epoch": ["2024-01-01T00:00:00Z"], "values": [1.0]})
        with pytest.raises(ValueError, match="non-numeric.*set time_unit to iso or auto"):
            _load(df)

    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            ([e for e in BASE_EPOCHS], "s"),
            ([e * 1e3 for e in BASE_EPOCHS], "ms"),
            ([e * 1e6 for e in BASE_EPOCHS], "us"),
            ([int(e * 1e9) for e in BASE_EPOCHS], "ns"),
            ([0.0, 60.0, 120.0], "s"),  # relative seconds are small numbers too
            (["2023-11-14T22:13:20Z", "2023-11-14T22:14:20.5Z", "2023-11-14T22:15:20Z"], "iso"),
        ],
    )
    def test_auto_time_unit(
        self, values: list[object], expected: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        df = pd.DataFrame({"epoch": values, "values": [1.0, 2.0, 3.0]})
        with caplog.at_level("INFO", logger="jitterbug"):
            ds = _load(df, time_unit="auto")
        assert ds.metadata["time_unit"] == expected
        assert ds.metadata["time_unit_inferred"] is True
        assert f"time_unit auto -> {expected}" in caplog.text
        if expected != "s" or values[0] != 0.0:
            assert_allclose([m.epoch for m in ds.measurements], BASE_EPOCHS, rtol=0, atol=1e-6)

    def test_auto_reads_a_datetime_column_as_iso(self) -> None:
        times = pd.to_datetime(["2024-01-01T00:00:00Z"])
        ds = _load(pd.DataFrame({"epoch": times, "values": [1.0]}), time_unit="auto")
        assert ds.metadata["time_unit"] == "iso"

    @pytest.mark.parametrize(
        ("unit", "raw"), [("ms", 1.5), ("us", 1500.0), ("s", 0.0015), ("ns", 1_500_000.0)]
    )
    def test_rtt_units_are_converted_to_milliseconds(self, unit: str, raw: float) -> None:
        ds = _load(pd.DataFrame({"epoch": [1.0], "values": [raw]}), rtt_unit=unit)
        assert_allclose(ds.measurements[0].rtt_value, 1.5)
        assert ds.metadata["rtt_unit"] == unit

    def test_range_check_applies_after_conversion(self) -> None:
        df = pd.DataFrame({"epoch": [1.0, 2.0], "values": [0.02, 20.0]})  # 20 ms, 20 s
        ds = _load(df, rtt_unit="s")
        assert [m.rtt_value for m in ds.measurements] == [20.0]
        assert ds.metadata["dropped_rows"]["too_large"] == 1


class TestRowPolicies:
    FRAME = pd.DataFrame(
        {
            "epoch": [1.0, 2.0, 2.0, 2.0, 3.0],
            "values": [10.0, 30.0, 20.0, 20.0, 40.0],
            "source": list("abcde"),
        }
    )

    def test_duplicates_are_kept_and_counted_by_default(self) -> None:
        ds = _load(self.FRAME)
        assert len(ds) == 5
        assert ds.metadata["duplicate_rows"] == 2
        assert ds.metadata["duplicates_removed"] == 0

    @pytest.mark.parametrize(
        ("policy", "sources"), [("first", ["a", "b", "e"]), ("min", ["a", "c", "e"])]
    )
    def test_duplicates_are_thinned(
        self, policy: str, sources: list[str], caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING", logger="jitterbug"):
            ds = _load(self.FRAME, duplicates=policy)
        assert [m.source for m in ds.measurements] == sources  # min: first of the tied 20s
        assert ds.metadata["duplicates_removed"] == 2
        assert f"duplicates: {policy}" in caplog.text

    @pytest.mark.parametrize(("policy", "kept"), [("first", [30.0, 10.0]), ("min", [5.0, 10.0])])
    def test_duplicates_follow_file_order_after_sorting(
        self, policy: str, kept: list[float]
    ) -> None:
        df = pd.DataFrame({"epoch": [2.0, 1.0, 1.0, 2.0], "values": [10.0, 30.0, 5.0, 11.0]})
        ds = _load(df, duplicates=policy)
        assert [m.epoch for m in ds.measurements] == [1.0, 2.0]
        assert [m.rtt_value for m in ds.measurements] == kept

    def test_duplicates_error(self) -> None:
        with pytest.raises(ValueError, match=r"2 row\(s\) share a timestamp.*2\.0 s"):
            _load(self.FRAME, duplicates="error")
        _load(self.FRAME.drop_duplicates("epoch"), duplicates="error")  # no duplicate: fine

    def test_out_of_range_error(self) -> None:
        df = pd.DataFrame({"epoch": [1.0, 2.0, 3.0], "values": [10.0, -1.0, np.nan]})
        with pytest.raises(ValueError, match=r"1 RTT value\(s\) outside.*-1\.0 ms"):
            _load(df, out_of_range="error")
        ok = _load(df.iloc[[0, 2]], out_of_range="error")  # missing values are still dropped
        assert len(ok) == 1

    def test_unsorted_error(self) -> None:
        df = pd.DataFrame({"epoch": [1.0, 3.0, 2.0], "values": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match=r"not in time order: time 2\.0 s comes after 3\.0"):
            _load(df, unsorted="error")
        assert _load(df.iloc[[0, 2, 1]], unsorted="error").metadata["sorted_on_load"] is False


# ---------------------------------------------------------------------- reader registry


class TestReaders:
    def test_builtin_formats_are_registered(self) -> None:
        from jitterbug.io.readers import available_formats

        assert set(available_formats()) >= {"csv", "tsv", "json", "scamper", "jsonl"}

    def test_unknown_format_lists_the_available_ones(self, loader: DataLoader, csv_file: Path):
        with pytest.raises(ValueError, match="Unsupported file format: xml.*csv, tsv"):
            loader.load_from_file(csv_file, file_format="xml")

    def test_registering_a_reader_is_enough_to_load_a_format(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from jitterbug.io.readers import RawData, Reader, base, register_reader

        monkeypatch.setattr(base, "_REGISTRY", dict(base._REGISTRY))

        def read_pipe(path: Path, config: object) -> RawData:
            rows = [line.split("|") for line in path.read_text().splitlines()]
            return RawData(frame=pd.DataFrame(rows[1:], columns=rows[0]))

        register_reader(Reader(name="pipe", label="pipe", read=read_pipe, extensions=(".pipe",)))
        path = tmp_path / "rtts.pipe"
        path.write_text("epoch|values\n1|2.5\n2|3.5\n")
        ds = DataLoader().load_from_file(path)
        assert [m.rtt_value for m in ds.measurements] == [2.5, 3.5]
        assert ds.metadata["source"] == "pipe"

    def test_names_and_extensions_are_unique(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from jitterbug.io.readers import Reader, base, register_reader
        from jitterbug.io.readers.delimited import read_csv

        monkeypatch.setattr(base, "_REGISTRY", dict(base._REGISTRY))
        with pytest.raises(ValueError, match="already registered"):
            register_reader(Reader(name="csv", label="x", read=read_csv))
        with pytest.raises(ValueError, match="already belong to 'csv'"):
            register_reader(Reader(name="other", label="x", read=read_csv, extensions=(".csv",)))

    @pytest.mark.parametrize(
        ("name", "first_line", "expected"),
        [
            ("a.tsv", "", "tsv"),
            ("a.TAB", "", "tsv"),
            ("a.ndjson", "", "json"),
            ("a.txt", "epoch\tvalues", "tsv"),
            ("a.txt", "epoch;values", "csv"),
            ("a.txt", '{"epoch": 1, "values": 2}', "json"),  # has commas, still JSON
            ("a.txt", "\n\nepoch,values", "csv"),  # leading blank lines are skipped
        ],
    )
    def test_format_inference(
        self, loader: DataLoader, tmp_path: Path, name: str, first_line: str, expected: str
    ) -> None:
        path = tmp_path / name
        path.write_text(first_line + "\n")
        assert loader._infer_format(path) == expected


class TestDelimited:
    ROWS = [("epoch", "values"), ("1700000000", "20.5"), ("1700000060", "21.0")]

    def _write(self, path: Path, sep: str) -> Path:
        path.write_text("\n".join(sep.join(r) for r in self.ROWS) + "\n")
        return path

    @pytest.mark.parametrize(
        ("name", "sep", "fmt"),
        [
            ("r.tsv", "\t", "tsv"),
            ("r.csv", "\t", "csv"),  # tab-separated with a .csv name: detected
            ("r.csv", ";", "csv"),
            ("r.dat", "\t", "tsv"),
            ("r.dat", ";", "csv"),
        ],
    )
    def test_separator_is_detected(self, tmp_path: Path, name: str, sep: str, fmt: str) -> None:
        ds = DataLoader().load_from_file(self._write(tmp_path / name, sep))
        assert [m.rtt_value for m in ds.measurements] == [20.5, 21.0]
        assert ds.metadata["source"] == fmt
        assert ds.metadata["delimiter"] == sep

    def test_configured_delimiter_wins(self, tmp_path: Path) -> None:
        from jitterbug.models import InputConfig

        path = self._write(tmp_path / "r.csv", "|")
        ds = DataLoader(InputConfig(delimiter="|")).load_from_file(path)
        assert len(ds) == 2
        assert ds.metadata["delimiter"] == "|"

    @pytest.mark.parametrize(
        ("header", "expected"),
        [("a,b", ","), ("a\tb", "\t"), ("a;b", ";"), ("a;b;c,d", ";"), ("a", ","), ("a\tb,c", ",")],
    )
    def test_detect_delimiter(self, header: str, expected: str) -> None:
        from jitterbug.io.readers.delimited import detect_delimiter

        assert detect_delimiter(header) == expected

    def test_file_mapping_and_units(self, tmp_path: Path) -> None:
        from jitterbug.models import InputConfig

        path = tmp_path / "r.tsv"
        path.write_text("time_ms\trtt_us\n1700000000000\t20500\n1700000060000\t21000\n")
        config = InputConfig(time_column="time_ms", time_unit="ms", rtt_column="rtt_us")
        ds = DataLoader(config.model_copy(update={"rtt_unit": "us"})).load_from_file(path)
        assert [m.epoch for m in ds.measurements] == [1700000000.0, 1700000060.0]
        assert [m.rtt_value for m in ds.measurements] == [20.5, 21.0]

    def test_empty_file_is_an_error(self, tmp_path: Path) -> None:
        path = tmp_path / "r.csv"
        path.write_text("")
        with pytest.raises(ValueError, match="Failed to load CSV file"):
            DataLoader().load_from_file(path)


class TestGenericJSONL:
    @staticmethod
    def _write(path: Path, records: list[object]) -> Path:
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        return path

    def test_flat_records_are_read_like_a_table(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path / "r.jsonl",
            [
                {"epoch": 1700000060, "rtt": 21.0, "source": "a"},
                {"epoch": 1700000000, "rtt": 20.5, "source": "a", "extra": True},
            ],
        )
        ds = DataLoader().load_from_file(path)
        assert [m.rtt_value for m in ds.measurements] == [20.5, 21.0]  # sorted
        assert ds.metadata["format"] == "jsonl"
        assert ds.metadata["source"] == "jsonl"
        assert ds.metadata["sorted_on_load"] is True

    def test_fields_are_mapped_through_the_input_config(self, tmp_path: Path) -> None:
        from jitterbug.models import InputConfig

        path = self._write(
            tmp_path / "r.json",
            [{"ts": "2024-01-01T00:00:00Z", "delay_us": 1500}, {"ts": "2024-01-01T00:01:00Z"}],
        )
        config = InputConfig(time_column="ts", time_unit="auto", rtt_column="delay_us")
        ds = DataLoader(config.model_copy(update={"rtt_unit": "us"})).load_from_file(path)
        assert [m.rtt_value for m in ds.measurements] == [1.5]
        assert ds.metadata["dropped_rows"]["missing"] == 1  # second record has no RTT
        assert ds.metadata["time_unit"] == "iso"

    def test_a_type_ping_field_alone_is_not_scamper(self, tmp_path: Path) -> None:
        """RIPE Atlas and others write "type": "ping" too; scamper pings have responses."""
        path = self._write(tmp_path / "r.jsonl", [{"type": "ping", "epoch": 1, "rtt": 5.0}])
        assert DataLoader().load_from_file(path).metadata["format"] == "jsonl"

    @pytest.mark.parametrize("kind", ["list", "cycle-def", "cycle-start", "cycle-stop"])
    def test_scamper_bookkeeping_records_mark_a_scamper_file(
        self, tmp_path: Path, kind: str
    ) -> None:
        path = self._write(tmp_path / "r.jsonl", [{"type": kind}, {"epoch": 1, "rtt": 5.0}])
        with pytest.raises(ValueError, match="No valid RTT measurements found in JSON"):
            DataLoader().load_from_file(path)

    def test_explicit_formats_skip_the_content_check(
        self, tmp_path: Path, scamper_file: Path
    ) -> None:
        with pytest.raises(ValueError, match=r"Failed to load JSONL file.*no 'epoch' column"):
            DataLoader().load_from_file(scamper_file, file_format="jsonl")
        ds = DataLoader().load_from_file(scamper_file, file_format="scamper")
        assert ds.metadata["format"] == "scamper_warts"

    def test_non_object_lines_are_an_error(self, tmp_path: Path) -> None:
        path = self._write(tmp_path / "r.jsonl", [{"epoch": 1, "rtt": 5.0}, [1, 2]])
        with pytest.raises(ValueError, match=r"1 line\(s\) are not JSON objects"):
            DataLoader().load_from_file(path)

    def test_invalid_lines_are_skipped_and_counted(self, tmp_path: Path) -> None:
        path = tmp_path / "r.jsonl"
        path.write_text('{"epoch": 1, "rtt": 5.0}\nnot json\n{"epoch": 2, "rtt": 6.0}\n')
        ds = DataLoader().load_from_file(path)
        assert len(ds) == 2
        assert ds.metadata["invalid_lines"] == 1


class TestScamperUnderInputSettings:
    def test_user_mapping_does_not_apply_to_scamper(self, scamper_file: Path) -> None:
        """scamper output has a fixed schema; user column names and units are ignored."""
        from jitterbug.models import InputConfig

        config = InputConfig(time_column="ts", time_unit="ms", rtt_column="x", rtt_unit="us")
        ds = DataLoader(config).load_from_file(scamper_file)
        assert [m.rtt_value for m in ds.measurements] == [11.0, 12.5]
        assert ds.measurements[1].epoch == 1700000100.5

    def test_row_policies_do_apply_to_scamper(self, tmp_path: Path) -> None:
        from jitterbug.models import InputConfig

        record = {
            "type": "ping",
            "src": "a",
            "dst": "b",
            "responses": [{"rtt": 0.0, "tx": {"sec": 1700000000, "usec": 0}}],
        }
        path = tmp_path / "p.json"
        path.write_text(json.dumps(record))
        with pytest.raises(ValueError, match="out_of_range: error"):
            DataLoader(InputConfig(out_of_range="error")).load_from_file(path)


# --------------------------------------------------------- gzip, bzip2 and xz compression

STDLIB_CODECS = [(".gz", b"\x1f\x8b"), (".bz2", b"BZh"), (".xz", b"\xfd7zXZ\x00")]


class TestStdlibCompression:
    @pytest.mark.parametrize(("suffix", "magic"), STDLIB_CODECS)
    def test_compressed_inputs_load_like_the_plain_ones(
        self, tmp_path: Path, csv_file: Path, scamper_file: Path, suffix: str, magic: bytes
    ) -> None:
        loader = DataLoader()
        for plain, name in [(csv_file, "rtts.csv"), (scamper_file, "pings.jsonl")]:
            packed = tmp_path / f"{name}{suffix}"
            with compression.open_text(packed, "w", newline="") as f:
                f.write(plain.read_text())
            assert packed.read_bytes().startswith(magic)
            assert_allclose(
                loader.load_from_file(packed).to_arrays()[1],
                loader.load_from_file(plain).to_arrays()[1],
            )

    @pytest.mark.parametrize(("suffix", "magic"), STDLIB_CODECS)
    def test_format_is_sniffed_through_the_codec(
        self, loader: DataLoader, tmp_path: Path, suffix: str, magic: bytes
    ) -> None:
        packed = tmp_path / f"data{suffix}"
        with compression.open_text(packed, "w") as f:
            f.write("epoch\tvalues\n1\t2\n")
        assert loader._infer_format(packed) == "tsv"

    @pytest.mark.parametrize(
        ("name", "codec", "inner"),
        [
            ("r.csv.gz", "gzip", ".csv"),
            ("R.TSV.BZ2", "bzip2", ".tsv"),
            ("r.jsonl.xz", "xz", ".jsonl"),
            ("r.json.zst", "zstd", ".json"),
            ("r.csv", None, ".csv"),
        ],
    )
    def test_suffixes(self, name: str, codec: str | None, inner: str) -> None:
        assert compression.compression_of(name) == codec
        assert compression.is_compressed(name) is (codec is not None)
        assert compression.inner_suffix(name) == inner

    def test_corrupt_archive_is_a_value_error(self, loader: DataLoader, tmp_path: Path) -> None:
        packed = tmp_path / "rtts.csv.gz"
        packed.write_bytes(b"not gzip at all")
        with pytest.raises(ValueError, match="Failed to load CSV file"):
            loader.load_from_file(packed)

    @pytest.mark.parametrize("suffix", [".gz", ".bz2", ".xz"])
    def test_compressed_parquet_is_rejected(
        self, results: CongestionInferenceResult, tmp_path: Path, suffix: str
    ) -> None:
        with pytest.raises(ValueError, match=f"already compressed; drop the {suffix} suffix"):
            ResultExporter().export_to_parquet(results, tmp_path / f"r.parquet{suffix}")


def test_analyzer_uses_the_input_config(tmp_path: Path) -> None:
    from jitterbug import JitterbugAnalyzer, JitterbugConfig
    from jitterbug.models import InputConfig

    config = JitterbugConfig(input=InputConfig(time_column="ts", time_unit="ms"))
    analyzer = JitterbugAnalyzer(config)
    assert analyzer.data_loader.input_config.time_column == "ts"
    df = pd.DataFrame({"ts": [1_700_000_000_000.0], "values": [1.0]})
    assert analyzer.data_loader.load_from_dataframe(df).measurements[0].epoch == 1700000000.0
