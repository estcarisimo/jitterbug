"""Readers added on the registry: Parquet/Arrow, RIPE Atlas, ping and fping output."""

import bz2
import gzip
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from jitterbug.cli.main import app
from jitterbug.io import DataLoader
from jitterbug.io.readers import infer_format
from jitterbug.io.readers.atlas import read_atlas
from jitterbug.io.readers.json_records import looks_like_json
from jitterbug.io.readers.ping import ProbeLine, parse_fping_line, parse_ping_line
from jitterbug.models.config import InputConfig

runner = CliRunner()

T0 = 1_700_000_000


@pytest.fixture
def frame() -> pd.DataFrame:
    """Three samples in the default schema, out of order."""
    return pd.DataFrame({"epoch": [T0 + 60.0, T0 + 0.0, T0 + 120.0], "values": [21.0, 20.5, 19.5]})


# ------------------------------------------------------------------- Parquet and Arrow


@pytest.fixture
def pyarrow() -> object:
    return pytest.importorskip("pyarrow")


def _write_arrow(df: pd.DataFrame, path: Path, kind: str) -> Path:
    import pyarrow as pa
    import pyarrow.feather as feather

    table = pa.Table.from_pandas(df, preserve_index=False)
    if kind == "parquet":
        import pyarrow.parquet as pq

        pq.write_table(table, path)
    elif kind == "feather2":
        feather.write_feather(df, path, version=2)
    else:  # IPC stream
        with pa.OSFile(str(path), "wb") as sink, pa.ipc.new_stream(sink, table.schema) as w:
            w.write_table(table)
    return path


@pytest.mark.usefixtures("pyarrow")
class TestColumnar:
    @pytest.mark.parametrize(
        ("name", "kind", "expected"),
        [
            ("rtts.parquet", "parquet", "parquet"),
            ("rtts.PQ", "parquet", "parquet"),
            ("rtts.arrow", "feather2", "arrow"),
            ("rtts.feather", "feather2", "arrow"),
            ("rtts.arrows", "stream", "arrow"),
        ],
    )
    def test_loads_by_extension(
        self, tmp_path: Path, frame: pd.DataFrame, name: str, kind: str, expected: str
    ) -> None:
        path = _write_arrow(frame, tmp_path / name, kind)
        assert infer_format(path) == expected
        ds = DataLoader().load_from_file(path)
        assert [m.rtt_value for m in ds.measurements] == [20.5, 21.0, 19.5]  # sorted by time
        assert ds.metadata["source"] == expected
        assert ds.metadata["sorted_on_load"] is True

    @pytest.mark.parametrize(
        ("kind", "expected"),
        [("parquet", "parquet"), ("feather2", "arrow"), ("stream", "arrow")],
    )
    def test_detected_by_magic_bytes(
        self, tmp_path: Path, frame: pd.DataFrame, kind: str, expected: str
    ) -> None:
        path = _write_arrow(frame, tmp_path / "rtts.dat", kind)
        assert infer_format(path) == expected
        assert len(DataLoader().load_from_file(path)) == 3

    @pytest.mark.parametrize("tz", [None, "UTC", "Europe/Paris"])
    def test_timestamp_column_needs_no_time_unit(self, tmp_path: Path, tz: str | None) -> None:
        times = pd.to_datetime([T0, T0 + 60], unit="s", utc=True)
        times = times.tz_convert(tz) if tz else times.tz_localize(None)
        df = pd.DataFrame({"epoch": times, "rtt": [10.0, 11.0]})
        ds = DataLoader().load_from_file(_write_arrow(df, tmp_path / "t.parquet", "parquet"))
        assert [m.epoch for m in ds.measurements] == [T0, T0 + 60]
        assert ds.metadata["time_unit"] == "iso"

    def test_input_mapping_applies(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"ts_ms": [T0 * 1000, T0 * 1000 + 500], "rtt_us": [1500, 2500]})
        path = _write_arrow(df, tmp_path / "t.arrow", "feather2")
        config = InputConfig(
            time_column="ts_ms", time_unit="ms", rtt_column="rtt_us", rtt_unit="us"
        )
        ds = DataLoader(config).load_from_file(path)
        assert [(m.epoch, m.rtt_value) for m in ds.measurements] == [(T0, 1.5), (T0 + 0.5, 2.5)]

    def test_missing_column_names_the_format(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"time": [T0], "values": [1.0]})
        path = _write_arrow(df, tmp_path / "t.parquet", "parquet")
        with pytest.raises(ValueError, match="Failed to load Parquet file.*'epoch' column"):
            DataLoader().load_from_file(path)

    @pytest.mark.parametrize("name", ["rtts.parquet.gz", "rtts.gz"])
    def test_outer_compression_is_rejected(
        self, tmp_path: Path, frame: pd.DataFrame, name: str
    ) -> None:
        plain = _write_arrow(frame, tmp_path / "plain.parquet", "parquet")
        packed = tmp_path / name
        packed.write_bytes(gzip.compress(plain.read_bytes()))
        assert infer_format(packed) == "parquet"  # decompressed magic, not text sniffing
        with pytest.raises(ValueError, match="already compressed; drop the .gz suffix"):
            DataLoader().load_from_file(packed)

    def test_corrupt_file_is_a_value_error(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.parquet"
        path.write_bytes(b"PAR1 this is not parquet")
        with pytest.raises(ValueError, match="Failed to load Parquet file"):
            DataLoader().load_from_file(path)

    def test_cli_validate_parquet(self, tmp_path: Path, frame: pd.DataFrame) -> None:
        path = _write_arrow(frame, tmp_path / "rtts.pq", "parquet")
        result = runner.invoke(app, ["validate", str(path)])
        assert result.exit_code == 0, result.output
        assert "Data validation passed" in result.output

    def test_cli_validate_arrow_with_format(self, tmp_path: Path, frame: pd.DataFrame) -> None:
        path = _write_arrow(frame, tmp_path / "rtts.bin", "stream")
        result = runner.invoke(app, ["validate", str(path), "--format", "arrow"])
        assert result.exit_code == 0, result.output
        assert "Data validation passed" in result.output


class TestColumnarWithoutPyarrow:
    """These run everywhere: they never get as far as decoding the file."""

    @pytest.mark.parametrize(
        ("head", "expected"),
        [
            (b"PAR1\x15\x04", "parquet"),
            (b"ARROW1\x00\x00", "arrow"),
            (b"FEA1\x00", "arrow"),
            (b"\xff\xff\xff\xff\x10\x00", "arrow"),
        ],
    )
    def test_magic_bytes_win_over_text_sniffing(
        self, tmp_path: Path, head: bytes, expected: str
    ) -> None:
        path = tmp_path / "data.txt"
        path.write_bytes(head + b"\xde\xad,\xbe\xef\n")
        assert infer_format(path) == expected

    def test_bz2_of_parquet_is_recognized(self, tmp_path: Path) -> None:
        path = tmp_path / "data.bz2"
        path.write_bytes(bz2.compress(b"PAR1\x15\x04 rest"))
        assert infer_format(path) == "parquet"

    @pytest.mark.parametrize(("name", "fmt"), [("t.parquet", "parquet"), ("t.feather", "arrow")])
    def test_missing_pyarrow_names_the_extra(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str, fmt: str
    ) -> None:
        for module in ("pyarrow", "pyarrow.parquet", "pyarrow.feather", "pyarrow.ipc"):
            monkeypatch.setitem(sys.modules, module, None)
        path = tmp_path / name
        path.write_bytes(b"PAR1" if fmt == "parquet" else b"ARROW1\x00\x00")
        with pytest.raises(ImportError, match=r"jitterbug-inference\[parquet\]"):
            DataLoader().load_from_file(path)


# --------------------------------------------------------------------------- RIPE Atlas


def _atlas_result(timestamp: int, entries: list[dict], **fields: object) -> dict:
    record = {
        "fw": 5080,
        "af": 4,
        "type": "ping",
        "prb_id": 6001,
        "msm_id": 1001,
        "timestamp": timestamp,
        "from": "193.0.0.1",
        "src_addr": "192.168.1.10",
        "dst_addr": "8.8.8.8",
        "dst_name": "dns.google",
        "proto": "ICMP",
        "sent": len(entries),
        "rcvd": sum("rtt" in e for e in entries),
        "result": entries,
    }
    record.update(fields)
    return record


@pytest.fixture
def atlas_results() -> list[dict]:
    """Three ping results with replies, a timeout, an error and a duplicate, and one
    traceroute result that must be skipped."""
    return [
        _atlas_result(T0 + 240, [{"rtt": 13.0}, {"x": "*"}]),
        _atlas_result(T0, [{"rtt": 12.5}, {"rtt": 12.7}, {"rtt": 12.7, "dup": 1}]),
        _atlas_result(T0 + 480, [{"error": "sendto failed: Network is unreachable"}]),
        {"type": "traceroute", "prb_id": 6001, "msm_id": 5001, "timestamp": T0, "result": []},
    ]


class TestAtlas:
    @pytest.mark.parametrize("layout", ["array", "pretty", "lines"])
    def test_json_extension_detects_atlas(
        self, tmp_path: Path, atlas_results: list[dict], layout: str
    ) -> None:
        path = tmp_path / "results.json"
        if layout == "array":
            path.write_text(json.dumps(atlas_results))
        elif layout == "pretty":
            path.write_text(json.dumps(atlas_results, indent=2))
        else:
            path.write_text("\n".join(json.dumps(r) for r in atlas_results) + "\n")
        ds = DataLoader().load_from_file(path)
        assert [(m.epoch, m.rtt_value) for m in ds.measurements] == [
            (T0, 12.5),
            (T0, 12.7),
            (T0 + 240, 13.0),
        ]
        assert {(m.source, m.destination) for m in ds.measurements} == {("193.0.0.1", "8.8.8.8")}
        meta = ds.metadata
        assert meta["source"] == "json" and meta["format"] == "ripe_atlas"
        assert (meta["timeouts"], meta["errors"], meta["duplicate_replies"]) == (1, 1, 1)
        assert (meta["ping_results"], meta["skipped_records"], meta["series"]) == (3, 1, 1)
        assert meta["dropped_responses"] == 0

    def test_probe_and_measurement_ids_are_columns(
        self, tmp_path: Path, atlas_results: list[dict]
    ) -> None:
        path = tmp_path / "results.json"
        path.write_text(json.dumps(atlas_results))
        raw = read_atlas(path, InputConfig())
        assert {"prb_id", "msm_id", "src_addr"} <= set(raw.frame.columns)
        assert set(raw.frame["prb_id"]) == {6001}

    def test_detected_by_content_and_forced(
        self, tmp_path: Path, atlas_results: list[dict]
    ) -> None:
        path = tmp_path / "atlas.txt"
        path.write_text(json.dumps(atlas_results, indent=1))
        assert infer_format(path) == "json"
        assert len(DataLoader().load_from_file(path)) == 3
        assert len(DataLoader().load_from_file(path, file_format="atlas")) == 3

    def test_compressed_lines(self, tmp_path: Path, atlas_results: list[dict]) -> None:
        path = tmp_path / "results.jsonl.gz"
        path.write_bytes(gzip.compress("\n".join(map(json.dumps, atlas_results)).encode()))
        assert DataLoader().load_from_file(path).metadata["format"] == "ripe_atlas"

    def test_source_falls_back_to_src_addr_and_destination_to_dst_name(
        self, tmp_path: Path
    ) -> None:
        record = _atlas_result(T0, [{"rtt": 5.0}], **{"from": "", "dst_addr": None})
        path = tmp_path / "r.json"
        path.write_text(json.dumps([record]))
        (m,) = DataLoader().load_from_file(path).measurements
        assert (m.source, m.destination) == ("192.168.1.10", "dns.google")

    def test_several_probes_are_pooled_with_a_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        records = [
            _atlas_result(T0, [{"rtt": 5.0}]),
            _atlas_result(T0 + 60, [{"rtt": 50.0}], prb_id=6002),
        ]
        path = tmp_path / "r.json"
        path.write_text(json.dumps(records))
        with caplog.at_level("WARNING"):
            ds = DataLoader().load_from_file(path)
        assert len(ds) == 2 and ds.metadata["series"] == 2
        assert "2 (probe, measurement, destination) combinations" in caplog.text

    def test_out_of_range_rtts_are_counted(self, tmp_path: Path) -> None:
        path = tmp_path / "r.json"
        path.write_text(json.dumps([_atlas_result(T0, [{"rtt": 5.0}, {"rtt": 0.0}])]))
        assert DataLoader().load_from_file(path).metadata["dropped_responses"] == 1

    def test_only_timeouts_is_an_error(self, tmp_path: Path) -> None:
        path = tmp_path / "r.json"
        path.write_text(json.dumps([_atlas_result(T0, [{"x": "*"}, {"x": "*"}])]))
        with pytest.raises(ValueError, match="1 ping result.*2 timeout"):
            DataLoader().load_from_file(path)

    def test_forcing_atlas_on_other_json_is_an_error(self, tmp_path: Path) -> None:
        path = tmp_path / "r.json"
        path.write_text('{"epoch": 1, "rtt": 2}\n')
        with pytest.raises(ValueError, match="No RIPE Atlas ping results"):
            DataLoader().load_from_file(path, file_format="atlas")

    def test_scamper_takes_precedence(self, tmp_path: Path, atlas_results: list[dict]) -> None:
        scamper = {
            "type": "ping",
            "src": "10.0.0.1",
            "dst": "8.8.8.8",
            "responses": [{"rtt": 1.0, "tx": {"sec": T0, "usec": 0}}],
        }
        path = tmp_path / "mixed.json"
        path.write_text("\n".join(json.dumps(r) for r in [scamper, *atlas_results]))
        assert DataLoader().load_from_file(path).metadata["format"] == "scamper_warts"

    def test_type_ping_alone_is_generic_jsonl(self, tmp_path: Path) -> None:
        path = tmp_path / "r.json"
        path.write_text('{"type": "ping", "epoch": 1700000000, "rtt": 2.0, "result": []}\n')
        assert DataLoader().load_from_file(path).metadata["format"] == "jsonl"

    def test_json_array_of_flat_objects_is_generic(self, tmp_path: Path) -> None:
        path = tmp_path / "r.json"
        path.write_text(json.dumps([{"epoch": T0, "rtt": 2.0}, {"epoch": T0 + 1, "rtt": 3.0}]))
        ds = DataLoader().load_from_file(path)
        assert ds.metadata["format"] == "jsonl" and len(ds) == 2

    def test_cli_validate_atlas(self, tmp_path: Path, atlas_results: list[dict]) -> None:
        path = tmp_path / "results.json"
        path.write_text(json.dumps(atlas_results))
        result = runner.invoke(app, ["validate", str(path), "--format", "atlas"])
        assert result.exit_code == 0, result.output
        assert "Data validation passed" in result.output


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("{", True),
        ("[", True),
        ('[{"a": 1, "b": 2}]', True),
        ("[ {", True),
        ("[]", True),
        ("[1700000000.1] 64 bytes from 8.8.8.8: icmp_seq=1 ttl=57 time=1 ms", False),
        ("[1700000000.1] 8.8.8.8 : [0], 64 bytes, 1.0 ms (1.0 avg, 0% loss)", False),
        ("epoch,values", False),
    ],
)
def test_looks_like_json(line: str, expected: bool) -> None:
    assert looks_like_json(line) is expected


# ----------------------------------------------------------------------- ping and fping

PING_D = """\
PING dns.google (8.8.8.8) 56(84) bytes of data.
[1700000000.100000] 64 bytes from dns.google (8.8.8.8): icmp_seq=1 ttl=117 time=12.3 ms
[1700000001.100000] 64 bytes from dns.google (8.8.8.8): icmp_seq=2 ttl=117 time=11.9 ms
[1700000001.200000] 64 bytes from dns.google (8.8.8.8): icmp_seq=2 ttl=117 time=12.0 ms (DUP!)
[1700000003.100000] no answer yet for icmp_seq=3
[1700000004.100000] From 192.168.1.1 icmp_seq=4 Destination Host Unreachable
[1700000005.100000] 64 bytes from dns.google (8.8.8.8): icmp_seq=5 ttl=117 time=13 ms

--- dns.google ping statistics ---
5 packets transmitted, 3 received, +1 duplicates, +1 errors, 40% packet loss, time 4005ms
rtt min/avg/max/mdev = 11.900/12.400/13.000/0.454 ms
"""

FPING_D = """\
[1700000000.10000] 8.8.8.8     : [0], 64 bytes, 12.3 ms (12.3 avg, 0% loss)
[1700000000.10000] 1.1.1.1     : [0], 64 bytes, 4.1 ms (4.1 avg, 0% loss)
[1700000001.10000] 8.8.8.8     : [1], timed out (12.3 avg, 50% loss)
[1700000001.20000] 1.1.1.1     : duplicate for [0], 64 bytes, 4.5 ms
[1700000002.10000] 8.8.8.8     : [2], 64 bytes, 11.8 ms (12.0 avg, 33% loss)

8.8.8.8     : xmt/rcv/%loss = 3/2/33%, min/avg/max = 11.8/12.0/12.3
1.1.1.1     : xmt/rcv/%loss = 1/1/0%, min/avg/max = 4.1/4.1/4.1
"""


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        (
            "[1700000000.123456] 64 bytes from 8.8.8.8: icmp_seq=1 ttl=57 time=12.3 ms",
            ProbeLine("reply", 1700000000.123456, 12.3, "8.8.8.8", 1),
        ),
        (
            "[1700000000.5] 64 bytes from dns.google (8.8.8.8): icmp_seq=7 ttl=57 time=9 ms\n",
            ProbeLine("reply", 1700000000.5, 9.0, "8.8.8.8", 7),
        ),
        (
            "[1700000000.5] 64 bytes from 2001:4860:4860::8888: icmp_seq=1 ttl=117 time=8.1 ms",
            ProbeLine("reply", 1700000000.5, 8.1, "2001:4860:4860::8888", 1),
        ),
        (  # BSD / macOS: no timestamp, sequence from 0, three decimals
            "64 bytes from 8.8.8.8: icmp_seq=0 ttl=117 time=12.345 ms",
            ProbeLine("reply", None, 12.345, "8.8.8.8", 0),
        ),
        (  # BusyBox
            "64 bytes from 8.8.8.8: seq=0 ttl=117 time=12.345 ms",
            ProbeLine("reply", None, 12.345, "8.8.8.8", 0),
        ),
        (
            "[1700000000.2] 64 bytes from 8.8.8.8: icmp_seq=1 ttl=57 time=12.4 ms (DUP!)",
            ProbeLine("duplicate", 1700000000.2, 12.4, "8.8.8.8", 1),
        ),
        ("[1700000000.2] no answer yet for icmp_seq=3", ProbeLine("timeout", 1700000000.2, seq=3)),
        ("Request timeout for icmp_seq 3", ProbeLine("timeout", seq=3)),
        (
            "[1700000000.2] From 10.0.0.1 icmp_seq=4 Destination Host Unreachable",
            ProbeLine("error", 1700000000.2, destination="10.0.0.1", seq=4),
        ),
        (
            "92 bytes from 10.0.0.1: Destination Host Unreachable",
            ProbeLine("error", destination="10.0.0.1"),
        ),
        ("PING 8.8.8.8 (8.8.8.8) 56(84) bytes of data.", None),
        ("--- 8.8.8.8 ping statistics ---", None),
        ("rtt min/avg/max/mdev = 11.900/12.400/13.000/0.454 ms", None),
        ("epoch,values", None),
    ],
)
def test_parse_ping_line(line: str, expected: ProbeLine | None) -> None:
    assert parse_ping_line(line) == expected


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        (
            "[1700000000.12345] 8.8.8.8 : [0], 64 bytes, 12.3 ms (12.3 avg, 0% loss)",
            ProbeLine("reply", 1700000000.12345, 12.3, "8.8.8.8", 0),
        ),
        (
            "dns.google  : [3], 84 bytes, 9.87 ms (10.1 avg, 0% loss)",
            ProbeLine("reply", None, 9.87, "dns.google", 3),
        ),
        (
            "[1700000000.1] 8.8.8.8 : [1], timed out (NaN avg, 100% loss)",
            ProbeLine("timeout", 1700000000.1, destination="8.8.8.8", seq=1),
        ),
        (
            "[1700000000.1] 8.8.8.8 : duplicate for [0], 64 bytes, 4.5 ms",
            ProbeLine("duplicate", 1700000000.1, 4.5, "8.8.8.8", 0),
        ),
        (
            "ICMP Host Unreachable from 10.0.0.1 for ICMP Echo sent to 10.0.0.2",
            ProbeLine("error", destination="10.0.0.2"),
        ),
        ("8.8.8.8 : xmt/rcv/%loss = 3/2/33%, min/avg/max = 11.8/12.0/12.3", None),
        ("8.8.8.8 : 12.3 - 11.8", None),  # fping -C summary
    ],
)
def test_parse_fping_line(line: str, expected: ProbeLine | None) -> None:
    assert parse_fping_line(line) == expected


class TestPingFiles:
    @pytest.mark.parametrize("name", ["ping.txt", "ping.log", "ping-output"])
    def test_ping_detected_by_content(self, tmp_path: Path, name: str) -> None:
        path = tmp_path / name
        path.write_text(PING_D)
        assert infer_format(path) == "ping"
        ds = DataLoader().load_from_file(path)
        assert [m.rtt_value for m in ds.measurements] == [12.3, 11.9, 13.0]
        assert ds.measurements[0].epoch == 1700000000.1
        assert {m.destination for m in ds.measurements} == {"8.8.8.8"}
        meta = ds.metadata
        assert (meta["source"], meta["format"]) == ("ping", "ping")
        assert (meta["timeouts"], meta["errors"], meta["duplicate_replies"]) == (1, 1, 1)
        assert meta["untimestamped_replies"] == 0
        assert meta["destinations"] == ["8.8.8.8"]

    def test_ping_without_header_and_compressed(self, tmp_path: Path) -> None:
        path = tmp_path / "ping.txt.gz"
        body = "".join(line + "\n" for line in PING_D.splitlines()[1:])
        path.write_bytes(gzip.compress(body.encode()))
        assert infer_format(path) == "ping"
        assert len(DataLoader().load_from_file(path)) == 3

    def test_ping_without_timestamps_says_to_use_dash_d(self, tmp_path: Path) -> None:
        path = tmp_path / "ping.txt"
        path.write_text(
            "PING 8.8.8.8 (8.8.8.8): 56 data bytes\n"
            "64 bytes from 8.8.8.8: icmp_seq=0 ttl=117 time=12.345 ms\n"
            "64 bytes from 8.8.8.8: icmp_seq=1 ttl=117 time=11.9 ms\n"
        )
        assert infer_format(path) == "ping"
        with pytest.raises(ValueError, match=r"2 reply line\(s\) but no timestamps.*ping -D"):
            DataLoader().load_from_file(path)

    def test_untimestamped_replies_among_timestamped_ones_are_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "ping.txt"
        path.write_text(PING_D + "64 bytes from 8.8.8.8: icmp_seq=9 ttl=117 time=1.0 ms\n")
        ds = DataLoader().load_from_file(path)
        assert len(ds) == 3 and ds.metadata["untimestamped_replies"] == 1

    def test_no_replies_is_an_error(self, tmp_path: Path) -> None:
        path = tmp_path / "ping.txt"
        path.write_text("PING 8.8.8.8 (8.8.8.8) 56(84) bytes of data.\n")
        with pytest.raises(ValueError, match="No ping reply lines"):
            DataLoader().load_from_file(path, file_format="ping")

    def test_fping_detected_by_content_not_as_csv(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = tmp_path / "fping.log"
        path.write_text(FPING_D)
        assert infer_format(path) == "fping"  # the first line has commas
        with caplog.at_level("WARNING"):
            ds = DataLoader().load_from_file(path)
        assert "2 destinations (1.1.1.1, 8.8.8.8); they are pooled" in caplog.text
        assert ds.metadata["destinations"] == ["1.1.1.1", "8.8.8.8"]
        assert [(m.destination, m.rtt_value) for m in ds.measurements] == [
            ("8.8.8.8", 12.3),
            ("1.1.1.1", 4.1),
            ("8.8.8.8", 11.8),
        ]
        meta = ds.metadata
        assert (meta["timeouts"], meta["duplicate_replies"], meta["duplicate_rows"]) == (1, 1, 1)

    def test_fping_without_timestamps_says_to_use_dash_d(self, tmp_path: Path) -> None:
        path = tmp_path / "fping.txt"
        path.write_text("8.8.8.8 : [0], 64 bytes, 12.3 ms (12.3 avg, 0% loss)\n")
        with pytest.raises(ValueError, match="fping -D"):
            DataLoader().load_from_file(path)

    @pytest.mark.parametrize(("fmt", "text"), [("ping", PING_D), ("fping", FPING_D)])
    def test_cli_validate(self, tmp_path: Path, fmt: str, text: str) -> None:
        path = tmp_path / f"{fmt}.txt"
        path.write_text(text)
        result = runner.invoke(app, ["validate", str(path)])
        assert result.exit_code == 0, result.output
        assert "Data validation passed" in result.output


# ------------------------------------------------------------------------ stream --format


def _window_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("streaming:\n  rerun_every_bins: 4\n")
    return cfg


def _series(n: int = 1200) -> list[tuple[float, float]]:
    """10 h at 30 s with a jump and wider jitter after 5 h."""
    samples = []
    for i in range(n):
        base, spread = (10.0, 0.2) if i < n // 2 else (30.0, 6.0)
        samples.append((T0 + 30.0 * i, round(base + spread * ((i * 7919) % 97) / 97, 3)))
    return samples


def _stream(tmp_path: Path, text: str, *extra: str) -> tuple[int, str, list[dict]]:
    args = ["stream", "--backend", "window", "--config", str(_window_config(tmp_path))]
    result = runner.invoke(app, [*args, *extra], input=text)
    events = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    return result.exit_code, result.output, events


@pytest.mark.parametrize("fmt", ["ping", "fping"])
def test_stream_format_matches_csv(tmp_path: Path, fmt: str) -> None:
    samples = _series()
    csv = "epoch,values\n" + "".join(f"{t},{r}\n" for t, r in samples)
    if fmt == "ping":
        lines = ["PING 8.8.8.8 (8.8.8.8) 56(84) bytes of data."] + [
            f"[{t:.6f}] 64 bytes from 8.8.8.8: icmp_seq={i} ttl=57 time={r} ms"
            for i, (t, r) in enumerate(samples)
        ]
    else:
        lines = [
            f"[{t:.5f}] 8.8.8.8 : [{i}], 64 bytes, {r} ms ({r} avg, 0% loss)"
            for i, (t, r) in enumerate(samples)
        ]
    code_csv, _, from_csv = _stream(tmp_path, csv)
    code, output, from_probe = _stream(tmp_path, "\n".join(lines) + "\n", "--format", fmt)
    assert code_csv == 0 and code == 0, output
    assert len(from_probe) >= 2 and from_probe == from_csv
    assert f"{len(samples)} samples read" in output


def test_stream_explicit_csv_is_the_default(tmp_path: Path) -> None:
    text = "epoch,values\n" + "".join(f"{t},{r}\n" for t, r in _series(600)) + "junk\n"
    default = runner.invoke(app, ["stream", "--backend", "window"], input=text)
    explicit = runner.invoke(app, ["stream", "--backend", "window", "--format", "csv"], input=text)
    assert default.exit_code == explicit.exit_code == 0, default.output
    assert default.stdout == explicit.stdout
    assert "600 samples read, 2 lines skipped" in explicit.output


def test_stream_warns_once_about_missing_timestamps(tmp_path: Path) -> None:
    text = "".join(f"64 bytes from 8.8.8.8: icmp_seq={i} ttl=57 time=1.{i} ms\n" for i in range(5))
    code, output, events = _stream(tmp_path, text, "--format", "ping")
    assert code == 0, output
    assert output.count("without a timestamp") == 1 and "ping -D" in output
    assert "0 samples read, 5 lines skipped" in output
    assert events == []


def test_stream_warns_once_about_several_destinations(tmp_path: Path) -> None:
    text = "".join(
        f"[{T0 + i}.0] {host} : [{i}], 64 bytes, 1.{i} ms (1.0 avg, 0% loss)\n"
        for i in range(6)
        for host in ("8.8.8.8", "1.1.1.1", "9.9.9.9")
    )
    code, output, _ = _stream(tmp_path, text, "--format", "fping")
    assert code == 0, output
    assert output.count("pooled into one RTT series") == 1
    assert "8.8.8.8 and 1.1.1.1" in output


def test_stream_rejects_an_unknown_format() -> None:
    result = runner.invoke(app, ["stream", "--format", "parquet"], input="")
    assert result.exit_code == 2
    assert "--format must be csv, ping, fping" in result.output
