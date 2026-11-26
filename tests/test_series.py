"""Multi-series input: ``input.series`` filters, ``input.group_by`` keys, the mixed-input
warning, per-series analysis and grouped exports (``jitterbug.io.series``)."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from jitterbug import JitterbugAnalyzer, JitterbugConfig
from jitterbug.io import DataLoader
from jitterbug.io.series import format_series_key, split_series
from jitterbug.models import InputConfig, MultiSeriesResult, SeriesResult

# --------------------------------------------------------------------------- fixtures


def _rows(destination: str, epochs: list[float], rtts: list[float], **extra: str) -> list[dict]:
    return [
        {"epoch": e, "rtt": r, "source": "10.0.0.1", "destination": destination, **extra}
        for e, r in zip(epochs, rtts, strict=True)
    ]


@pytest.fixture
def mixed_frame() -> pd.DataFrame:
    """Two destinations interleaved; ``a`` has a duplicate and an out-of-order row,
    ``b`` has one non-positive RTT."""
    rows = [
        *_rows("a", [10, 20], [5.0, 6.0]),
        *_rows("b", [10, 20, 30], [7.0, -1.0, 8.0]),
        *_rows("a", [15, 20], [4.0, 9.0]),
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def mixed_csv(tmp_path: Path, mixed_frame: pd.DataFrame) -> Path:
    path = tmp_path / "mixed.csv"
    mixed_frame.to_csv(path, index=False)
    return path


@pytest.fixture
def mixed_jsonl(tmp_path: Path, mixed_frame: pd.DataFrame) -> Path:
    path = tmp_path / "mixed.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in mixed_frame.to_dict("records")))
    return path


@pytest.fixture
def mixed_scamper(tmp_path: Path) -> Path:
    """scamper pings to two destinations from one source."""

    def ping(dst: str, *samples: tuple[int, float]) -> dict:
        return {
            "type": "ping",
            "src": "192.168.1.1",
            "dst": dst,
            "responses": [{"rtt": rtt, "tx": {"sec": sec, "usec": 0}} for sec, rtt in samples],
        }

    records = [
        ping("8.8.8.8", (100, 10.0), (101, 11.0)),
        ping("1.1.1.1", (100, 20.0)),
        ping("8.8.8.8", (102, 12.0)),
        ping("1.1.1.1", (103, 21.0), (104, 22.0)),
    ]
    path = tmp_path / "pings.json"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


EXAMPLE_CSV = (
    Path(__file__).resolve().parents[1] / "examples" / "network_analysis" / "data" / "raw.csv"
)


# --------------------------------------------------------------------------- configuration


def test_group_by_and_series_default_to_empty_and_stay_out_of_dumps():
    config = InputConfig()
    assert config.group_by == [] and config.series == {}
    assert "group_by" not in config.model_dump() and "series" not in config.model_dump()
    assert "group_by" not in JitterbugConfig().model_dump()["input"]


def test_group_by_and_series_round_trip_through_a_dump():
    config = JitterbugConfig(
        input=InputConfig(group_by=["destination"], series={"source": "10.0.0.1"})
    )
    data = config.model_dump()
    assert data["input"]["group_by"] == ["destination"]
    assert JitterbugConfig.model_validate(data).input == config.input


@pytest.mark.parametrize("group_by", [["destination", "destination"], [""], ["  "]])
def test_invalid_group_by_is_rejected(group_by: list[str]):
    with pytest.raises(ValidationError):
        InputConfig(group_by=group_by)


@pytest.mark.parametrize(
    ("value", "text"), [(1234, "1234"), (1234.0, "1234"), (0.5, "0.5"), (True, "True")]
)
def test_numeric_series_values_become_text(value: object, text: str):
    assert InputConfig(series={"probe": value}).series == {"probe": text}


def test_numeric_series_value_in_a_config_file_selects_rows(tmp_path: Path):
    config_file = tmp_path / "c.yaml"
    config_file.write_text("input:\n  series:\n    probe: 2\n")
    config = JitterbugConfig.from_file(config_file)
    df = pd.DataFrame({"epoch": [1, 2, 3], "rtt": [1.0, 2.0, 3.0], "probe": [1, 2, 2]})
    assert len(DataLoader(config.input).load_from_dataframe(df)) == 2


def test_empty_series_filter_name_is_rejected():
    with pytest.raises(ValidationError):
        InputConfig(series={"": "x"})


# --------------------------------------------------------------------------- keys


def test_source_and_destination_resolve_to_the_mapped_columns():
    df = pd.DataFrame({"t": [1, 2, 3], "ms": [1.0, 2.0, 3.0], "dst": ["x", "y", "x"]})
    config = InputConfig(destination_column="dst", group_by=["destination"])
    frames = split_series(df, config)
    assert list(frames) == [(("destination", "x"),), (("destination", "y"),)]
    assert frames[(("destination", "x"),)].index.tolist() == [0, 2]


def test_integral_float_keys_are_written_without_a_decimal_point():
    df = pd.DataFrame({"probe": [1.0, np.nan, 2.0]})
    frames = split_series(df, InputConfig(group_by=["probe"]))
    assert list(frames) == [(("probe", "1"),), (("probe", None),), (("probe", "2"),)]
    assert format_series_key((("probe", None),)) == "probe=<missing>"


def test_missing_key_column_is_an_error_naming_the_columns():
    df = pd.DataFrame({"epoch": [1], "rtt": [1.0]})
    with pytest.raises(ValueError, match=r"no column for series key 'destination'.*columns"):
        split_series(df, InputConfig(group_by=["destination"]))


# --------------------------------------------------------------------------- filters


@pytest.mark.parametrize("fixture", ["mixed_csv", "mixed_jsonl"])
def test_series_filter_keeps_one_destination(fixture: str, request: pytest.FixtureRequest):
    path = request.getfixturevalue(fixture)
    ds = DataLoader(InputConfig(series={"destination": "b"})).load_from_file(path)
    assert [m.rtt_value for m in ds.measurements] == [7.0, 8.0]
    assert {m.destination for m in ds.measurements} == {"b"}
    assert ds.metadata["dropped_rows"]["non_positive"] == 1
    assert ds.metadata["source_destination_pairs"] == 1


def test_filter_that_matches_nothing_names_the_available_values(mixed_csv: Path):
    loader = DataLoader(InputConfig(series={"destination": "c"}))
    with pytest.raises(ValueError, match=r"destination=c; available destination values: a, b"):
        loader.load_from_file(mixed_csv)


def test_filters_combine_and_list_values_among_the_rows_still_kept(mixed_csv: Path):
    loader = DataLoader(InputConfig(series={"destination": "a", "epoch": "30"}))
    with pytest.raises(ValueError, match=r"available epoch values: 10, 15, 20$"):
        loader.load_from_file(mixed_csv)


def test_filter_on_the_dataframe_entry_point(mixed_frame: pd.DataFrame):
    ds = DataLoader(InputConfig(series={"destination": "a"})).load_from_dataframe(mixed_frame)
    assert len(ds) == 4


# --------------------------------------------------------------------------- grouping


@pytest.mark.parametrize("fixture", ["mixed_csv", "mixed_jsonl"])
def test_load_series_applies_policies_per_series(fixture: str, request: pytest.FixtureRequest):
    path = request.getfixturevalue(fixture)
    config = InputConfig(group_by=["destination"], duplicates="first")
    datasets = DataLoader(config).load_series(path)
    assert list(datasets) == [(("destination", "a"),), (("destination", "b"),)]
    a, b = datasets.values()
    # a: rows 10, 20, 15, 20 -> sorted, the second 20 (rtt 9) is a duplicate of 20 (rtt 6)
    assert [(m.epoch, m.rtt_value) for m in a.measurements] == [(10, 5.0), (15, 4.0), (20, 6.0)]
    assert a.metadata["sorted_on_load"] and a.metadata["duplicates_removed"] == 1
    assert a.metadata["series"] == {"destination": "a"}
    # b: in order, no duplicates once grouped (in the mixed series 10 and 20 repeat)
    assert [m.epoch for m in b.measurements] == [10, 30]
    assert not b.metadata["sorted_on_load"] and b.metadata["duplicate_rows"] == 0


def test_mixed_series_counts_duplicates_across_destinations(mixed_csv: Path):
    ds = DataLoader().load_from_file(mixed_csv)
    # valid times 10, 20, 10, 30, 15, 20 (b's 20 is dropped): 10 and 20 repeat once each
    assert ds.metadata["duplicate_rows"] == 2


def test_duplicates_error_fails_the_whole_file(mixed_csv: Path):
    loader = DataLoader(InputConfig(group_by=["destination"], duplicates="error"))
    with pytest.raises(ValueError, match="duplicates: error"):
        loader.load_series(mixed_csv)


def test_scamper_input_groups_and_filters_by_destination(mixed_scamper: Path):
    datasets = DataLoader(InputConfig(group_by=["source", "destination"])).load_series(
        mixed_scamper
    )
    assert {format_series_key(k): len(v) for k, v in datasets.items()} == {
        "source=192.168.1.1,destination=8.8.8.8": 3,
        "source=192.168.1.1,destination=1.1.1.1": 3,
    }
    one = DataLoader(InputConfig(series={"destination": "1.1.1.1"})).load_from_file(mixed_scamper)
    assert [m.rtt_value for m in one.measurements] == [20.0, 21.0, 22.0]
    assert one.metadata["format"] == "scamper_warts"


def test_series_whose_rows_are_all_dropped_is_skipped(tmp_path: Path):
    path = tmp_path / "rtts.csv"
    pd.DataFrame(_rows("a", [1, 2], [1.0, 2.0]) + _rows("b", [1, 2], [-1.0, 0.0])).to_csv(
        path, index=False
    )
    skipped: dict = {}
    datasets = DataLoader(InputConfig(group_by=["destination"])).load_series(path, None, skipped)
    assert list(datasets) == [(("destination", "a"),)]
    assert list(skipped) == [(("destination", "b"),)]
    assert "no valid RTT rows" in skipped[(("destination", "b"),)]


def test_auto_time_unit_is_inferred_once_for_all_series():
    # Series b alone has small times that auto would read as seconds; all rows are ms.
    df = pd.DataFrame(
        {
            "epoch": [1.7e12, 1.7e12 + 1000, 5000.0, 6000.0, 1.7e12 + 2000],
            "rtt": [1.0, 2.0, 3.0, 4.0, 5.0],
            "destination": ["a", "a", "b", "b", "a"],
        }
    )
    config = InputConfig(group_by=["destination"], time_unit="auto")
    datasets = DataLoader(config).load_series_from_dataframe(df)
    for ds in datasets.values():
        assert ds.metadata["time_unit"] == "ms" and ds.metadata["time_unit_inferred"]


def test_load_from_file_refuses_several_series_and_lists_them(mixed_csv: Path):
    loader = DataLoader(InputConfig(group_by=["destination"]))
    with pytest.raises(ValueError, match=r"2 series under group_by.*destination=a; destination=b"):
        loader.load_from_file(mixed_csv)
    ds = DataLoader(
        InputConfig(group_by=["destination"], series={"destination": "a"})
    ).load_from_file(mixed_csv)
    assert ds.metadata["series"] == {"destination": "a"}


# --------------------------------------------------------------------------- mixed input


def test_mixed_input_logs_one_warning_and_records_the_count(
    mixed_csv: Path, caplog: pytest.LogCaptureFixture
):
    with caplog.at_level(logging.WARNING, logger="jitterbug"):
        ds = DataLoader().load_from_file(mixed_csv)
    warnings = [r.message for r in caplog.records if "distinct (source, destination)" in r.message]
    assert len(warnings) == 1
    assert "2 distinct" in warnings[0]
    assert "--group-by" in warnings[0] and "--series" in warnings[0]
    assert ds.metadata["source_destination_pairs"] == 2
    assert len(ds) == 6  # still analyzed as one series, as before


@pytest.mark.parametrize(
    "config",
    [InputConfig(), InputConfig(series={"source": "10.0.0.1"})],
    ids=["single-pair", "filtered"],
)
def test_no_mixed_warning_for_one_pair_or_with_a_filter(
    config: InputConfig, caplog: pytest.LogCaptureFixture
):
    df = pd.DataFrame(_rows("a", [1, 2], [1.0, 2.0]))
    with caplog.at_level(logging.WARNING, logger="jitterbug"):
        ds = DataLoader(config).load_from_dataframe(df)
    assert not [r for r in caplog.records if "distinct (source" in r.message]
    assert ds.metadata["source_destination_pairs"] == 1


def test_no_source_or_destination_columns_count_as_one_pair():
    ds = DataLoader().load_from_dataframe(pd.DataFrame({"epoch": [1, 2], "rtt": [1.0, 2.0]}))
    assert ds.metadata["source_destination_pairs"] == 1


# --------------------------------------------------------------------------- analysis


@pytest.fixture(scope="module")
def two_destinations() -> pd.DataFrame:
    """The two halves of the paper dataset as two destinations, plus a third series too
    short to analyze (a single 15-minute interval)."""
    if not EXAMPLE_CSV.exists():
        pytest.skip("example dataset not present")
    raw = pd.read_csv(EXAMPLE_CSV)
    short = raw.iloc[:5].assign(destination="9.9.9.9")
    return pd.concat(
        [
            raw.iloc[:23000].assign(destination="8.8.8.8"),
            raw.iloc[23000:].assign(destination="1.1.1.1"),
            short,
        ],
        ignore_index=True,
    )


def test_each_series_is_analyzed_as_it_would_be_alone(two_destinations: pd.DataFrame):
    config = JitterbugConfig(input=InputConfig(group_by=["destination"]))
    grouped = JitterbugAnalyzer(config).analyze_series_from_dataframe(two_destinations)
    assert [e.key for e in grouped.series] == [
        {"destination": "8.8.8.8"},
        {"destination": "1.1.1.1"},
        {"destination": "9.9.9.9"},
    ]
    for entry in grouped.analyzed():
        alone = two_destinations[two_destinations["destination"] == entry.key["destination"]]
        expected = JitterbugAnalyzer(JitterbugConfig()).analyze_from_dataframe(alone)
        assert entry.result is not None and entry.result.inferences
        assert entry.result.inferences == expected.inferences
    short = grouped.series[2]
    assert short.result is None and "need at least 2" in (short.skipped_reason or "")
    assert grouped.metadata == {
        "series_count": 3,
        "analyzed": 2,
        "skipped": 1,
        "series_filter": {},
    }


def test_analyze_series_from_file_skips_series_without_valid_rows(tmp_path: Path):
    path = tmp_path / "rtts.csv"
    pd.DataFrame(_rows("a", [1, 2], [1.0, 2.0]) + _rows("b", [1], [-1.0])).to_csv(path, index=False)
    config = JitterbugConfig(input=InputConfig(group_by=["destination"]))
    grouped = JitterbugAnalyzer(config).analyze_series_from_file(path)
    reasons = {e.key["destination"]: e.skipped_reason for e in grouped.series}
    assert "need at least 2" in reasons["a"]
    assert "no valid RTT rows" in reasons["b"]


def test_series_result_needs_exactly_one_of_result_and_reason():
    with pytest.raises(ValidationError):
        SeriesResult(key={"destination": "a"})


# --------------------------------------------------------------------------- exports


@pytest.fixture(scope="module")
def grouped(two_destinations: pd.DataFrame) -> tuple[JitterbugAnalyzer, MultiSeriesResult]:
    analyzer = JitterbugAnalyzer(JitterbugConfig(input=InputConfig(group_by=["destination"])))
    return analyzer, analyzer.analyze_series_from_dataframe(two_destinations)


def test_grouped_json_is_one_document_with_a_series_list(
    grouped: tuple[JitterbugAnalyzer, MultiSeriesResult], tmp_path: Path
):
    analyzer, results = grouped
    out = tmp_path / "r.json"
    analyzer.save_series_results(results, out, "json")
    doc = json.loads(out.read_text())
    assert doc["group_by"] == ["destination"]
    assert [sorted(e) for e in doc["series"]] == [
        ["key", "result"],
        ["key", "result"],
        ["key", "skipped_reason"],
    ]
    first = doc["series"][0]["result"]
    assert set(first) == {"inferences", "metadata"}
    assert first["metadata"]["config"]["input"]["group_by"] == ["destination"]
    assert doc["metadata"]["skipped"] == 1


@pytest.mark.parametrize("fmt", ["csv", "parquet"])
def test_grouped_tables_lead_with_the_key_columns(
    fmt: str, grouped: tuple[JitterbugAnalyzer, MultiSeriesResult], tmp_path: Path
):
    if fmt == "parquet":
        pytest.importorskip("pyarrow")
    analyzer, results = grouped
    out = tmp_path / f"r.{fmt}"
    analyzer.save_series_results(results, out, fmt)
    table = pd.read_csv(out) if fmt == "csv" else pd.read_parquet(out)
    assert list(table.columns) == ["destination", "starts", "ends", "congestion"]
    for entry in results.analyzed():
        assert entry.result is not None
        rows = table[table["destination"] == entry.key["destination"]]
        expected = entry.result.to_dataframe()
        np.testing.assert_allclose(rows["starts"], expected["starts"])
        assert rows["congestion"].tolist() == expected["congestion"].tolist()
    assert "9.9.9.9" not in set(table["destination"])  # skipped: no rows


def test_grouped_csv_is_compressed_like_the_single_series_one(
    grouped: tuple[JitterbugAnalyzer, MultiSeriesResult], tmp_path: Path
):
    analyzer, results = grouped
    out = tmp_path / "r.csv.gz"
    analyzer.save_series_results(results, out, "csv")
    assert list(pd.read_csv(out).columns)[0] == "destination"


def test_grouped_table_with_no_rows_still_has_its_columns():
    results = MultiSeriesResult(
        group_by=["probe"], series=[SeriesResult(key={"probe": "1"}, skipped_reason="x")]
    )
    assert list(results.to_dataframe().columns) == ["probe", "starts", "ends", "congestion"]


def test_key_named_like_a_result_column_is_rejected():
    results = MultiSeriesResult(group_by=["starts"], series=[])
    with pytest.raises(ValueError, match="clash"):
        results.to_dataframe()


def test_unknown_grouped_output_format_is_rejected(
    grouped: tuple[JitterbugAnalyzer, MultiSeriesResult], tmp_path: Path
):
    analyzer, results = grouped
    with pytest.raises(ValueError, match="Unsupported output format"):
        analyzer.save_series_results(results, tmp_path / "r.xml", "xml")
