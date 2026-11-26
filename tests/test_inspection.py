"""``jitterbug.io.inspection.inspect_file``: the dry run behind ``jitterbug inspect``."""

import gzip
import json
from pathlib import Path

import pytest

from jitterbug.io.inspection import inspect_file
from jitterbug.models import InputConfig


@pytest.fixture
def mixed_csv(tmp_path: Path) -> Path:
    """Destination ``a``: times 10, 20, 15, 20 (one out of order, one duplicate);
    destination ``b``: 10, 20, 30 with a negative RTT at 20 and a missing one at 30."""
    path = tmp_path / "mixed.csv"
    path.write_text(
        "epoch,rtt,destination\n10,5,a\n20,6,a\n10,7,b\n20,-1,b\n30,,b\n15,4,a\n20,9,a\n"
    )
    return path


def test_reports_format_mapping_and_row_counts(mixed_csv: Path):
    report = inspect_file(mixed_csv)
    assert report["format"] == "csv" and report["compression"] is None
    assert report["reader"] == {"delimiter": ","}
    assert report["columns"] == ["epoch", "rtt", "destination"]
    assert report["mapping"]["rtt_column"] == "rtt"
    assert report["mapping"]["rtt_column_configured"] is None
    assert report["mapping"]["source_column"] is None
    assert report["mapping"]["destination_column"] == "destination"
    assert report["rows"] == {
        "read": 7,
        "selected": 7,
        "missing": 1,
        "non_positive": 1,
        "too_large": 0,
        "valid": 5,
    }
    assert report["errors"] == []
    json.dumps(report)  # JSON-serializable


def test_without_group_by_lists_pairs_but_policies_cover_the_mixed_series(mixed_csv: Path):
    report = inspect_file(mixed_csv)
    assert report["grouped"] is False and report["series_by"] == ["destination"]
    keys = [e["key"] for e in report["series"]]
    assert keys == [{"destination": "a"}, {"destination": "b"}]
    a, b = report["series"]
    assert (a["rows"], a["samples"], a["duplicate_timestamps"], a["out_of_order_rows"]) == (
        4,
        4,
        1,
        1,
    )
    assert a["median_interval_seconds"] == 5.0 and a["max_gap_seconds"] == 5.0
    assert (b["rows"], b["samples"]) == (3, 1)
    assert b["median_interval_seconds"] is None
    # As one mixed series: valid times 10, 20, 10, 15, 20 -> 10 and 20 repeat
    assert report["policies"]["duplicates"]["rows"] == 2
    assert report["policies"]["unsorted"] == {
        "setting": "sort",
        "series_affected": 1,
        "rows": 1,  # 10 after 20
        "effect": "sorted",
    }


def test_group_by_counts_policies_per_series(mixed_csv: Path):
    report = inspect_file(mixed_csv, input_config=InputConfig(group_by=["destination"]))
    assert report["grouped"] is True
    assert report["policies"]["duplicates"]["rows"] == 1
    assert report["policies"]["unsorted"]["rows"] == 1


def test_duplicates_first_reports_the_rows_it_would_drop(mixed_csv: Path):
    config = InputConfig(group_by=["destination"], duplicates="first")
    report = inspect_file(mixed_csv, input_config=config)
    assert report["policies"]["duplicates"]["dropped"] == 1
    assert report["policies"]["duplicates"]["effect"] == "dropped"
    assert report["series"][0]["samples"] == 3


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (InputConfig(duplicates="error"), "duplicates: error"),
        (InputConfig(unsorted="error"), "unsorted: error"),
        (InputConfig(out_of_range="error"), "out_of_range: error"),
    ],
)
def test_policies_set_to_error_are_reported_not_raised(
    mixed_csv: Path, config: InputConfig, message: str
):
    report = inspect_file(mixed_csv, input_config=config)
    assert any(message in e for e in report["errors"])


def test_series_filter_reduces_the_rows(mixed_csv: Path):
    report = inspect_file(mixed_csv, input_config=InputConfig(series={"destination": "b"}))
    assert report["rows"]["read"] == 7 and report["rows"]["selected"] == 3
    assert [e["key"] for e in report["series"]] == [{"destination": "b"}]


def test_filter_matching_nothing_is_reported(mixed_csv: Path):
    report = inspect_file(mixed_csv, input_config=InputConfig(series={"destination": "z"}))
    assert "available destination values: a, b" in report["errors"][0]
    assert "mapping" not in report


def test_missing_time_column_stops_after_the_columns(mixed_csv: Path):
    report = inspect_file(mixed_csv, input_config=InputConfig(time_column="ts"))
    assert report["columns"] == ["epoch", "rtt", "destination"]
    assert "no 'ts' column" in report["errors"][0]
    assert "rows" not in report


def test_auto_time_unit_reports_what_it_inferred(tmp_path: Path):
    path = tmp_path / "ms.csv"
    path.write_text("epoch,rtt\n1700000000000,1\n1700000060000,2\n")
    report = inspect_file(path, input_config=InputConfig(time_unit="auto"))
    assert report["mapping"]["time_unit"] == "auto"
    assert report["mapping"]["time_unit_applied"] == "ms"
    assert report["series"][0]["start"].startswith("2023-11-14T22:13:20")
    assert report["series"][0]["key"] == {} and report["series_by"] == []


def test_compressed_scamper_file(tmp_path: Path):
    record = {
        "type": "ping",
        "src": "192.168.1.1",
        "dst": "8.8.8.8",
        "responses": [{"rtt": 1.5, "tx": {"sec": 100, "usec": 0}}],
    }
    path = tmp_path / "pings.json.gz"
    with gzip.open(path, "wt") as f:
        f.write(json.dumps(record) + "\n")
    report = inspect_file(path)
    assert report["format"] == "json" and report["format_detail"] == "scamper_warts"
    assert report["compression"] == "gzip"
    assert report["mapping"]["fixed_by_format"]["time_column"] == "epoch"
    assert report["series"][0]["key"] == {"source": "192.168.1.1", "destination": "8.8.8.8"}


def test_no_valid_rows_is_an_error(tmp_path: Path):
    path = tmp_path / "bad.csv"
    path.write_text("epoch,rtt\n1,-1\n2,0\n")
    report = inspect_file(path)
    assert "No valid RTT rows" in report["errors"][0]


def test_missing_group_by_column_is_reported(mixed_csv: Path):
    report = inspect_file(mixed_csv, input_config=InputConfig(group_by=["probe"]))
    assert "no column for series key 'probe'" in report["errors"][0]
    assert "rows" not in report
