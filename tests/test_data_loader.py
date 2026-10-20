import pandas as pd
import pytest

from src.data_loader import REQUIRED_SHEETS, DataLoadError, load_data


def _write_workbook(path, sheets):
    with pd.ExcelWriter(path) as writer:
        for name in sheets:
            pd.DataFrame({"x": [1]}).to_excel(writer, sheet_name=name, index=False)


def test_loads_all_seven_required_sheets():
    data = load_data()
    assert tuple(data) == REQUIRED_SHEETS
    assert all(isinstance(df, pd.DataFrame) for df in data.values())


def test_ignores_data_dictionary_sheet():
    assert "Data Dictionary" not in load_data()


def test_v2_dataset_row_counts():
    data = load_data()
    assert len(data["Resources"]) == 28
    assert len(data["Projects"]) == 9
    assert len(data["Project_Requirements"]) == 50
    assert len(data["Allocations"]) == 380
    assert len(data["Calendar"]) == 12
    assert len(data["Dependencies"]) == 4


def test_missing_file_raises_clear_error(tmp_path):
    with pytest.raises(DataLoadError, match="not found"):
        load_data(tmp_path / "nope.xlsx")


def test_missing_sheet_is_named_in_error(tmp_path):
    path = tmp_path / "partial.xlsx"
    _write_workbook(path, [s for s in REQUIRED_SHEETS if s != "Dependencies"])
    with pytest.raises(DataLoadError, match="Dependencies"):
        load_data(path)


def test_extra_sheets_are_ignored(tmp_path):
    path = tmp_path / "extra.xlsx"
    _write_workbook(path, list(REQUIRED_SHEETS) + ["Notes"])
    assert tuple(load_data(path)) == REQUIRED_SHEETS


def test_unreadable_file_raises_clear_error(tmp_path):
    path = tmp_path / "broken.xlsx"
    path.write_text("not a workbook")
    with pytest.raises(DataLoadError, match="Could not open"):
        load_data(path)
