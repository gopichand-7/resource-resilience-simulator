"""Step 4: load the seven data sheets from the Excel workbook.

No business calculations live here. The loader only finds the workbook,
checks that the required sheets exist, and returns one DataFrame per sheet.
Extra sheets (for example "Data Dictionary") are documentation and are ignored.
"""
from pathlib import Path

import pandas as pd

REQUIRED_SHEETS = (
    "Resources",
    "Resource_Skills",
    "Projects",
    "Project_Requirements",
    "Allocations",
    "Calendar",
    "Dependencies",
)

DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "resource_resilience_dataset_v2.xlsx"
)


class DataLoadError(Exception):
    """Raised when the workbook is missing or does not have the expected sheets."""


def load_data(path=None):
    """Return {sheet_name: DataFrame} for the seven required sheets."""
    path = Path(path) if path is not None else DEFAULT_DATA_PATH
    if not path.is_file():
        raise DataLoadError(f"Dataset not found: {path.name}. It should be in the data folder next to the app.")

    try:
        workbook = pd.ExcelFile(path)
    except Exception as exc:  # unreadable or not a real .xlsx
        raise DataLoadError(f"Could not open workbook {path.name}: {exc}") from exc

    with workbook:
        missing = [s for s in REQUIRED_SHEETS if s not in workbook.sheet_names]
        if missing:
            raise DataLoadError(f"Workbook is missing required sheet(s): {', '.join(missing)}")
        return {sheet: workbook.parse(sheet) for sheet in REQUIRED_SHEETS}
