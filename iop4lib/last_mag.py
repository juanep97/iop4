#!/usr/bin/env python
""" last_mag.py
    Retrieve the latest magnitude for blazars and update the Google Sheet table.
"""

# Prevent iop4lib directory from shadowing standard library modules (like typing)
# when executed directly as a script (e.g. `python iop4lib/last_mag.py`).
if not __package__:
    import sys
    from pathlib import Path
    _script_dir = str(Path(__file__).resolve().parent)
    _repo_dir = str(Path(__file__).resolve().parents[1])
    if sys.path and sys.path[0] == _script_dir:
        sys.path.pop(0)
    if _repo_dir not in sys.path:
        sys.path.insert(0, _repo_dir)

import os
import sys
import json
import math
import argparse
from pathlib import Path

# iop4lib config
import iop4lib.config
iop4conf = iop4lib.Config(config_db=True)

from django.conf import settings
if not settings.configured:
    iop4conf.configure_db()

# iop4lib imports
from iop4lib.db import AstroSource
from iop4lib.enums import SRCTYPES


# ---------------------------------------------------------------------------
# Google Sheets update logic
# ---------------------------------------------------------------------------

def update_google_sheet(
    data,
    spreadsheet_id: str,
    sheet_name=None,
    credentials_path=None,
    has_header=True,
):
    """Upload *data* to a Google Sheet, replacing any existing content.

    Parameters
    ----------
    data:
        The tabular data to upload. Accepted types:

        - ``list[list]``        – written as-is (include a header row yourself
                                  if you want one).
        - Any object with ``columns`` and ``values`` attributes (e.g. DataFrame)
                                  – column headers are written as the first row.

    spreadsheet_id:
        The Spreadsheet ID found in the Google Sheets URL
        (``https://docs.google.com/spreadsheets/d/<SPREADSHEET_ID>/...``).

    sheet_name:
        Name of the worksheet tab to write to.  If ``None`` the first sheet
        is used.

    credentials_path:
        Path to the Service Account JSON key file.  When ``None``, the value
        of the ``GOOGLE_SA_KEY_PATH`` environment variable is used.

    has_header:
        Whether the first row of ``data`` is a header row (default: True).
        Used when reporting the number of data rows written.

    Raises
    ------
    EnvironmentError
        If no credentials path is provided and ``GOOGLE_SA_KEY_PATH`` is not set.
    FileNotFoundError
        If the credentials file cannot be found at the resolved path.
    ImportError
        If ``gspread`` or ``google-auth`` are not installed.
    RuntimeError
        If any Google Sheets API call fails.
    """
    # ------------------------------------------------------------------
    # 1. Resolve credentials path
    # ------------------------------------------------------------------
    if credentials_path is None:
        credentials_path = os.environ.get("GOOGLE_SA_KEY_PATH")
        if not credentials_path:
            raise EnvironmentError(
                "No credentials path provided and the environment variable "
                "'GOOGLE_SA_KEY_PATH' is not set. Set it to the path of your "
                "Service Account JSON key file, or pass the path directly via "
                "the 'credentials_path' argument."
            )

    if not os.path.isfile(credentials_path):
        raise FileNotFoundError(
            f"Service Account credentials file not found: {credentials_path!r}"
        )

    # ------------------------------------------------------------------
    # 2. Convert data to a list of lists (rows)
    # ------------------------------------------------------------------
    is_dataframe = False
    if hasattr(data, "columns") and hasattr(data, "values"):
        # Duck-type support (e.g., pandas DataFrame) without importing pandas
        rows = [list(data.columns)] + [list(r) for r in data.values]
        is_dataframe = True
    elif isinstance(data, (list, tuple)):
        rows = list(data)
    else:
        raise TypeError(
            f"'data' must be a list of lists, "
            f"got {type(data).__name__!r}."
        )

    # Ensure every cell is JSON-serialisable (handle None, NaN, numpy scalars, etc.)
    serialisable_rows = []
    for row in rows:
        serialisable_row = []
        for cell in row:
            if cell is None:
                serialisable_row.append("")
                continue

            # Convert numpy scalar to Python native (e.g., np.float64 -> float)
            if hasattr(cell, "item") and callable(cell.item):
                try:
                    cell = cell.item()
                except (TypeError, ValueError):
                    pass

            # Check for NaN float
            try:
                if math.isnan(cell):
                    serialisable_row.append("")
                    continue
            except (TypeError, ValueError):
                pass

            serialisable_row.append(cell)
        serialisable_rows.append(serialisable_row)

    # ------------------------------------------------------------------
    # 3. Authenticate and open the spreadsheet
    # ------------------------------------------------------------------
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except ImportError as exc:
        raise ImportError(
            "Required packages are missing. Install them with:\n"
            "  pip install gspread google-auth"
        ) from exc

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]

    try:
        creds = Credentials.from_service_account_file(credentials_path, scopes=scopes)
        client = gspread.authorize(creds)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to authenticate with Google APIs using credentials at "
            f"{credentials_path!r}: {exc}"
        ) from exc

    try:
        spreadsheet = client.open_by_key(spreadsheet_id)
    except gspread.exceptions.SpreadsheetNotFound:
        raise RuntimeError(
            f"Spreadsheet with ID {spreadsheet_id!r} not found. Make sure the "
            "Service Account has been granted Editor access to it."
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not open spreadsheet {spreadsheet_id!r}: {exc}"
        ) from exc

    # ------------------------------------------------------------------
    # 4. Select worksheet
    # ------------------------------------------------------------------
    try:
        if sheet_name:
            worksheet = spreadsheet.worksheet(sheet_name)
        else:
            worksheet = spreadsheet.get_worksheet(0)
    except gspread.exceptions.WorksheetNotFound:
        raise RuntimeError(
            f"Worksheet {sheet_name!r} not found in spreadsheet "
            f"{spreadsheet_id!r}."
        )

    # ------------------------------------------------------------------
    # 5. Clear existing content and write fresh data
    # ------------------------------------------------------------------
    try:
        worksheet.clear()
        worksheet.update(range_name="A1", values=serialisable_rows)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to write data to the Google Sheet: {exc}"
        ) from exc

    n_data_rows = max(0, len(serialisable_rows) - 1) if (is_dataframe or has_header) else len(serialisable_rows)
    print(
        f"✓ Google Sheet updated: {n_data_rows} data row(s) written to "
        f"worksheet {worksheet.title!r} in spreadsheet {spreadsheet_id!r}."
    )


# ---------------------------------------------------------------------------
# Data retrieval logic
# ---------------------------------------------------------------------------

def get_last_mag_data(srctype=None):
    """Retrieve tabular data of the latest R-band magnitude for sources.

    Parameters
    ----------
    srctype : str, optional
        Source type to filter by (default is 'blazar' / SRCTYPES.BLAZAR).
        If None, SRCTYPES.BLAZAR is used.

    Returns
    -------
    list[list]
        A list of rows including the header:
        [["Nombre", "Otros Nombres", "Magnitud", "Fecha de la Última Magnitud"], ...]
    """
    if srctype is None:
        srctype = SRCTYPES.BLAZAR

    sources = AstroSource.objects.filter(srctype=srctype).all()  # Retrieving all the astrosources

    fields = ["Nombre", "Otros Nombres", "Magnitud", "Fecha de la Última Magnitud"]
    last_mag = [fields]
    for a in sources:
        try:
            mag, err, night = a.last_night_mag_R
            last_mag.append([a.name, a.other_names, mag, str(night)])
        except Exception:
            continue

    return last_mag


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------

def main(args=None):
    parser = argparse.ArgumentParser(description="Update Google Sheet with the latest magnitude for blazars.")
    parser.add_argument("-c", "--config", dest="config_path", help="Path to spsheet_config.json", default=None)
    parser.add_argument("-s", "--spreadsheet-id", dest="spreadsheet_id", help="Google Spreadsheet ID", default=None)
    parser.add_argument("-n", "--sheet-name", dest="sheet_name", help="Worksheet tab name", default=None)
    parser.add_argument("-k", "--credentials", dest="credentials_path", help="Path to Service Account JSON key", default=None)
    parsed_args = parser.parse_args(args)

    lm = get_last_mag_data(srctype=SRCTYPES.BLAZAR)

    # ---------------------------------------------------------------------------
    # Push data to Google Sheets
    # Configuration is loaded from (in priority order):
    #   1. Command-line arguments
    #   2. Environment variables: SPREADSHEET_ID, SHEET_NAME, GOOGLE_SA_KEY_PATH
    #   3. spsheet_config.json from candidate locations
    # ---------------------------------------------------------------------------

    config = {}
    config_candidates = [
        parsed_args.config_path,
        os.environ.get("IOP4_LAST_MAG_CONFIG"),
        Path("./last_mag_config/spsheet_config.json"),
        Path(getattr(iop4conf, "datadir", "~/workplace/iop4data/")).expanduser() / "last_mag_config" / "spsheet_config.json",
        Path(__file__).resolve().parent / "last_mag_config" / "spsheet_config.json",
    ]

    config_path = None
    for cand in config_candidates:
        if cand and Path(cand).is_file():
            config_path = Path(cand).resolve()
            break

    if config_path:
        with open(config_path, "r") as f:
            config = json.load(f)
    elif os.path.exists("./last_mag_config/spsheet_config.json"):
        config_path = "./last_mag_config/spsheet_config.json"
        with open(config_path, "r") as f:
            config = json.load(f)

    spreadsheet_id = parsed_args.spreadsheet_id or os.environ.get("SPREADSHEET_ID") or config.get("spreadsheet_id")
    sheet_name = parsed_args.sheet_name or os.environ.get("SHEET_NAME") or config.get("sheet_name")
    credentials_path = parsed_args.credentials_path or os.environ.get("GOOGLE_SA_KEY_PATH") or config.get("sa_credentials_path")

    print(f"Config file {config_path}")
    print(f"    - Spreadsheet ID: {spreadsheet_id}")
    print(f"    - Sheet name: {sheet_name}")
    print(f"    - Credentials path: {credentials_path}")

    # If credentials_path is relative and not found in cwd, resolve relative to the config file's directory
    if credentials_path and not os.path.isabs(credentials_path) and not os.path.isfile(credentials_path) and config_path:
        candidate_cred = (config_path.parent / credentials_path).resolve()
        if candidate_cred.is_file():
            credentials_path = str(candidate_cred)
        else:
            candidate_cred_parent = (config_path.parent.parent / credentials_path).resolve()
            if candidate_cred_parent.is_file():
                credentials_path = str(candidate_cred_parent)

    if spreadsheet_id:
        update_google_sheet(
            data=lm,
            spreadsheet_id=spreadsheet_id,
            sheet_name=sheet_name,
            credentials_path=credentials_path,
            has_header=True,
        )
    else:
        print(
            "Skipping Google Sheets upload: set the SPREADSHEET_ID environment "
            "variable to enable automatic upload."
        )


if __name__ == '__main__':
    main()
