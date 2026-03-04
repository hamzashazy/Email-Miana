import openpyxl
from pathlib import Path


COLUMN_MAP = {
    0: "id",
    1: "classification",
    2: "matched_keyword",
    3: "agent_name",
    4: "from_email",
    5: "property_address",
    6: "offer_type",
    7: "offer_price",
    8: "subject",
    9: "snippet",
    10: "received_at",
}


def load_emails(filepath: str | Path) -> list[dict]:
    """Load emails from the xlsx report. Returns list of dicts."""
    wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)
    ws = wb.active

    records = []
    for row_idx, row in enumerate(ws.iter_rows(min_row=3, values_only=True)):
        if row[0] is None:
            continue

        record = {}
        for col_idx, field_name in COLUMN_MAP.items():
            val = row[col_idx] if col_idx < len(row) else None
            if isinstance(val, str):
                val = val.replace("_x000D_", "").replace("\r", "").strip()
            record[field_name] = val

        record["offer_price"] = record.get("offer_price") or ""
        record["snippet"] = record.get("snippet") or ""
        record["subject"] = record.get("subject") or ""

        records.append(record)

    wb.close()
    return records
