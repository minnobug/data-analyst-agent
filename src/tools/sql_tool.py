import os
import re
from pathlib import Path

import duckdb
from langchain_core.tools import tool

# ── DB path ───────────────────────────────────────────────────────────────────
# Ưu tiên env var WAREHOUSE_DB (cho staging/production).
# Fallback: absolute path tính từ vị trí file này.
DB_PATH: Path = Path(
    os.getenv("WAREHOUSE_DB")
    or Path(__file__).parent.parent / "data" / "sample" / "warehouse.db"
)

# ── SQL safety guards ─────────────────────────────────────────────────────────
# Chỉ cho phép câu lệnh bắt đầu bằng SELECT / WITH / EXPLAIN.
_ALLOWED_START = re.compile(r"^\s*(SELECT|WITH|EXPLAIN)\b", re.IGNORECASE)

# Chặn các keyword nguy hiểm dù nằm ở đâu trong query.
_DANGEROUS_KEYWORDS = re.compile(
    r"\b(DROP|DELETE|INSERT|UPDATE|ALTER|TRUNCATE|REPLACE"
    r"|ATTACH|DETACH|COPY|EXPORT|IMPORT|PRAGMA|CALL|EXECUTE)\b",
    re.IGNORECASE,
)

# Fix INT32 overflow: unit_price * quantity → CAST(unit_price AS BIGINT) * quantity
_OVERFLOW_PATTERN = re.compile(r"\bunit_price\s*\*\s*quantity\b", re.IGNORECASE)
_OVERFLOW_SAFE = "CAST(unit_price AS BIGINT) * quantity"


def _sanitize(sql: str) -> tuple[bool, str, str]:
    """
    Validate và sanitize SQL query.

    Returns:
        (ok, safe_sql, error_message)
        ok=True  → safe_sql sẵn sàng để execute, error_message=""
        ok=False → safe_sql="", error_message mô tả lý do từ chối
    """
    stripped = sql.strip()
    if not stripped:
        return False, "", "SQL Error: query is empty."

    if not _ALLOWED_START.match(stripped):
        return (
            False,
            "",
            ("SQL Error: only SELECT / WITH / EXPLAIN queries are allowed."),
        )

    if _DANGEROUS_KEYWORDS.search(stripped):
        kw = _DANGEROUS_KEYWORDS.search(stripped).group(0).upper()
        return False, "", f"SQL Error: keyword '{kw}' is not permitted."

    safe = _OVERFLOW_PATTERN.sub(_OVERFLOW_SAFE, stripped)
    return True, safe, ""


@tool
def list_tables() -> str:
    """
    List all available tables in the DuckDB database
    along with their column names and types.
    """
    try:
        conn = duckdb.connect(str(DB_PATH), read_only=True)
        tables_df = conn.execute("SHOW TABLES").fetchdf()
        if tables_df.empty:
            conn.close()
            return "No tables found in database."

        result: list[str] = []
        for table in tables_df["name"].tolist():
            cols = conn.execute(f"DESCRIBE {table}").fetchdf()
            col_info = ", ".join(
                f"{row['column_name']} ({row['column_type']})"
                for _, row in cols.iterrows()
            )
            result.append(f"Table '{table}': {col_info}")

        conn.close()
        return "\n".join(result)

    except Exception as exc:
        return f"Error: {exc}"


@tool
def query_sql(sql: str) -> str:
    """
    Execute a SQL query on the DuckDB database and return results as a string.

    Only SELECT / WITH / EXPLAIN statements are accepted.
    Revenue must be calculated as: CAST(unit_price AS BIGINT) * quantity

    Args:
        sql: A valid SELECT query string.
    """
    ok, safe_sql, err = _sanitize(sql)
    if not ok:
        return err

    try:
        conn = duckdb.connect(str(DB_PATH), read_only=True)
        result_df = conn.execute(safe_sql).fetchdf()
        conn.close()

        if result_df.empty:
            return "Query returned no results."

        return result_df.to_string(index=False)

    except Exception as exc:
        return f"SQL Error: {exc}"
