from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

# ---------------------------------------------------------------------------
# Lazy import duckdb để test có thể mock dễ dàng
# ---------------------------------------------------------------------------
try:
    import duckdb

    _DUCKDB_AVAILABLE = True
except ImportError:
    _DUCKDB_AVAILABLE = False
    duckdb = None  # type: ignore[assignment]

try:
    from langchain_core.tools import tool as _lc_tool

    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False

    # Fallback decorator khi không có langchain (ví dụ: môi trường test)
    def _lc_tool(fn):  # type: ignore[misc]
        fn.name = fn.__name__
        fn.invoke = lambda args: fn(**args) if isinstance(args, dict) else fn(args)
        return fn


# ---------------------------------------------------------------------------
# DB path resolution
# ---------------------------------------------------------------------------


def _resolve_db_path() -> Path:
    """
    Ưu tiên env var WAREHOUSE_DB.
    Fallback: <project_root>/data/sample/warehouse.db
    Không dùng Path(None) — luôn trả Path hợp lệ.
    """
    env_val = os.getenv("WAREHOUSE_DB", "").strip()
    if env_val:
        return Path(env_val)
    # __file__ = src/tools/sql_tool.py → parent.parent.parent = project root
    return (
        Path(__file__).resolve().parent.parent.parent
        / "data"
        / "sample"
        / "warehouse.db"
    )


DB_PATH: Path = _resolve_db_path()


# ---------------------------------------------------------------------------
# SQL safety guards
# ---------------------------------------------------------------------------

# Chỉ cho phép câu lệnh bắt đầu bằng SELECT / WITH / EXPLAIN
_ALLOWED_START = re.compile(r"^\s*(SELECT|WITH|EXPLAIN)\b", re.IGNORECASE)

# Chặn các keyword nguy hiểm. Dùng word boundary để tránh false positive
# trên tên cột như "updated_at". Tuy nhiên vẫn có thể bị bypass qua string
# literal — đây là best-effort safety, không phải security boundary.
_DANGEROUS_KEYWORDS = re.compile(
    r"\b(DROP|DELETE|INSERT|UPDATE|ALTER|TRUNCATE|REPLACE"
    r"|ATTACH|DETACH|COPY|EXPORT|IMPORT|PRAGMA|CALL|EXECUTE"
    r"|LOAD|INSTALL)\b",
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
        ok=True  → safe_sql sẵn sàng để execute
        ok=False → error_message mô tả lý do từ chối
    """
    stripped = sql.strip()

    if not stripped:
        return False, "", "SQL Error: query is empty."

    if not _ALLOWED_START.match(stripped):
        first_word = stripped.split()[0].upper() if stripped.split() else ""
        return (
            False,
            "",
            f"SQL Error: only SELECT / WITH / EXPLAIN queries are allowed. "
            f"Got: '{first_word}'",
        )

    match = _DANGEROUS_KEYWORDS.search(stripped)
    if match:
        kw = match.group(0).upper()
        return False, "", f"SQL Error: keyword '{kw}' is not permitted."

    safe = _OVERFLOW_PATTERN.sub(_OVERFLOW_SAFE, stripped)
    return True, safe, ""


# ---------------------------------------------------------------------------
# Connection factory (extensible cho S3 phase 2)
# ---------------------------------------------------------------------------


def get_connection():
    """
    Trả về DuckDB connection.
    Phase 1: local .db file.
    Phase 2: sẽ thêm S3 Parquet mode khi kết nối SmartCity repo.
    """
    if not _DUCKDB_AVAILABLE:
        raise RuntimeError("duckdb is not installed. Run: pip install duckdb")
    return duckdb.connect(str(DB_PATH), read_only=True)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@_lc_tool
def list_tables() -> str:
    """
    List all available tables in the DuckDB database
    along with their column names and types.
    """
    conn = None
    try:
        conn = get_connection()

        # SHOW TABLES là cú pháp DuckDB chuẩn
        tables_df = conn.execute("SHOW TABLES").fetchdf()
        if tables_df.empty:
            return "No tables found in database."

        result: list[str] = []
        for table in tables_df["name"].tolist():
            # PRAGMA table_info() là cú pháp DuckDB/SQLite chuẩn hơn DESCRIBE
            cols_df = conn.execute(
                f"SELECT column_name, data_type FROM information_schema.columns "
                f"WHERE table_name = ? ORDER BY ordinal_position",
                [table],
            ).fetchdf()

            if cols_df.empty:
                result.append(f"Table '{table}': (no columns found)")
                continue

            col_info = ", ".join(
                f"{row['column_name']} ({row['data_type']})"
                for _, row in cols_df.iterrows()
            )
            result.append(f"Table '{table}': {col_info}")

        return "\n".join(result)

    except Exception as exc:
        return f"Error listing tables: {exc}"
    finally:
        if conn is not None:
            conn.close()


@_lc_tool
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

    conn = None
    try:
        conn = get_connection()
        result_df = conn.execute(safe_sql).fetchdf()

        if result_df.empty:
            return "Query returned no results."

        return result_df.to_string(index=False)

    except Exception as exc:
        return f"SQL Error: {exc}"
    finally:
        if conn is not None:
            conn.close()
