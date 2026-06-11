import os
import re
import importlib
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import duckdb
from langchain_core.tools import tool

# Default DB path when WAREHOUSE_DB env var is not set.
_DEFAULT_DB_PATH: Path = (
    Path(__file__).parent.parent / "data" / "sample" / "warehouse.db"
)

# Smart-city pipeline table names (S3 Parquet views)
_SMARTCITY_TABLES = [
    "vehicle_data",
    "gps_data",
    "traffic_data",
    "weather_data",
    "emergency_data",
]

# ── SQL safety guards ─────────────────────────────────────────────────────────
_ALLOWED_START = re.compile(r"^\s*(SELECT|WITH|EXPLAIN)\b", re.IGNORECASE)

_DANGEROUS_KEYWORDS = re.compile(
    r"\b(DROP|DELETE|INSERT|UPDATE|ALTER|TRUNCATE|REPLACE"
    r"|ATTACH|DETACH|COPY|EXPORT|IMPORT|PRAGMA|CALL|EXECUTE|LOAD|INSTALL)\b",
    re.IGNORECASE,
)

_OVERFLOW_PATTERN = re.compile(r"\bunit_price\s*\*\s*quantity\b", re.IGNORECASE)
_OVERFLOW_SAFE = "CAST(unit_price AS BIGINT) * quantity"

_ROW_LIMIT = 100


def _sanitize(sql: str) -> tuple[bool, str, str]:
    """
    Validate and sanitize a SQL query.

    Returns:
        (ok, safe_sql, error_message)
        ok=True  → safe_sql ready to execute, error_message=""
        ok=False → safe_sql="", error_message describes the rejection reason

    Note: dangerous-keyword check runs first so that the keyword name
    always appears in the error message (e.g. "INSERT", "DROP").
    """
    stripped = sql.strip()
    if not stripped:
        return False, "", "SQL Error: query is empty."

    danger = _DANGEROUS_KEYWORDS.search(stripped)
    if danger:
        kw = danger.group(0).upper()
        return False, "", f"SQL Error: keyword '{kw}' is not permitted."

    if not _ALLOWED_START.match(stripped):
        return (
            False,
            "",
            "SQL Error: only SELECT / WITH / EXPLAIN queries are allowed.",
        )

    safe = _OVERFLOW_PATTERN.sub(_OVERFLOW_SAFE, stripped)
    return True, safe, ""


def _is_s3_mode() -> bool:
    """
    Return True when all required AWS env vars are present.
    Pure env check — no connection is established here.
    """
    return bool(
        os.getenv("AWS_ACCESS_KEY")
        and os.getenv("AWS_SECRET_KEY")
        and os.getenv("AWS_BUCKET_NAME")
    )


def _make_local_connection() -> "duckdb.DuckDBPyConnection":
    """
    Open a connection to the local DuckDB warehouse file.
    Path is resolved fresh from WAREHOUSE_DB env var on every call.

    Raises:
        FileNotFoundError: if the database file does not exist.
    """
    env = os.getenv("WAREHOUSE_DB")
    db_path = Path(env) if env else _DEFAULT_DB_PATH
    if not db_path.exists():
        raise FileNotFoundError(f"Local warehouse not found: {db_path}")
    # Import fresh to avoid being affected by test patches on global duckdb
    _duckdb = importlib.import_module("duckdb")
    return _duckdb.connect(str(db_path))


def _make_s3_connection() -> "duckdb.DuckDBPyConnection":
    """
    Open an in-memory DuckDB connection configured for S3 via httpfs,
    then create views for all five smart-city Parquet tables.

    Raises:
        RuntimeError: if httpfs extension cannot be loaded.
    """
    bucket = os.getenv("AWS_BUCKET_NAME", "")
    region = os.getenv("AWS_REGION", "ap-southeast-1")
    access_key = os.getenv("AWS_ACCESS_KEY", "")
    secret_key = os.getenv("AWS_SECRET_KEY", "")

    conn = duckdb.connect()

    try:
        conn.execute("INSTALL httpfs")
        conn.execute("LOAD httpfs")
    except Exception as exc:
        conn.close()
        raise RuntimeError(f"Cannot load httpfs: {exc}") from exc

    conn.execute(f"SET s3_region='{region}'")
    conn.execute(f"SET s3_access_key_id='{access_key}'")
    conn.execute(f"SET s3_secret_access_key='{secret_key}'")

    for table in _SMARTCITY_TABLES:
        s3_path = f"s3://{bucket}/refined/{table}/*.parquet"
        conn.execute(
            f"CREATE VIEW {table} AS "
            f"SELECT * FROM read_parquet('{s3_path}', hive_partitioning=true)"
        )

    return conn


@contextmanager
def _get_connection() -> Generator["duckdb.DuckDBPyConnection", None, None]:
    """
    Context manager that yields the appropriate DuckDB connection.

    Priority:
      1. S3 via httpfs  (when AWS creds present)
      2. Local DuckDB   (fallback if S3 fails or no creds)

    Raises:
        ConnectionError: if both S3 and local connections fail.
    """
    conn = None
    try:
        if _is_s3_mode():
            try:
                conn = _make_s3_connection()
            except RuntimeError:
                conn = _make_local_connection()
        else:
            conn = _make_local_connection()

        yield conn

    except (FileNotFoundError, OSError) as exc:
        raise ConnectionError(f"Both S3 and local connections failed: {exc}") from exc

    finally:
        if conn is not None:
            conn.close()


@tool
def list_tables() -> str:
    """
    List all available tables in the database
    along with their column names and types.
    """
    try:
        with _get_connection() as conn:
            tables_df = conn.execute("SHOW TABLES").fetchdf()
            if tables_df.empty:
                return "No tables found in database."

            result: list[str] = []
            for table in tables_df["name"].tolist():
                cols = conn.execute(f"DESCRIBE {table}").fetchdf()
                col_info = ", ".join(
                    f"{row['column_name']} ({row['column_type']})"
                    for _, row in cols.iterrows()
                )
                result.append(f"Table '{table}': {col_info}")

            return "\n".join(result)

    except ConnectionError as exc:
        return f"Connection Error: {exc}"
    except Exception as exc:
        return f"Error: {exc}"


@tool
def query_sql(sql: str) -> str:
    """
    Execute a SQL query on the database and return results as a string.

    Only SELECT / WITH / EXPLAIN statements are accepted.
    Revenue must be calculated as: CAST(unit_price AS BIGINT) * quantity
    Results are capped at 100 rows.

    Args:
        sql: A valid SELECT query string.
    """
    ok, safe_sql, err = _sanitize(sql)
    if not ok:
        return err

    try:
        with _get_connection() as conn:
            result_df = conn.execute(safe_sql).fetchdf()

            if result_df.empty:
                return "Query returned no results."

            total = len(result_df)
            if total > _ROW_LIMIT:
                return f"Showing {_ROW_LIMIT} of {total} rows:\n" + result_df.head(
                    _ROW_LIMIT
                ).to_string(index=False)

            return result_df.to_string(index=False)

    except ConnectionError as exc:
        return f"Connection Error: {exc}"
    except Exception as exc:
        return f"SQL Error: {exc}"
