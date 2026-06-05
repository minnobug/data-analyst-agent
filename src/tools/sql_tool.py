import duckdb
from langchain_core.tools import tool

DB_PATH = "data/sample/warehouse.db"

@tool
def list_tables() -> str:
    """List all available tables in the DuckDB database with their column names and types."""
    try:
        conn = duckdb.connect(DB_PATH)
        tables = conn.execute("SHOW TABLES").fetchdf()
        if tables.empty:
            return "No tables found in database."
        result = []
        for table in tables["name"].tolist():
            cols = conn.execute(f"DESCRIBE {table}").fetchdf()
            col_info = ", ".join([f"{r['column_name']} ({r['column_type']})" for _, r in cols.iterrows()])
            result.append(f"Table '{table}': {col_info}")
        conn.close()
        return "\n".join(result)
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def query_sql(sql: str) -> str:
    """Execute a SQL query on the DuckDB database and return results.

    Args:
        sql: Valid SQL query string. Use simple column names like unit_price and quantity.
    """
    try:
        conn = duckdb.connect(DB_PATH)
        # Tự động fix overflow INT32 khi nhân unit_price * quantity
        safe_sql = sql.replace(
            "unit_price * quantity",
            "CAST(unit_price AS BIGINT) * quantity"
        ).replace(
            "SUM(CAST(unit_price AS BIGINT) * quantity)",
            "SUM(CAST(unit_price AS BIGINT) * quantity)"
        )
        result = conn.execute(safe_sql).fetchdf()
        conn.close()
        if result.empty:
            return "Query returned no results."
        return result.to_string(index=False)
    except Exception as e:
        return f"SQL Error: {str(e)}"
