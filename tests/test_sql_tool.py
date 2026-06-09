import os
import sys
import unittest
from unittest.mock import MagicMock, patch, call
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import conftest stubs trước
import tests.conftest  # noqa: F401

from src.tools.sql_tool import _sanitize, _resolve_db_path, DB_PATH


class TestResolveDbPath(unittest.TestCase):
    """_resolve_db_path phải xử lý đúng mọi trường hợp env var."""

    def test_env_var_set_returns_env_path(self):
        with patch.dict(os.environ, {"WAREHOUSE_DB": "/tmp/mydb.db"}):
            from importlib import reload
            import src.tools.sql_tool as m

            path = m._resolve_db_path()
        self.assertEqual(str(path), "/tmp/mydb.db")

    def test_env_var_empty_string_uses_fallback(self):
        with patch.dict(os.environ, {"WAREHOUSE_DB": ""}):
            from src.tools.sql_tool import _resolve_db_path

            path = _resolve_db_path()
        self.assertIn("data", str(path))
        self.assertIn("warehouse.db", str(path))

    def test_env_var_not_set_uses_fallback(self):
        env = {k: v for k, v in os.environ.items() if k != "WAREHOUSE_DB"}
        with patch.dict(os.environ, env, clear=True):
            from src.tools.sql_tool import _resolve_db_path

            path = _resolve_db_path()
        self.assertIn("warehouse.db", str(path))

    def test_fallback_path_ends_with_warehouse_db(self):
        env = {k: v for k, v in os.environ.items() if k != "WAREHOUSE_DB"}
        with patch.dict(os.environ, env, clear=True):
            from src.tools.sql_tool import _resolve_db_path

            path = _resolve_db_path()
        self.assertTrue(str(path).endswith("warehouse.db"))

    def test_db_path_is_path_object(self):
        from pathlib import Path
        from src.tools.sql_tool import DB_PATH

        self.assertIsInstance(DB_PATH, Path)


class TestSanitize(unittest.TestCase):
    """_sanitize phải accept SELECT/WITH/EXPLAIN và reject mọi thứ khác."""

    # ── Allowed queries ──────────────────────────────────────────────────

    def test_select_allowed(self):
        ok, sql, err = _sanitize("SELECT * FROM sales")
        self.assertTrue(ok)
        self.assertEqual(err, "")

    def test_select_case_insensitive(self):
        ok, sql, err = _sanitize("select * from sales")
        self.assertTrue(ok)

    def test_with_cte_allowed(self):
        ok, sql, err = _sanitize("WITH cte AS (SELECT 1) SELECT * FROM cte")
        self.assertTrue(ok)

    def test_explain_allowed(self):
        ok, sql, err = _sanitize("EXPLAIN SELECT * FROM sales")
        self.assertTrue(ok)

    def test_leading_whitespace_allowed(self):
        ok, sql, err = _sanitize("  \n  SELECT 1")
        self.assertTrue(ok)

    # ── Blocked statements ───────────────────────────────────────────────

    def test_drop_blocked(self):
        ok, _, err = _sanitize("DROP TABLE sales")
        self.assertFalse(ok)
        self.assertIn("DROP", err)

    def test_delete_blocked(self):
        ok, _, err = _sanitize("DELETE FROM sales WHERE 1=1")
        self.assertFalse(ok)

    def test_insert_blocked(self):
        ok, _, err = _sanitize("INSERT INTO sales VALUES (1,2,3,4,5)")
        self.assertFalse(ok)

    def test_update_blocked(self):
        ok, _, err = _sanitize("UPDATE sales SET quantity = 0")
        self.assertFalse(ok)

    def test_alter_blocked(self):
        ok, _, err = _sanitize("ALTER TABLE sales ADD COLUMN x INT")
        self.assertFalse(ok)

    def test_truncate_blocked(self):
        ok, _, err = _sanitize("TRUNCATE TABLE sales")
        self.assertFalse(ok)

    def test_attach_blocked(self):
        ok, _, err = _sanitize("ATTACH DATABASE 'hack.db'")
        self.assertFalse(ok)

    def test_copy_blocked(self):
        ok, _, err = _sanitize("COPY sales TO '/tmp/out.csv'")
        self.assertFalse(ok)

    def test_install_blocked(self):
        ok, _, err = _sanitize("INSTALL httpfs")
        self.assertFalse(ok)

    def test_load_blocked(self):
        ok, _, err = _sanitize("LOAD 'some_extension'")
        self.assertFalse(ok)

    def test_empty_query_blocked(self):
        ok, _, err = _sanitize("")
        self.assertFalse(ok)
        self.assertIn("empty", err.lower())

    def test_whitespace_only_blocked(self):
        ok, _, err = _sanitize("   \n\t  ")
        self.assertFalse(ok)

    # ── Keyword in SELECT context — blocked (best-effort) ───────────────

    def test_drop_after_select_blocked(self):
        """DROP trong câu SELECT vẫn bị block — best-effort safety."""
        ok, _, err = _sanitize("SELECT 1; DROP TABLE sales")
        self.assertFalse(ok)

    # ── Overflow rewrite ─────────────────────────────────────────────────

    def test_overflow_rewrite_applied(self):
        ok, safe, _ = _sanitize("SELECT unit_price * quantity FROM sales")
        self.assertTrue(ok)
        self.assertIn("CAST(unit_price AS BIGINT)", safe)
        self.assertNotIn("unit_price * quantity", safe)

    def test_overflow_rewrite_case_insensitive(self):
        ok, safe, _ = _sanitize("SELECT UNIT_PRICE * QUANTITY FROM sales")
        self.assertTrue(ok)
        self.assertIn("CAST(unit_price AS BIGINT)", safe)

    def test_overflow_rewrite_preserves_rest(self):
        sql = "SELECT city, unit_price * quantity as revenue FROM sales"
        ok, safe, _ = _sanitize(sql)
        self.assertTrue(ok)
        self.assertIn("city", safe)
        self.assertIn("CAST(unit_price AS BIGINT) * quantity", safe)

    def test_no_overflow_pattern_unchanged(self):
        sql = "SELECT city, SUM(quantity) FROM sales GROUP BY city"
        ok, safe, _ = _sanitize(sql)
        self.assertTrue(ok)
        self.assertNotIn("CAST", safe)

    def test_overflow_multiple_occurrences(self):
        sql = "SELECT unit_price * quantity, unit_price * quantity FROM sales"
        ok, safe, _ = _sanitize(sql)
        self.assertTrue(ok)
        self.assertEqual(safe.count("CAST(unit_price AS BIGINT)"), 2)

    # ── Return tuple structure ───────────────────────────────────────────

    def test_returns_three_tuple(self):
        result = _sanitize("SELECT 1")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    def test_ok_true_has_empty_error(self):
        ok, sql, err = _sanitize("SELECT 1")
        self.assertTrue(ok)
        self.assertEqual(err, "")
        self.assertNotEqual(sql, "")

    def test_ok_false_has_empty_sql(self):
        ok, sql, err = _sanitize("DROP TABLE x")
        self.assertFalse(ok)
        self.assertEqual(sql, "")
        self.assertNotEqual(err, "")


class TestListTables(unittest.TestCase):
    """list_tables phải trả schema đúng và luôn close connection."""

    def _make_mock_conn(self, tables=None, cols=None):
        conn = MagicMock()
        conn.__enter__ = lambda s: s
        conn.__exit__ = MagicMock(return_value=False)

        # SHOW TABLES result
        if tables is None:
            tables = ["sales"]
        tables_df = pd.DataFrame({"name": tables})

        # Column info result
        if cols is None:
            cols = pd.DataFrame(
                {
                    "column_name": [
                        "month",
                        "city",
                        "product",
                        "unit_price",
                        "quantity",
                    ],
                    "data_type": [
                        "VARCHAR",
                        "VARCHAR",
                        "VARCHAR",
                        "INTEGER",
                        "INTEGER",
                    ],
                }
            )

        def execute_side_effect(sql, params=None):
            mock_result = MagicMock()
            if "SHOW TABLES" in sql.upper():
                mock_result.fetchdf.return_value = tables_df
            elif "information_schema" in sql.lower():
                mock_result.fetchdf.return_value = cols
            else:
                mock_result.fetchdf.return_value = pd.DataFrame()
            return mock_result

        conn.execute.side_effect = execute_side_effect
        return conn

    def test_returns_table_and_columns(self):
        mock_conn = self._make_mock_conn()
        with patch("src.tools.sql_tool.get_connection", return_value=mock_conn):
            from src.tools.sql_tool import list_tables

            result = list_tables.invoke({})
        self.assertIn("sales", result)
        self.assertIn("month", result)

    def test_empty_database_returns_no_tables_message(self):
        conn = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchdf.return_value = pd.DataFrame({"name": []})
        conn.execute.return_value = mock_result

        with patch("src.tools.sql_tool.get_connection", return_value=conn):
            from src.tools.sql_tool import list_tables

            result = list_tables.invoke({})
        self.assertIn("No tables", result)

    def test_connection_closed_on_success(self):
        mock_conn = self._make_mock_conn()
        with patch("src.tools.sql_tool.get_connection", return_value=mock_conn):
            from src.tools.sql_tool import list_tables

            list_tables.invoke({})
        mock_conn.close.assert_called_once()

    def test_connection_closed_on_exception(self):
        conn = MagicMock()
        conn.execute.side_effect = RuntimeError("db error")

        with patch("src.tools.sql_tool.get_connection", return_value=conn):
            from src.tools.sql_tool import list_tables

            result = list_tables.invoke({})

        conn.close.assert_called_once()
        self.assertIn("Error", result)

    def test_multiple_tables_all_listed(self):
        cols = pd.DataFrame(
            {
                "column_name": ["id", "name"],
                "data_type": ["INTEGER", "VARCHAR"],
            }
        )
        mock_conn = self._make_mock_conn(
            tables=["sales", "products", "users"], cols=cols
        )
        with patch("src.tools.sql_tool.get_connection", return_value=mock_conn):
            from src.tools.sql_tool import list_tables

            result = list_tables.invoke({})
        for table in ["sales", "products", "users"]:
            self.assertIn(table, result)


class TestQuerySql(unittest.TestCase):
    """query_sql phải sanitize, execute, và luôn close connection."""

    def _make_conn_with_result(self, df: pd.DataFrame):
        conn = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchdf.return_value = df
        conn.execute.return_value = mock_result
        return conn

    def test_valid_query_returns_data(self):
        df = pd.DataFrame({"city": ["Hanoi", "HCMC"], "total": [100, 200]})
        conn = self._make_conn_with_result(df)
        with patch("src.tools.sql_tool.get_connection", return_value=conn):
            from src.tools.sql_tool import query_sql

            result = query_sql.invoke(
                {"sql": "SELECT city, SUM(quantity) as total FROM sales GROUP BY city"}
            )
        self.assertIn("Hanoi", result)
        self.assertIn("HCMC", result)

    def test_empty_result_returns_message(self):
        conn = self._make_conn_with_result(pd.DataFrame())
        with patch("src.tools.sql_tool.get_connection", return_value=conn):
            from src.tools.sql_tool import query_sql

            result = query_sql.invoke({"sql": "SELECT * FROM sales WHERE 1=0"})
        self.assertIn("no results", result.lower())

    def test_sanitize_rejects_dangerous_sql(self):
        with patch("src.tools.sql_tool.get_connection") as mock_gc:
            from src.tools.sql_tool import query_sql

            result = query_sql.invoke({"sql": "DROP TABLE sales"})
        mock_gc.assert_not_called()
        self.assertIn("SQL Error", result)

    def test_sanitize_rejects_empty_sql(self):
        with patch("src.tools.sql_tool.get_connection") as mock_gc:
            from src.tools.sql_tool import query_sql

            result = query_sql.invoke({"sql": ""})
        mock_gc.assert_not_called()
        self.assertIn("SQL Error", result)

    def test_connection_closed_on_success(self):
        df = pd.DataFrame({"x": [1]})
        conn = self._make_conn_with_result(df)
        with patch("src.tools.sql_tool.get_connection", return_value=conn):
            from src.tools.sql_tool import query_sql

            query_sql.invoke({"sql": "SELECT 1 as x"})
        conn.close.assert_called_once()

    def test_connection_closed_on_db_exception(self):
        conn = MagicMock()
        conn.execute.side_effect = RuntimeError("connection failed")

        with patch("src.tools.sql_tool.get_connection", return_value=conn):
            from src.tools.sql_tool import query_sql

            result = query_sql.invoke({"sql": "SELECT * FROM sales"})

        conn.close.assert_called_once()
        self.assertIn("SQL Error", result)

    def test_overflow_rewrite_before_execute(self):
        """unit_price * quantity phải được rewrite trước khi execute."""
        df = pd.DataFrame({"revenue": [1000000]})
        conn = self._make_conn_with_result(df)
        with patch("src.tools.sql_tool.get_connection", return_value=conn):
            from src.tools.sql_tool import query_sql

            query_sql.invoke({"sql": "SELECT unit_price * quantity FROM sales"})

        # Kiểm tra SQL thực tế được truyền vào execute
        executed_sql = conn.execute.call_args[0][0]
        self.assertIn("CAST(unit_price AS BIGINT)", executed_sql)
        self.assertNotIn(
            "unit_price * quantity",
            executed_sql.replace("CAST(unit_price AS BIGINT) * quantity", ""),
        )

    def test_result_as_string(self):
        df = pd.DataFrame({"col": ["val1", "val2"]})
        conn = self._make_conn_with_result(df)
        with patch("src.tools.sql_tool.get_connection", return_value=conn):
            from src.tools.sql_tool import query_sql

            result = query_sql.invoke({"sql": "SELECT col FROM t"})
        self.assertIsInstance(result, str)

    def test_get_connection_not_available_returns_error(self):
        with patch("src.tools.sql_tool._DUCKDB_AVAILABLE", False):
            with patch(
                "src.tools.sql_tool.get_connection",
                side_effect=RuntimeError("duckdb not installed"),
            ):
                from src.tools.sql_tool import query_sql

                result = query_sql.invoke({"sql": "SELECT 1"})
        self.assertIn("SQL Error", result)


class TestGetConnection(unittest.TestCase):
    """get_connection phải forward đúng params vào duckdb.connect."""

    def test_calls_duckdb_connect_with_db_path(self):
        mock_conn = MagicMock()
        with patch("src.tools.sql_tool.duckdb") as mock_duckdb:
            mock_duckdb.connect.return_value = mock_conn
            with patch("src.tools.sql_tool._DUCKDB_AVAILABLE", True):
                from src.tools.sql_tool import get_connection, DB_PATH

                get_connection()
        mock_duckdb.connect.assert_called_once_with(str(DB_PATH), read_only=True)

    def test_raises_when_duckdb_not_available(self):
        with patch("src.tools.sql_tool._DUCKDB_AVAILABLE", False):
            from src.tools.sql_tool import get_connection

            with self.assertRaises(RuntimeError) as ctx:
                get_connection()
        self.assertIn("duckdb", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
