import pytest
from src.tools.sql_tool import query_sql, list_tables


def test_list_tables():
    result = list_tables.invoke({})
    assert "sales" in result


def test_query_sql_basic():
    result = query_sql.invoke({"sql": "SELECT COUNT(*) FROM sales"})
    assert "6" in result  # 6 rows in sample data


def test_query_sql_revenue():
    """Test INT32 overflow fix — unit_price * quantity với số lớn"""
    result = query_sql.invoke(
        {"sql": "SELECT SUM(CAST(unit_price AS BIGINT) * quantity) AS total FROM sales"}
    )
    assert "SQL Error" not in result
    assert "1.11" in result  # 11,100,000,000 hiển thị dạng scientific notation


def test_query_sql_invalid():
    result = query_sql.invoke({"sql": "SELECT * FROM nonexistent_table"})
    assert "SQL Error" in result
