"""
tests/test_agent.py
-------------------
Unit tests cho src/agent/agent.py — không cần Groq API thật.

Covers:
    - LLM trả lời thẳng (không gọi tool)
    - LLM gọi tool rồi tổng hợp kết quả
    - Tool không tồn tại → agent xử lý gracefully
    - Đạt max 5 vòng lặp → trả fallback message
    - Rate limit 429 → retry rồi trả thông báo lỗi
    - Fallback XML parsing (Groq quirk)

Run:
    pytest tests/test_agent.py -v
"""

import pytest
from unittest.mock import MagicMock, patch


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_response(content: str = "", tool_calls: list | None = None) -> MagicMock:
    """Tạo mock response giống AIMessage của LangChain."""
    r = MagicMock()
    r.content = content
    r.tool_calls = tool_calls or []
    return r


def _make_tool_call(name: str, args: dict, call_id: str = "tc_001") -> dict:
    return {"name": name, "args": args, "id": call_id}


def _make_fake_tool(name: str, return_value: str = "ok") -> MagicMock:
    t = MagicMock()
    t.name = name
    t.invoke.return_value = return_value
    return t


def _build_mock_llm(responses: list) -> MagicMock:
    """
    Trả về mock object simulate ChatGroq instance.
    .bind_tools() trả về chính nó (fluent interface).
    .invoke() trả lần lượt các response trong danh sách.
    """
    llm = MagicMock()
    llm.bind_tools.return_value = llm
    llm.invoke.side_effect = responses
    return llm


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def patch_groq():
    """Patch ChatGroq ở đúng module path."""
    with patch("src.agent.agent.ChatGroq") as MockGroq:
        yield MockGroq


# ── Test cases ────────────────────────────────────────────────────────────────


class TestDirectAnswer:
    """LLM trả lời ngay, không cần gọi tool."""

    def test_returns_llm_content(self, patch_groq):
        patch_groq.return_value = _build_mock_llm(
            [_make_response(content="Doanh thu Q1 là 5 tỷ đồng.")]
        )
        from src.agent.agent import create_analyst_agent

        agent = create_analyst_agent([])

        result = agent("tổng doanh thu Q1?")

        assert "5 tỷ" in result

    def test_empty_tool_list_still_works(self, patch_groq):
        patch_groq.return_value = _build_mock_llm(
            [_make_response(content="Không có dữ liệu phù hợp.")]
        )
        from src.agent.agent import create_analyst_agent

        agent = create_analyst_agent([])

        result = agent("câu hỏi không cần tool")

        assert result == "Không có dữ liệu phù hợp."


class TestToolCallFlow:
    """LLM gọi tool một lần rồi tổng hợp."""

    def test_tool_invoked_with_correct_args(self, patch_groq):
        fake_tool = _make_fake_tool("query_sql", return_value="total | 5000000")
        tc = _make_tool_call(
            "query_sql",
            {"sql": "SELECT SUM(CAST(unit_price AS BIGINT)*quantity) FROM sales"},
        )

        patch_groq.return_value = _build_mock_llm(
            [
                _make_response(tool_calls=[tc]),
                _make_response(content="Tổng doanh thu là 5 triệu."),
            ]
        )
        from src.agent.agent import create_analyst_agent

        agent = create_analyst_agent([fake_tool])

        result = agent("tổng doanh thu?")

        fake_tool.invoke.assert_called_once_with(
            {"sql": "SELECT SUM(CAST(unit_price AS BIGINT)*quantity) FROM sales"}
        )
        assert "5 triệu" in result

    def test_tool_result_passed_to_next_llm_call(self, patch_groq):
        fake_tool = _make_fake_tool(
            "list_tables", return_value="Table 'sales': month, city, product"
        )
        tc = _make_tool_call("list_tables", {}, call_id="tc_lt")

        patch_groq.return_value = _build_mock_llm(
            [
                _make_response(tool_calls=[tc]),
                _make_response(
                    content="Có bảng sales với các cột: month, city, product."
                ),
            ]
        )
        from src.agent.agent import create_analyst_agent

        agent = create_analyst_agent([fake_tool])

        result = agent("có những bảng nào?")

        assert "sales" in result
        assert fake_tool.invoke.call_count == 1

    def test_multiple_tool_calls_in_one_step(self, patch_groq):
        tool_a = _make_fake_tool("list_tables", return_value="Table 'sales'")
        tool_b = _make_fake_tool("query_sql", return_value="rows: 6")
        tc_a = _make_tool_call("list_tables", {}, call_id="tc_a")
        tc_b = _make_tool_call(
            "query_sql", {"sql": "SELECT COUNT(*) FROM sales"}, call_id="tc_b"
        )

        patch_groq.return_value = _build_mock_llm(
            [
                _make_response(tool_calls=[tc_a, tc_b]),
                _make_response(content="Có 6 bản ghi trong bảng sales."),
            ]
        )
        from src.agent.agent import create_analyst_agent

        agent = create_analyst_agent([tool_a, tool_b])

        result = agent("bao nhiêu dòng trong bảng sales?")

        tool_a.invoke.assert_called_once()
        tool_b.invoke.assert_called_once()
        assert "6" in result


class TestUnknownTool:
    """LLM gọi tool không tồn tại trong tool_map."""

    def test_unknown_tool_does_not_raise(self, patch_groq):
        tc = _make_tool_call("nonexistent_tool", {})
        patch_groq.return_value = _build_mock_llm(
            [
                _make_response(tool_calls=[tc]),
                _make_response(content="Xin lỗi, không thực hiện được."),
            ]
        )
        from src.agent.agent import create_analyst_agent

        agent = create_analyst_agent([])

        result = agent("làm điều không thể?")

        assert result is not None
        assert isinstance(result, str)

    def test_unknown_tool_fallback_message_sent_to_llm(self, patch_groq):
        tc = _make_tool_call("ghost_tool", {}, call_id="tc_ghost")
        patch_groq.return_value = _build_mock_llm(
            [
                _make_response(tool_calls=[tc]),
                _make_response(content="Tool không tồn tại."),
            ]
        )
        from src.agent.agent import create_analyst_agent

        agent = create_analyst_agent([])

        result = agent("gọi tool ma?")

        # LLM invoke lần 2 → nhận ToolMessage với nội dung lỗi
        second_call_args = patch_groq.return_value.invoke.call_args_list[1]
        messages_passed = second_call_args[0][0]
        tool_msgs = [m for m in messages_passed if hasattr(m, "tool_call_id")]
        assert any("ghost_tool" in str(m.content) for m in tool_msgs)


class TestMaxIterations:
    """Agent đạt giới hạn 5 vòng mà không thoát được."""

    def test_returns_fallback_string_after_5_steps(self, patch_groq):
        fake_tool = _make_fake_tool("query_sql", return_value="ok")
        tc = _make_tool_call("query_sql", {"sql": "SELECT 1"})

        # LLM luôn trả tool_call → không bao giờ kết thúc
        patch_groq.return_value = _build_mock_llm([_make_response(tool_calls=[tc])] * 5)
        from src.agent.agent import create_analyst_agent

        agent = create_analyst_agent([fake_tool])

        result = agent("vòng lặp vô tận")

        assert "5 vòng" in result

    def test_tool_called_at_most_5_times(self, patch_groq):
        fake_tool = _make_fake_tool("query_sql", return_value="ok")
        tc = _make_tool_call("query_sql", {"sql": "SELECT 1"})

        patch_groq.return_value = _build_mock_llm([_make_response(tool_calls=[tc])] * 5)
        from src.agent.agent import create_analyst_agent

        agent = create_analyst_agent([fake_tool])
        agent("loop forever")

        assert fake_tool.invoke.call_count <= 5


class TestRateLimitRetry:
    """Groq 429 → tenacity retry → sau cùng báo lỗi rõ ràng."""

    def test_rate_limit_returns_friendly_message(self, patch_groq):
        from tenacity import RetryError

        # _invoke_llm là nested function nên không patch được từ ngoài.
        # Thay vào đó: cho llm.invoke raise RetryError liên tục (3 lần)
        # để tenacity exhausted và reraise — agent bắt RetryError → friendly msg.
        mock_llm = _build_mock_llm([])
        mock_llm.invoke.side_effect = RetryError(MagicMock())
        patch_groq.return_value = mock_llm

        from src.agent.agent import create_analyst_agent

        agent = create_analyst_agent([])
        result = agent("bất kỳ câu hỏi nào")

        assert "rate limit" in result.lower() or "Groq" in result


class TestFallbackXmlParsing:
    """Groq trả về XML-style tool call trong exception message."""

    def test_fallback_tool_executed_from_exception(self, patch_groq):
        fake_tool = _make_fake_tool("query_sql", return_value="rows: 3")

        # Lần 1: invoke raise exception có XML tool call
        xml_err = Exception(
            'Failed: \'<function=query_sql {"sql": "SELECT COUNT(*) FROM sales"} </function>\''
        )
        # Lần 2: LLM tổng hợp (sau khi tool result được nhét vào messages)
        patch_groq.return_value = _build_mock_llm(
            [
                xml_err,
                _make_response(content="Có 3 hàng."),
            ]
        )

        from src.agent.agent import create_analyst_agent

        agent = create_analyst_agent([fake_tool])
        result = agent("đếm số hàng trong sales")

        fake_tool.invoke.assert_called_once()
        assert "3" in result or result  # không crash

    def test_fallback_with_unknown_tool_in_exception(self, patch_groq):
        xml_err = Exception("Failed: '<function=nonexistent {\"x\": 1} </function>'")
        patch_groq.return_value = _build_mock_llm([xml_err])

        from src.agent.agent import create_analyst_agent

        agent = create_analyst_agent([])
        result = agent("trigger unknown fallback")

        assert isinstance(result, str)
        assert result  # không trả về string rỗng


class TestToolException:
    """Tool.invoke() ném exception → agent xử lý gracefully, không crash."""

    def test_tool_exception_handled(self, patch_groq):
        broken_tool = MagicMock()
        broken_tool.name = "query_sql"
        broken_tool.invoke.side_effect = RuntimeError("DuckDB connection failed")

        tc = _make_tool_call("query_sql", {"sql": "SELECT 1"})
        patch_groq.return_value = _build_mock_llm(
            [
                _make_response(tool_calls=[tc]),
                _make_response(content="Có lỗi xảy ra khi truy vấn."),
            ]
        )
        from src.agent.agent import create_analyst_agent

        agent = create_analyst_agent([broken_tool])

        result = agent("chạy query bị lỗi")

        assert result is not None
        assert isinstance(result, str)
