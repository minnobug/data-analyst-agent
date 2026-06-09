from __future__ import annotations

import os
import re
import json
import time
import uuid
from typing import Callable

from src.logging_config import get_logger

log = get_logger("agent")

# ---------------------------------------------------------------------------
# Lazy imports — cho phép test mock dễ dàng
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv optional

try:
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False
    ChatGroq = None  # type: ignore[assignment,misc]
    HumanMessage = SystemMessage = ToolMessage = None  # type: ignore[assignment,misc]

try:
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_exponential,
        retry_if_exception,
        RetryError,
    )

    _TENACITY_AVAILABLE = True
except ImportError:
    _TENACITY_AVAILABLE = False
    # Stub tenacity khi không có package (test environment)
    RetryError = Exception  # type: ignore[assignment,misc]

    def retry(**kwargs):  # type: ignore[misc]
        def decorator(fn):
            return fn

        return decorator

    def stop_after_attempt(n):
        return None  # type: ignore[misc]

    def wait_exponential(**kw):
        return None  # type: ignore[misc]

    def retry_if_exception(fn):
        return None  # type: ignore[misc]


try:
    from rich.console import Console

    _console = Console()

    def _print_dim(msg: str) -> None:
        _console.print(f"[dim]{msg}[/dim]")
except ImportError:

    def _print_dim(msg: str) -> None:  # type: ignore[misc]
        print(msg)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_STEPS = 5

# ---------------------------------------------------------------------------
# Rate-limit detection
# ---------------------------------------------------------------------------
try:
    from groq import RateLimitError as _GroqRateLimitError

    _HAS_GROQ_ERROR = True
except ImportError:
    _GroqRateLimitError = None  # type: ignore[assignment,misc]
    _HAS_GROQ_ERROR = False


def _is_rate_limit_error(exc: BaseException) -> bool:
    """
    Trả True nếu exception là Groq rate limit (429).
    Chỉ retry các lỗi này — không retry SystemExit hay KeyboardInterrupt.
    """
    if not isinstance(exc, Exception):
        return False
    if _HAS_GROQ_ERROR and isinstance(exc, _GroqRateLimitError):
        return True
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "rate_limit" in msg


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are a Data Analyst assistant that helps analyze data using SQL.

IMPORTANT - Database schema:
Table 'sales' has these exact columns:
- month (VARCHAR): e.g. '2024-01'
- city (VARCHAR): e.g. 'Hanoi', 'HCMC', 'Danang'
- product (VARCHAR): e.g. 'Laptop', 'Phone', 'Tablet'
- unit_price (INTEGER): price per unit
- quantity (INTEGER): number of units sold
- Revenue = CAST(unit_price AS BIGINT) * quantity (no revenue column, must calculate)

RULES:
- Always use list_tables first to check schema before querying
- Never use columns that don't exist
- To calculate revenue always write: CAST(unit_price AS BIGINT) * quantity
- NEVER write: unit_price * quantity (causes INT32 overflow)
- Respond in the same language as the user (Vietnamese if user writes Vietnamese)
- Always call tools using valid JSON format only"""


# ---------------------------------------------------------------------------
# Fallback XML parser (Groq quirk: tool call bị trả về dạng exception message)
# ---------------------------------------------------------------------------


def _extract_fallback_tool_call(error_str: str) -> tuple[str | None, dict | None]:
    """
    Một số version Groq SDK trả tool call dạng XML trong exception string.
    Pattern: '<function=tool_name {...args...} </function>'
    """
    # Linh hoạt hơn với whitespace giữa tên tool và args
    match = re.search(
        r"<function=(\w+)\s*(\{.*?\})\s*</function>",
        error_str,
        re.DOTALL,
    )
    if match:
        tool_name = match.group(1)
        try:
            tool_args = json.loads(match.group(2))
            return tool_name, tool_args
        except json.JSONDecodeError:
            log.warning(
                "fallback_xml_parse_json_error",
                extra={"raw_args": match.group(2)[:200]},
            )
    return None, None


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def create_analyst_agent(tools: list) -> Callable[[str], str]:
    """
    Factory trả về hàm run_agent(user_input: str) -> str.

    Args:
        tools: List LangChain tool objects để bind vào LLM.

    Returns:
        Callable nhận câu hỏi và trả lời dạng string.
    """
    if not _LANGCHAIN_AVAILABLE:
        raise RuntimeError(
            "langchain-groq is not installed. Run: pip install langchain-groq"
        )

    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=0,
    ).bind_tools(tools, tool_choice="auto")

    tool_map: dict = {t.name: t for t in tools}

    # ── Retry wrapper quanh raw LLM call ─────────────────────────────────
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(_is_rate_limit_error),
        reraise=True,
    )
    def _invoke_llm(messages: list):
        return llm.invoke(messages)

    # ── Main agent loop ───────────────────────────────────────────────────
    def run_agent(user_input: str) -> str:
        req_id = str(uuid.uuid4())[:8]
        t_start = time.perf_counter()

        log.info(
            "request_start",
            extra={
                "req_id": req_id,
                "input_len": len(user_input),
                "input_preview": user_input[:120],
            },
        )

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_input),
        ]

        for step in range(MAX_STEPS):
            # ── Gọi LLM ──────────────────────────────────────────────────
            try:
                response = _invoke_llm(messages)

            except RetryError as exc:
                log.error(
                    "llm_retry_exhausted",
                    extra={
                        "req_id": req_id,
                        "step": step,
                        "error": str(exc)[:300],
                    },
                )
                return (
                    "Lỗi: Groq API rate limit, đã thử lại 3 lần. "
                    "Vui lòng thử lại sau vài giây."
                )

            except Exception as exc:
                error_str = str(exc)
                log.warning(
                    "llm_error",
                    extra={
                        "req_id": req_id,
                        "step": step,
                        "error": error_str[:300],
                    },
                )

                # Groq quirk: tool call nằm trong exception string
                tool_name, tool_args = _extract_fallback_tool_call(error_str)
                if tool_name and tool_name in tool_map:
                    _print_dim(f"→ Fallback tool: {tool_name}({tool_args})")
                    log.info(
                        "fallback_tool_call",
                        extra={
                            "req_id": req_id,
                            "step": step,
                            "tool": tool_name,
                        },
                    )
                    try:
                        result = tool_map[tool_name].invoke(tool_args)
                        messages.append(
                            HumanMessage(
                                content=(
                                    f"Tool {tool_name} returned: {result}\n"
                                    "Please summarize this result for the user."
                                )
                            )
                        )
                        continue
                    except Exception as tool_exc:
                        log.error(
                            "fallback_tool_error",
                            extra={
                                "req_id": req_id,
                                "tool": tool_name,
                                "error": str(tool_exc),
                            },
                        )
                        return f"Lỗi khi chạy tool '{tool_name}': {tool_exc}"

                elif tool_name and tool_name not in tool_map:
                    # Tool trong XML không tồn tại
                    log.warning(
                        "fallback_unknown_tool",
                        extra={
                            "req_id": req_id,
                            "tool": tool_name,
                        },
                    )
                    return (
                        f"Lỗi: Tool '{tool_name}' không tồn tại "
                        f"trong danh sách tools được cấu hình."
                    )

                return f"Lỗi gọi API: {error_str}"

            messages.append(response)

            # ── Không có tool call → câu trả lời cuối ────────────────────
            if not response.tool_calls:
                content = response.content or ""

                # Groq quirk: đôi khi trả về content rỗng mà không có tool call
                if not content.strip():
                    log.warning(
                        "empty_response",
                        extra={
                            "req_id": req_id,
                            "step": step,
                        },
                    )
                    return "Xin lỗi, tôi không thể tạo câu trả lời cho câu hỏi này."

                elapsed_ms = round((time.perf_counter() - t_start) * 1000)
                log.info(
                    "request_done",
                    extra={
                        "req_id": req_id,
                        "steps": step + 1,
                        "latency_ms": elapsed_ms,
                    },
                )
                return content

            # ── Thực thi từng tool call ───────────────────────────────────
            for tc in response.tool_calls:
                t_name: str = tc["name"]
                t_args: dict = tc["args"]

                _print_dim(f"→ Tool: {t_name}({t_args})")
                log.info(
                    "tool_call",
                    extra={
                        "req_id": req_id,
                        "step": step,
                        "tool": t_name,
                        "args_preview": str(t_args)[:200],
                    },
                )

                if t_name in tool_map:
                    try:
                        result = tool_map[t_name].invoke(t_args)
                        log.info(
                            "tool_result",
                            extra={
                                "req_id": req_id,
                                "tool": t_name,
                                "result_preview": str(result)[:200],
                            },
                        )
                    except Exception as tool_exc:
                        result = f"Tool error: {tool_exc}"
                        log.error(
                            "tool_exception",
                            extra={
                                "req_id": req_id,
                                "tool": t_name,
                                "error": str(tool_exc),
                            },
                        )
                else:
                    result = f"Tool '{t_name}' không tồn tại."
                    log.warning(
                        "unknown_tool",
                        extra={
                            "req_id": req_id,
                            "tool": t_name,
                        },
                    )

                messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=tc["id"],
                    )
                )

        # ── Hết MAX_STEPS mà vẫn chưa có câu trả lời ─────────────────────
        elapsed_ms = round((time.perf_counter() - t_start) * 1000)
        log.warning(
            "max_iterations_reached",
            extra={
                "req_id": req_id,
                "max_steps": MAX_STEPS,
                "latency_ms": elapsed_ms,
            },
        )
        return f"Không thể hoàn thành sau {MAX_STEPS} vòng."

    return run_agent
