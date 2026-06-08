import os
import re
import json
import time
import uuid

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from rich.console import Console
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
)

from src.logging_config import get_logger

load_dotenv()

console = Console()
log = get_logger("agent")

# ── System prompt ─────────────────────────────────────────────────────────────
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
- NEVER write: unit_price * quantity (causes overflow)
- Respond in the same language as the user (Vietnamese if user writes Vietnamese)
- Always call tools using valid JSON format only"""

# ── Exceptions that warrant a retry ──────────────────────────────────────────
# Groq SDK raises groq.RateLimitError on 429; fall back to broad Exception
# with message inspection when the SDK version doesn't export it.
try:
    from groq import RateLimitError as _GroqRateLimitError

    _RETRY_ON = (_GroqRateLimitError,)
except ImportError:
    _RETRY_ON = (Exception,)  # type: ignore[assignment]


def _is_rate_limit(exc: Exception) -> bool:
    return (
        isinstance(exc, _RETRY_ON)
        or "429" in str(exc)
        or "rate limit" in str(exc).lower()
    )


def create_analyst_agent(tools: list):
    """
    Factory that returns a run_agent(user_input) -> str closure.

    Args:
        tools: List of LangChain tool objects to bind to the LLM.
    """
    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=0,
    ).bind_tools(tools, tool_choice="auto")

    tool_map: dict = {t.name: t for t in tools}

    # ── Retry wrapper around the raw LLM call ─────────────────────────────────
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(_RETRY_ON),
        reraise=True,
    )
    def _invoke_llm(messages: list):
        return llm.invoke(messages)

    # ── Fallback: parse Groq XML-style tool call from exception string ────────
    def _extract_fallback_tool_call(error_str: str) -> tuple[str | None, dict | None]:
        match = re.search(r"'<function=(\w+)\s+(\{.*?\})\s*</function>'", error_str)
        if match:
            tool_name = match.group(1)
            try:
                tool_args = json.loads(match.group(2))
                return tool_name, tool_args
            except json.JSONDecodeError:
                pass
        return None, None

    # ── Main agent loop ───────────────────────────────────────────────────────
    def run_agent(user_input: str) -> str:
        req_id = str(uuid.uuid4())[:8]
        t_start = time.perf_counter()

        log.info(
            "request_start",
            extra={
                "req_id": req_id,
                "input_preview": user_input[:200],
            },
        )

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_input),
        ]

        for step in range(5):
            # ── Call LLM (with retry on rate-limit) ──────────────────────────
            try:
                response = _invoke_llm(messages)
            except RetryError as exc:
                log.error(
                    "llm_retry_exhausted", extra={"req_id": req_id, "error": str(exc)}
                )
                return "Lỗi: Groq API rate limit, đã thử lại 3 lần. Vui lòng thử sau."
            except Exception as exc:
                error_str = str(exc)
                log.warning(
                    "llm_error_fallback",
                    extra={"req_id": req_id, "error": error_str[:300]},
                )

                # Groq trả về XML-style tool call → tự parse và chạy
                tool_name, tool_args = _extract_fallback_tool_call(error_str)
                if tool_name and tool_name in tool_map:
                    console.print(
                        f"[dim]→ Fallback tool: {tool_name}({tool_args})[/dim]"
                    )
                    log.info(
                        "fallback_tool_call",
                        extra={
                            "req_id": req_id,
                            "tool": tool_name,
                            "step": step,
                        },
                    )
                    try:
                        result = tool_map[tool_name].invoke(tool_args)
                        messages.append(
                            HumanMessage(
                                content=(
                                    f"Tool {tool_name} returned: {result}\n"
                                    "Please summarize this result in Vietnamese."
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
                        return f"Lỗi tool: {tool_exc}"

                return f"Lỗi gọi API: {error_str}"

            messages.append(response)

            # ── No tool calls → final answer ──────────────────────────────────
            if not response.tool_calls:
                elapsed_ms = round((time.perf_counter() - t_start) * 1000)
                log.info(
                    "request_done",
                    extra={
                        "req_id": req_id,
                        "steps": step + 1,
                        "latency_ms": elapsed_ms,
                    },
                )
                return response.content

            # ── Execute each tool call ────────────────────────────────────────
            for tc in response.tool_calls:
                t_name: str = tc["name"]
                t_args: dict = tc["args"]

                console.print(
                    f"[dim]→ Tool: [magenta]{t_name}[/magenta]({t_args})[/dim]"
                )
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
                                "success": "Error" not in str(result),
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
                        "unknown_tool", extra={"req_id": req_id, "tool": t_name}
                    )

                messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=tc["id"],
                    )
                )

        elapsed_ms = round((time.perf_counter() - t_start) * 1000)
        log.warning(
            "max_iterations_reached",
            extra={
                "req_id": req_id,
                "latency_ms": elapsed_ms,
            },
        )
        return "Không thể hoàn thành sau 5 vòng."

    return run_agent
