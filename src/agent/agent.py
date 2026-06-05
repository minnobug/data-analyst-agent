import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from rich.console import Console

load_dotenv()
console = Console()

def create_analyst_agent(tools: list):
    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=0,
    ).bind_tools(tools, tool_choice="auto")

    system_prompt = """You are a Data Analyst assistant that helps analyze data using SQL.

IMPORTANT - Database schema:
Table 'sales' has these exact columns:
- month (VARCHAR): e.g. '2024-01'
- city (VARCHAR): e.g. 'Hanoi', 'HCMC', 'Danang'  
- product (VARCHAR): e.g. 'Laptop', 'Phone', 'Tablet'
- unit_price (INTEGER): price per unit
- quantity (INTEGER): number of units sold
- Revenue = unit_price * quantity (no revenue column, must calculate)

RULES:
- Always use list_tables first to check schema before querying
- Never use columns that don't exist
- To calculate revenue: unit_price * quantity
- Respond in the same language as the user (Vietnamese if user writes Vietnamese)"""

    tool_map = {t.name: t for t in tools}

    def run_agent(user_input: str) -> str:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input),
        ]

        for i in range(5):
            try:
                response = llm.invoke(messages)
            except Exception as e:
                return f"Lỗi gọi API: {str(e)}"

            messages.append(response)

            if not response.tool_calls:
                return response.content

            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]
                console.print(f"[dim]→ Tool: {tool_name}({tool_args})[/dim]")

                if tool_name in tool_map:
                    try:
                        result = tool_map[tool_name].invoke(tool_args)
                    except Exception as e:
                        result = f"Tool error: {str(e)}"
                else:
                    result = f"Tool '{tool_name}' không tồn tại"

                messages.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tc["id"],
                ))

        return "Không thể hoàn thành sau 5 vòng."

    return run_agent
