import duckdb
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from src.agent.agent import create_analyst_agent
from src.tools.sql_tool import query_sql, list_tables

console = Console()

def seed_sample_data():
    Path("data/sample").mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect("data/sample/warehouse.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sales AS
        SELECT * FROM (VALUES
            ('2024-01', 'Hanoi',  'Laptop', 150000000, 12),
            ('2024-01', 'HCMC',   'Phone',   80000000, 45),
            ('2024-02', 'Hanoi',  'Tablet',  60000000, 20),
            ('2024-02', 'HCMC',   'Laptop', 150000000,  8),
            ('2024-03', 'Danang', 'Phone',   80000000, 30),
            ('2024-03', 'HCMC',   'Tablet',  60000000, 15)
        ) t(month, city, product, unit_price, quantity)
    """)
    conn.close()
    console.print("[green]✓ Sample data created[/green]")

def main():
    console.print(Panel.fit(
        "[bold cyan]Data Analyst Agent[/bold cyan]\n"
        "Powered by Groq (Llama 3) + LangChain + DuckDB\n"
        "Type [yellow]'exit'[/yellow] to quit",
        border_style="cyan",
    ))

    seed_sample_data()
    tools = [query_sql, list_tables]
    run_agent = create_analyst_agent(tools)

    while True:
        try:
            question = console.input("\n[bold cyan]You:[/bold cyan] ").strip()
            if question.lower() in ("exit", "quit", "q"):
                console.print("[yellow]Goodbye![/yellow]")
                break
            if not question:
                continue
            response = run_agent(question)
            console.print(f"\n[bold green]Agent:[/bold green] {response}")
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopped.[/yellow]")
            break

if __name__ == "__main__":
    main()
