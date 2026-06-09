# Data Analyst Agent

AI agent that answers questions about your data in plain English or Vietnamese — no SQL knowledge needed.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![CI](https://github.com/minnobug/data-analyst-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/minnobug/data-analyst-agent/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## What it does

Ask questions in natural language → agent writes and runs SQL → returns the answer.

```
You:   Sản phẩm nào bán chạy nhất?
Agent: Phone — 75 units sold across all cities.

You:   Total revenue by city?
Agent: Danang 2.4B  |  Hanoi 3.0B  |  HCMC 5.7B
```

Supports **English and Vietnamese** out of the box.

---

## Stack

| Layer | Tool |
|---|---|
| LLM | Groq — Llama 3.3 70B |
| Agent framework | LangChain |
| Database | DuckDB |
| CLI | Rich |
| Retry / resilience | Tenacity |

---

## Project structure

```
data-analyst-agent/
├── src/
│   ├── agent/
│   │   └── agent.py          # LangChain agentic loop + Groq integration
│   ├── tools/
│   │   ├── sql_tool.py       # query_sql and list_tables tools (DuckDB)
│   │   └── file_tool.py      # extensible file ingestion (WIP)
│   └── logging_config.py     # JSON structured logger
├── tests/
│   ├── conftest.py            # stubs for all external deps (no live services needed)
│   ├── test_agent.py          # 43 cases — agent loop, retry, fallback XML
│   ├── test_sql_tool.py       # 47 cases — sanitize, overflow rewrite, conn lifecycle
│   └── test_logging_config.py # 26 cases — JsonFormatter, extra fields, LOG_LEVEL
├── data/sample/
│   └── warehouse.db           # auto-created on first run
├── main.py                    # entry point
├── pyproject.toml
└── .env.example
```

---

## Quick start

**1. Clone and install**
```bash
git clone https://github.com/minnobug/data-analyst-agent.git
cd data-analyst-agent
pip install -e .
```

**2. Set up environment**
```bash
cp .env.example .env
# Add your Groq API key — free at console.groq.com
```

```env
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
LOG_LEVEL=INFO
```

**3. Run**
```bash
python main.py
```

---

## Run tests

```bash
# All 116 tests — no live services or API keys needed
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing
```

---

## Configuration

| Env var | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | required | Groq API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Model to use |
| `WAREHOUSE_DB` | `data/sample/warehouse.db` | Path to DuckDB file |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` |

---

## Roadmap

- [x] Natural language → SQL (English + Vietnamese)
- [x] DuckDB local warehouse
- [x] Groq rate-limit retry with tenacity
- [x] JSON structured logging
- [x] 116 unit tests, CI on GitHub Actions
- [ ] Connect to SmartCity pipeline (S3 Parquet via DuckDB httpfs)
- [ ] File ingestion tool (CSV, Parquet upload)
- [ ] Streamlit web UI

---

## License

MIT — see [LICENSE](LICENSE)