# 🤖 Data Analyst Agent

> AI agent phân tích dữ liệu bằng ngôn ngữ tự nhiên — hỏi bằng tiếng Việt hay tiếng Anh, agent tự viết SQL và trả kết quả.

Built with **Groq (Llama 3.3 70B)** + **LangChain** + **DuckDB**

---

## ✨ Features

- 💬 Natural language to SQL — không cần biết SQL
- 🇻🇳 Hỗ trợ tiếng Việt
- ⚡ Groq inference cực nhanh (Llama 3.3 70B)
- 🦆 DuckDB local warehouse — không cần server
- 🔄 Agentic loop — tự kiểm tra schema trước khi query
- 🛡️ Tự xử lý lỗi INT32 overflow khi tính doanh thu lớn

---

## 🗂️ Project Structure

```
data-analyst-agent/
├── src/
│   ├── agent/
│   │   └── agent.py        # LangChain agent + fallback error handler
│   └── tools/
│       ├── sql_tool.py     # DuckDB query + list_tables tools
│       └── file_tool.py    # (extensible)
├── tests/
├── main.py                 # Entry point
├── pyproject.toml
└── .env.example
```

---

## 🚀 Setup

**1. Clone & cài dependencies**
```bash
git clone https://github.com/minnobug/data-analyst-agent.git
cd data-analyst-agent
pip install -e .
```

**2. Tạo file `.env`**
```bash
cp .env.example .env
```

Thêm Groq API key vào `.env`:
```
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

> Lấy API key miễn phí tại [console.groq.com](https://console.groq.com)

**3. Chạy**
```bash
python main.py
```

---

## 💡 Example Queries

```
You: Sản phẩm nào bán chạy nhất?
Agent: Sản phẩm bán chạy nhất là Phone với tổng số lượng 75.

You: Tổng doanh thu theo từng thành phố?
Agent:
  - Đà Nẵng: 2.400.000.000
  - Hà Nội:  3.000.000.000
  - TP.HCM:  5.700.000.000

You: Tháng nào HCMC bán được nhiều nhất và sản phẩm gì?
Agent: Tháng 2024-01 với sản phẩm Phone, doanh thu 3.600.000.000.
```

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| LLM | Groq — Llama 3.3 70B Versatile |
| Agent framework | LangChain |
| Database | DuckDB |
| CLI UI | Rich |

---

## 📄 License

MIT
