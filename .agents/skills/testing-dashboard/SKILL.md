# Testing: Real-AI-R Streamlit Dashboard

## Overview
The Real-AI-R system is a Streamlit-based A-share quantitative trading platform with 4 tabs:
1. **策略回测** — Backtest strategies (MA Cross, MACD, Bollinger)
2. **板块监控** — Real-time sector monitoring (industry/concept boards)
3. **热门板块预测** — Hot sector prediction (5-factor scoring model)
4. **个股推荐** — Stock recommendation within sectors

## Prerequisites
- Python environment with dependencies installed: `pip install -e .`
- No API keys or credentials needed — AKShare is completely free
- No login required

## Devin Secrets Needed
None — all data sources (AKShare/东方财富) are free and require no authentication.

## Starting the App
```bash
cd /home/ubuntu/repos/Real-AI-R
streamlit run streamlit_app/app.py --server.headless true --server.port 8501
```
Wait ~5 seconds, then verify with `curl -s -o /dev/null -w "%{http_code}" http://localhost:8501` (expect 200).

If port 8501 is already in use, check for existing Streamlit processes: `lsof -i :8501`.

## Testing Each Tab

### Tab 1: 策略回测
- Default parameters are pre-filled (000001, 2023-2024, 双均线交叉)
- Click "🚀 运行回测" in the sidebar
- Wait ~10-15s for data fetch + backtest
- Verify: 5 metric cards, 4 charts (净值曲线/回撤/月度热力图/K线+信号), trade log table

### Tab 2: 板块监控
- Click the "📡 板块监控" tab — data auto-loads on tab switch
- Wait ~10s for AKShare API calls
- Verify: 4 stat cards (上涨/下跌/平盘/平均涨幅), histogram, Top20 bar chart, full table
- Can toggle between 行业板块 and 概念板块

### Tab 3: 热门板块预测
- Click "🔮 热门板块预测" tab
- Select board type (行业板块/概念板块), set 推荐数量 (default 10)
- Click "🔮 开始预测" button
- Wait ~20-30s for prediction (fetches board data + fund flow data)
- Verify: success message, score bar chart, radar chart (Top 5), detail table with factor scores
- Factor weight sliders available in "调整因子权重" expander

### Tab 4: 个股推荐
- Click "💎 个股推荐" tab
- Select board from dropdown (pre-populated based on board type)
- Click "💎 获取推荐" button
- Wait ~15-20s for stock data fetch
- Verify: success message, bar chart, scatter/bubble chart, detail table
- **Critical check**: No stock names should contain "ST" or "退市" (ST filtering)
- "查看全部成分股" expander shows unfiltered data for comparison

## Common Issues

### Slow API responses
AKShare fetches from 东方财富 APIs. During market hours, responses may be slower. Non-trading hours return cached/last-day data. Allow 20-30s for prediction tab which makes multiple API calls.

### Small sectors
Some sectors (e.g., 钴) have very few constituent stocks. After ST filtering, the recommended count may be less than the requested top_n. This is expected behavior.

### Port conflicts
If Streamlit fails to start on port 8501, kill existing processes: `pkill -f streamlit` then retry.

## Running Unit Tests
```bash
cd /home/ubuntu/repos/Real-AI-R
pytest tests/ -v
```
Expected: 38 tests pass (23 backtest + 15 sector analysis). All tests use mocked API calls.

## Lint Check
```bash
ruff check src/ streamlit_app/ tests/
```
