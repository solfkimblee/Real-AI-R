# Testing: Real-AI-R Streamlit Dashboard

## Overview
End-to-end testing of the A-share quantitative trading Streamlit dashboard, covering data fetching, backtest execution, strategy switching, chart rendering, and trade log display.

## Prerequisites

### Dependencies
```bash
pip install -e ".[dev]"
```

### Start the App
```bash
streamlit run streamlit_app/app.py --server.port 8501 --server.headless true
```
App will be available at `http://localhost:8501`.

## Testing Procedure

### 1. Welcome Page Verification
- Open `http://localhost:8501` in Chrome
- Verify title: "📈 Real-AI-R — A股量化交易回测平台"
- Verify sidebar shows: 回测参数 header, 标的选择 section, 策略选择 section, 资金设置 section
- Default values: stock code "000001", dates 2023/01/01-2024/12/31, strategy "双均线交叉"

### 2. Run Backtest (Primary Flow)
- The "🚀 运行回测" button is at the bottom of the sidebar — you may need to scroll the sidebar container down to see it. Use JS: `document.querySelector('[data-testid="stSidebarContent"]').scrollTop = 9999` or scroll within the sidebar area
- Click the button and wait for results
- Verify: success message "获取到 XXX 条日线数据" (XXX > 0)
- Verify 5 metric cards: 总收益率, 年化收益率, 夏普比率, 最大回撤, 日胜率
- Scroll main content to verify 4 charts: 组合净值曲线, 回撤曲线, 月度收益热力图, K线图+交易信号
- Scroll further to verify: 📋 详细绩效指标 table (15+ metrics), 📝 交易记录 table (买入/卖出 records)

### 3. Strategy Switching
All 3 strategies should be tested:

| Strategy | Sidebar Params | Expected Chart Label |
|----------|---------------|---------------------|
| 双均线交叉 | 短期均线 (5), 长期均线 (20) | MA_Cross(5,20) |
| MACD策略 | MACD快线 (12), MACD慢线 (26), 信号线 (9) | MACD(12,26,9) |
| 布林带策略 | 布林带周期 (20), 标准差倍数 (2.00) | Bollinger(20,2.0) |

- Click the strategy dropdown in the sidebar to switch
- Verify sidebar parameters change to match the selected strategy
- Click "🚀 运行回测" and verify results differ between strategies
- Each strategy should produce different metric values (proves strategy actually changed)

## Key Architecture Notes

- **Data source**: AKShare (free, no registration required). Fetches via `DataFetcher.get_stock_daily()`
- **Backtest engine**: Event-driven with T+1 trading rule, commission (万三), stamp tax (千一 on sell)
- **App entry point**: `streamlit_app/app.py`
- **Charts module**: `src/real_ai_r/analysis/charts.py` — uses Plotly for interactive charts
- **Unit tests**: `pytest tests/ -v` (23 tests covering engine, indicators, strategies)
- **Lint**: `ruff check src/ tests/ streamlit_app/`

## Common Issues

- The "🚀 运行回测" button might be offscreen in the sidebar on smaller viewports. Scroll the sidebar container or use JavaScript to scroll it into view.
- AKShare data fetching depends on network access to Chinese financial data APIs. If fetching fails, it might be a network/API availability issue rather than a code bug.
- Streamlit reruns the entire script when any widget changes, so switching strategy resets the page to the welcome state until you click run again.

## Devin Secrets Needed
None — AKShare requires no API keys or registration.
