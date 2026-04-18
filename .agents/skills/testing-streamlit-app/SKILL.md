# Testing Real-AI-R A股量化交易系统

## Prerequisites
- Python 3.10+
- Dependencies installed: `pip install -e ".[dev]"`
- No external API keys needed (AKShare is free, no registration)

## Starting the App
```bash
cd /home/ubuntu/repos/Real-AI-R
streamlit run streamlit_app/app.py --server.headless true
```
The app runs at `http://localhost:8501`. If port 8501 is occupied, kill the existing process first:
```bash
fuser -k 8501/tcp
```

## Unit Tests
```bash
pytest tests/ -v
ruff check src/
```
Expect 76+ tests passing. Run these before UI testing to catch logic errors early.

## App Structure (9 Tabs)
The dashboard has 9 tabs accessible via the top tab bar:

| Tab | Name | Data Source | Load Time |
|-----|------|------------|----------|
| 1 | 🚀 策略回测 | Single stock daily data | 3-5s |
| 2 | 📡 板块监控 | All sector boards | 5-10s |
| 3 | 🔮 热门板块预测 | Sector + fund flow | 10-15s |
| 4 | 💎 个股推荐 | Board constituent stocks | 5-10s |
| 5 | 🏷️ 板块分类 | All boards + classification | 5-10s |
| 6 | 🔄 周期轮动 | Index + commodity data | 5-10s |
| 7 | 🚀 科技赛道 | 6 tech track boards | 5-10s |
| 8 | 🚫 避雷指南 | Red-line boards | 3-5s |
| 9 | ⚔️ 攻防组合 | Multi-sector stock data | **30-60s** |

## Testing Each Tab

### Tab 1: 策略回测
- Select stock code (default 000001), date range, strategy
- Click "🚀 运行回测"
- Verify: 5 metric cards, 4 charts (net value, drawdown, heatmap, K-line), trade log table
- Test all 3 strategies: MA Cross, MACD, Bollinger — each should produce different results

### Tab 2: 板块监控
- Data loads automatically
- Verify: stat cards (board count, avg change), Top20 bar chart, full table

### Tab 3: 热门板块预测
- Click "开始预测" button
- Verify: 10 scored boards with radar chart, score range typically 50-80

### Tab 4: 个股推荐
- Select a board from dropdown, click "获取推荐"
- Verify: ST stocks filtered out, composite scores 0-100, scatter chart

### Tab 5: 板块分类
- Data loads automatically
- Verify: pie chart with 4 segments (科技主线/周期主线/红线禁区/其他), bar chart, classification table
- Adversarial: at least 1 科技主线 board should exist

### Tab 6: 周期轮动
- Data loads automatically
- Verify: exactly 5 stages displayed, each with temperature gauge and matched boards
- Stages should be in order: 贵金属 → 基本金属 → 传统能源 → 农业后周期 → 必选消费

### Tab 7: 科技赛道
- Data loads automatically
- Verify: 6 tracks with varying heat scores, radar chart, leading stocks for each track
- Adversarial: scores should not all be identical

### Tab 8: 避雷指南
- Data loads automatically
- Verify: red board count > 0, 房地产 appears in red list
- 4 red-line zones: 房地产, 白酒, 零售, 旧软件

### Tab 9: 攻防组合 (❗ Slowest tab)
- Adjust sliders if desired (default: 60% attack, 5 attack stocks, 4 defense stocks)
- Click "⚔️ 构建组合" and **wait 30-60 seconds**
- Verify: success message with counts, pie chart (2 segments), attack table (科技), defense table (周期)
- Adversarial: attack reasons should mention "科技赛道", defense reasons should mention "周期阶段"
- Stock codes should be 6-digit format (600xxx, 000xxx, 300xxx, 688xxx)

## Common Issues

### Port 8501 already in use
```bash
fuser -k 8501/tcp
```

### AKShare data fetch timeout
AKShare calls external APIs and may occasionally timeout, especially during non-trading hours (weekends, holidays, after 15:00 CST). Retry usually works. The 攻防组合 tab is the slowest because it fetches stock data from multiple sectors sequentially.

### Streamlit reruns on interaction
Streamlit reruns the entire script when any widget changes. This means clicking a tab will not lose data from other tabs, but buttons like "构建组合" need to be clicked again after any sidebar change.

## Devin Secrets Needed
None — AKShare is completely free and requires no API keys or registration.
