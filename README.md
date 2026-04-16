# Real-AI-R — 用宏观周期思维做 A 股量化

> **一句话**：为相信"A 股随周期而动"的 Python 交易员准备的宏观轮动回测框架。

把你的宏观假设（"周期强势→配置周期板块"）在真实 A 股历史上回测验证。
零 API key，零注册，一行 `pip install` 就开跑。

---

## 谁应该用它

- **定位**：A 股**板块轮动**策略的研究与回测
- **适合**：有 Python 基础、相信泽平宏观/美林时钟类框架的主动型交易员
- **不适合**：单股技术分析（这里关注的是板块/行业级别动量轮动）

---

## 60 秒入门

```bash
git clone https://github.com/solfkimblee/Real-AI-R.git
cd Real-AI-R
pip install -e ".[dev]"
python scripts/wf_backtest.py --demo          # 合成数据冒烟，30 秒出结果
```

看到 V5/V7/V8/V10 对比表 → 环境就绪。然后：

👉 **[docs/QUICKSTART.md](docs/QUICKSTART.md)** — 5 分钟跑完第一次真实回测
👉 **[docs/strategy-guide.md](docs/strategy-guide.md)** — V5 ~ V10 对比，挑策略

---

## 核心价值（三个层级）

### 🎯 Level 1 — 宏观板块轮动策略（主要价值）

`src/real_ai_r/macro/` 下的 **泽平宏观系列**，每日给 90+ 板块打分，选 Top-10 持仓：

| 版本 | 核心思想 | 真实 WF Sharpe |
|------|----------|--------------|
| **V8** ⭐ | 连续制度调整 + 双时间尺度 + 回撤制动 | **+2.86** (7/7 正窗口) |
| V7 | 动态科技/周期配比 | +2.60 |
| V9.2 | Hedge 元集成 (V5/V7/V8/V9 软融合) | +2.10 |
| V10 | 泽平三维度 × 产业链 × 卖铲人 | +0.75 |

```python
from real_ai_r.macro import ZepingMacroStrategyV8
strat = ZepingMacroStrategyV8()
result = strat.predict(board_df, top_n=10,
                       tech_history=..., cycle_history=...)
# result.predictions[0].board_name → 当日 Top-1 板块
```

### 🔧 Level 2 — 单股策略回测引擎

事件驱动，严格 T+1、手续费、滑点、印花税：

```python
from real_ai_r.data.fetcher import DataFetcher
from real_ai_r.engine.backtest import BacktestEngine
from real_ai_r.strategies.ma_cross import MACrossStrategy

fetcher = DataFetcher()
data = fetcher.get_stock_daily("000001", start_date="2023-01-01", end_date="2024-01-01")
engine = BacktestEngine(data=data, strategy=MACrossStrategy(5, 20),
                        initial_capital=100000.0,
                        commission_rate=0.0003, slippage=0.001)
print(engine.run().metrics)
```

### 📊 Level 3 — 可视化 Dashboard

```bash
streamlit run streamlit_app/app.py
```

---

## 快速回测你自己的想法

```bash
# 多策略一键对比 (需准备 panel_*.parquet，见 data/README.md)
python scripts/wf_backtest.py \
    --panel data/panel_val.parquet \
    --strategies V5,V7,V8,V9.2,V10 \
    --windows 7 --window-size 20 \
    --report-md report.md
```

输出：每策略的 Sharpe / IR / 累计超额 / 最大回撤 / 胜率 + 逐窗口分解。

---

## 项目结构

```
src/real_ai_r/
├── macro/           # ⭐ 宏观板块轮动策略 (V5/V7/V8/V9.2/V9.3/V10)
├── v9/              # V9 在线学习框架 (因子库 + HMM + QP + Hedge)
├── engine/          # 单股回测引擎
├── strategies/      # 单股策略 (MA/MACD/布林)
├── data/            # AKShare 封装 + 本地缓存
└── analysis/        # 绩效指标 + Plotly 图表

scripts/wf_backtest.py  # 走步回测 harness（支持全部策略对比）
streamlit_app/          # Streamlit Dashboard
docs/                   # QUICKSTART + strategy-guide
tests/                  # 200+ 单测
```

---

## 文档索引

- [docs/QUICKSTART.md](docs/QUICKSTART.md) — 5 分钟入门
- [docs/strategy-guide.md](docs/strategy-guide.md) — 策略版本选型
- [data/README.md](data/README.md) — 数据准备规范

## License

MIT
