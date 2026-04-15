# Real-AI-R 🔴 A股量化交易系统

A股量化交易系统 — 策略开发与回测平台

## 功能特性

- **数据获取** — 基于 AKShare，支持A股日线/分钟线/财务数据，零注册门槛
- **回测引擎** — 事件驱动架构，严格T+1规则、手续费、滑点模拟
- **策略模板** — 可插拔设计，内置经典策略（双均线、MACD、布林带）
- **绩效分析** — 收益率、夏普比率、最大回撤、胜率、年化收益等
- **可视化Dashboard** — Streamlit 交互式界面，支持策略回测与结果展示

## 项目结构

```
Real-AI-R/
├── src/real_ai_r/        # 核心代码
│   ├── data/             # 数据获取模块（AKShare封装）
│   ├── engine/           # 回测引擎（事件驱动）
│   ├── strategies/       # 策略模板
│   ├── analysis/         # 绩效分析
│   └── utils/            # 工具函数
├── streamlit_app/        # Streamlit Dashboard
├── tests/                # 单元测试
└── pyproject.toml        # 项目配置
```

## 快速开始

### 安装

```bash
pip install -e ".[dev]"
```

### 运行回测示例

```python
from real_ai_r.data.fetcher import DataFetcher
from real_ai_r.engine.backtest import BacktestEngine
from real_ai_r.strategies.ma_cross import MACrossStrategy

# 1. 获取数据
fetcher = DataFetcher()
data = fetcher.get_stock_daily("000001", start_date="2023-01-01", end_date="2024-01-01")

# 2. 创建策略
strategy = MACrossStrategy(short_window=5, long_window=20)

# 3. 运行回测
engine = BacktestEngine(
    data=data,
    strategy=strategy,
    initial_capital=100000.0,
    commission_rate=0.0003,  # 万三手续费
    slippage=0.001,          # 0.1% 滑点
)
result = engine.run()

# 4. 查看结果
print(result.summary())
```

### 启动Dashboard

```bash
streamlit run streamlit_app/app.py
```

## 内置策略

| 策略 | 说明 | 参数 |
|------|------|------|
| **双均线策略** | 短期均线上穿长期均线买入，下穿卖出 | short_window, long_window |
| **MACD策略** | 基于MACD金叉/死叉信号 | fast, slow, signal |
| **布林带策略** | 价格触及下轨买入，触及上轨卖出 | window, num_std |

## 技术栈

- **Python 3.10+**
- **AKShare** — A股数据获取
- **Pandas / NumPy** — 数据处理
- **TA-Lib (ta)** — 技术指标计算
- **Plotly** — 交互式图表
- **Streamlit** — Web Dashboard

## License

MIT
