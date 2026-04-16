# CLAUDE.md — Real-AI-R 项目开发指南

## 项目概述

Real-AI-R 是一个 **A股量化交易系统**，包含策略开发、回测引擎、板块预测和宏观分析四大模块。项目以"泽平宏观"投资方法论为核心框架，融合规则引擎与 ML 模型，提供板块级别的投资决策支持。

## 技术栈

- **Python 3.10+**，使用 `hatchling` 构建
- **AKShare** — A股数据获取（免费，零注册）
- **Pandas / NumPy** — 数据处理
- **ta** — 技术指标计算（SMA/EMA/MACD/RSI/布林带）
- **LightGBM + scikit-learn** — ML 板块热度预测
- **Plotly** — 交互式图表
- **Streamlit** — Web Dashboard
- **Ruff** — 代码检查（E/F/I/W 规则，行宽100）
- **Pytest** — 单元测试

## 常用命令

```bash
# 安装（开发模式）
pip install -e ".[dev]"

# 运行测试
pytest tests/

# 代码检查
ruff check src/ tests/

# 启动 Dashboard
streamlit run streamlit_app/app.py
```

## 目录结构与模块职责

```
src/real_ai_r/
├── data/               # 数据层
│   ├── fetcher.py        # DataFetcher — AKShare 封装，个股/指数日线、实时行情、CSV缓存
│   ├── indicators.py     # 技术指标计算（MA/EMA/MACD/RSI/布林带/StochRSI）
│   ├── em_history_db.py  # EMHistoryDB — SQLite 自建板块历史库（盘后采集快照）
│   └── em_realtime_enhancer.py  # 实时数据增强
│
├── engine/             # 回测引擎
│   └── backtest.py       # BacktestEngine — 事件驱动，T+1规则、手续费、滑点、仓位管理
│                         # BacktestResult — 净值曲线、持仓、交易记录
│
├── strategies/         # 经典策略模板
│   ├── base.py           # BaseStrategy(ABC) + Signal(BUY/SELL/HOLD) 枚举
│   ├── ma_cross.py       # MACrossStrategy — 双均线交叉
│   ├── macd_strategy.py  # MACDStrategy — MACD 金叉/死叉
│   └── bollinger_strategy.py  # BollingerStrategy — 布林带均值回归
│
├── analysis/           # 绩效分析与可视化
│   ├── performance.py    # calculate_metrics() — 夏普/索提诺/回撤/胜率/盈亏比等
│   └── charts.py         # Plotly 图表（净值曲线/回撤/K线信号/月度热力图）
│
├── sector/             # 板块监控与预测
│   ├── monitor.py        # SectorMonitor — 行业/概念板块实时行情、资金流向
│   ├── predictor.py      # HotSectorPredictor — 五因子评分预测明日热门板块
│   └── recommender.py    # StockRecommender — 板块内个股综合评分推荐
│
├── macro/              # 泽平宏观策略体系（核心）
│   ├── classifier.py     # SectorClassifier — 板块三分类（科技/周期/红线）+ 关键词匹配
│   │                     # TECH_TRACKS(6赛道) / CYCLE_STAGES(5段论) / REDLINE_ZONES(4禁区)
│   │                     # TECH_TRACKS_V5 — 扩展版9赛道，覆盖率3%→15%
│   ├── red_filter.py     # RedLineFilter — 红线禁区过滤（板块级+个股ST过滤）
│   ├── tech_tracker.py   # TechTracker — 科技六赛道热度追踪
│   ├── cycle_tracker.py  # CycleTracker — 大宗商品五段论温度计
│   ├── portfolio.py      # AttackDefensePortfolio — 攻防组合（科技矛6:周期盾4）
│   ├── zeping_strategy.py      # V1 — 三维度评分：宏观40%+量化40%+周期20%
│   ├── zeping_strategy_v2.py   # V2 — (中间版本)
│   ├── zeping_strategy_v5.py   # V5 — 修复科技赛道映射，覆盖率修复
│   ├── zeping_strategy_v6.py   # V6 — 科技内部智能轮换
│   ├── zeping_strategy_v7.py   # V7 — 动态科技/周期配比 + 大宗轮动
│   ├── zeping_strategy_v8.py   # V8 — 连续得分融合+双时间尺度+换手阻尼+回撤制动（最新）
│   └── zeping_v11_engine.py    # V1.1 — V1 + 独立反转保护模块
│
├── ml/                 # ML 板块预测系统
│   ├── features.py       # FeatureEngineer — 20+量化特征+宏观标签+市场环境特征
│   ├── model.py          # HotBoardModel — LightGBM 分类器，预测次日热门板块
│   ├── backtest.py       # ModelBacktester — 滚动窗口回测评估
│   ├── data_collector.py # BoardHistoryCollector / SnapshotCollector — 数据采集
│   └── registry.py       # ModelRegistry — 模型版本管理（保存/加载/对比/删除）
│
├── catalyst/           # 催化剂追踪
│   └── __init__.py       # CatalystTracker — 财报日历、宏观政策事件、产业催化剂
│
├── checklist/          # 投资决策检查清单
│   └── __init__.py       # InvestmentChecklist — 五问决策框架、宏观评估
│
└── utils/              # 工具函数（当前为空）

streamlit_app/          # Streamlit Dashboard（17个标签页）
├── app.py              # 主入口 — 标签页路由 + 策略回测交互
├── pages_macro.py      # 宏观分析页（分类/周期/科技/红线/攻防/泽平策略）
├── pages_sector.py     # 板块监控页（行情/预测/个股推荐）
├── pages_ml.py         # ML预测页（模型训练/预测/回测/版本管理/联动分析）
└── pages_catalyst.py   # 催化剂追踪 + 投资检查清单

tests/                  # 单元测试
├── test_engine.py      # 回测引擎测试
├── test_strategies.py  # 策略信号生成测试
├── test_indicators.py  # 技术指标计算测试
├── test_macro.py       # 宏观分类/策略测试
├── test_ml.py          # ML 特征/模型测试
└── test_sector.py      # 板块监控测试
```

## 核心架构说明

### 1. 回测引擎 (`engine/backtest.py`)

事件驱动架构，逐日遍历行情数据：
- 策略 `generate_signals(data)` 一次性生成全部信号（BUY=1/SELL=-1/HOLD=0）
- 引擎逐日消费信号，执行买卖（含滑点、佣金、印花税、T+1限制）
- 单标的单仓位模式（全仓买入/全仓卖出）
- 输出 `BacktestResult`（净值曲线、持仓、现金、交易记录）

### 2. 泽平宏观策略演进链

策略核心是对板块级别的**次日热门板块预测**（非个股选股）：

```
V1（基础版）→ V5（赛道映射修复）→ V6（科技内部轮换）→ V7（硬性配比）→ V8（连续融合，最新）
                                                                         ↗
V1 → V1.1（反转保护模块）
```

**V1 三维度评分**：
- 宏观维度(40%): 科技70基础分/周期55分/其他40分 + 赛道热度加成
- 量化维度(40%): 1日动量 + 换手率 + 上涨广度 + 领涨强度
- 周期维度(20%): 五段论阶段匹配 + 重点布局区(阶段四)加分

**V8 核心创新**（解决V7硬性配比的cliff效应）：
- 连续制度得分：不硬性分配科技/周期数量，用连续调整分融合
- 双时间尺度确认：3天短期 + 7天长期，方向一致才采信完整信号
- 换手阻尼：昨日持仓板块+2分惯性加成，自然降低换手
- 滚动表现过滤：近5日表现差的板块扣分
- 回撤制动：累计超额<-10%时回退到纯V5评分

### 3. ML 系统

LightGBM 二分类器预测"次日是否为热门板块"（Top 20%涨幅=热门）：
- 20个量化特征 + 4个宏观标签 + 4个市场环境特征
- 时序交叉验证（TimeSeriesSplit 3折）
- 滚动窗口回测（默认30天训练窗口，每5天重训练）
- 模型版本管理（~/.real_ai_r/models/ 下 joblib+JSON元数据）

### 4. 板块分类体系

基于关键词匹配将A股板块分为四类：
- **科技主线** (tech): 9个赛道（芯片/AI/机器人/AI医疗/航天/新能源车/清洁能源/通信/消费电子）
- **周期主线** (cycle): 5段论（贵金属→基本金属→传统能源→农业后周期→必选消费）
- **红线禁区** (redline): 房地产链/传统白酒/传统零售/旧软件外包 → 直接排除
- **其他** (neutral): 未匹配的板块

## 数据流

```
AKShare API
  ↓
DataFetcher / SectorMonitor / BoardHistoryCollector
  ↓
SectorClassifier（板块打标签）→ RedLineFilter（过滤红线）
  ↓                              ↓
FeatureEngineer（特征工程）     ZepingMacroStrategy V1-V8（规则评分）
  ↓                              ↓
HotBoardModel（ML预测）        ZepingBoardScore（排名推荐）
  ↓                              ↓
Streamlit Dashboard ←←←←←←←←←←←←←
```

## 关键约定

- **列名标准化**: AKShare返回中文列名（日期/开盘/收盘），`_standardize_daily()` 和 `col_map` 统一转为英文（date/open/close）
- **板块数据列**: `name, code, change_pct, turnover_rate, rise_count, fall_count, lead_stock, lead_stock_pct`
- **信号约定**: Signal.BUY=1, Signal.SELL=-1, Signal.HOLD=0
- **日期格式**: YYYYMMDD（AKShare接口）或 YYYY-MM-DD（内部使用 pd.Timestamp）
- **缓存**: 个股日线数据缓存在 `data_cache/` 目录下的CSV文件
- **模型存储**: `~/.real_ai_r/models/` (ModelRegistry) 和 `~/.real_ai_r/em_history.db` (EMHistoryDB)
- **策略版本**: V8 是当前最新的泽平宏观策略，基于 V5 评分框架 + 连续融合引擎

## 开发注意事项

- 所有策略继承 `BaseStrategy`，实现 `generate_signals(data) -> pd.Series`
- 回测引擎使用 `close` 价执行交易，买入加滑点，卖出减滑点
- `SectorClassifier` 默认使用 `TECH_TRACKS`（V1版6赛道），V5+策略显式传入 `TECH_TRACKS_V5`（9赛道）
- ML特征需要至少60天历史数据（`FeatureEngineer.MIN_HISTORY_DAYS = 60`）
- AKShare 有限流保护，批量请求间需加延迟（`BoardHistoryCollector.REQUEST_DELAY = 0.3s`）
- Ruff 配置: `target-version = "py310"`, `line-length = 100`, 只检查 E/F/I/W
- 测试不依赖网络（使用模拟数据），直接 `pytest tests/` 即可
