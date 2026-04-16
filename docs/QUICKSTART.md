# 5 分钟快速入门

让一个没用过本项目的人在 5 分钟内完成首次真实回测。

---

## 第 1 步（30 秒）：确认环境

```bash
git clone https://github.com/solfkimblee/Real-AI-R.git
cd Real-AI-R
pip install -e ".[dev]"
```

---

## 第 2 步（30 秒）：跑 demo，验证一切正常

```bash
python scripts/wf_backtest.py --demo
```

**期望输出**：

```
[info] generating synthetic panel (140 days × 37 boards)...
[info] strategies: ['V5', 'V7', 'V8', 'V9.2', 'V10']
[run] V5 ...
[done] V5   Sharpe=+... 累计=+...%
...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 对比表
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 策略  │ Sharpe │  IR   │ 累计超额 │ MaxDD │ 胜率
 V7    │ +7.57  │ +7.79 │  +28.2% │ -2.7% │ 68.3%
 V8    │ +7.45  │ +7.67 │  +26.0% │ -2.6% │ 70.5%
 ...
```

看到对比表 → ✅ 环境 OK，继续。

---

## 第 3 步（2 分钟）：跑真实 A 股数据

**选项 A — 使用 akshare 现拉数据**：

```python
# scripts/quickstart_ak.py
from real_ai_r.data.fetcher import DataFetcher
from real_ai_r.engine.backtest import BacktestEngine
from real_ai_r.strategies.ma_cross import MACrossStrategy

data = DataFetcher().get_stock_daily(
    "000001", start_date="2023-01-01", end_date="2024-12-31",
)
result = BacktestEngine(
    data=data, strategy=MACrossStrategy(5, 20),
).run()
print(result.summary())
```

运行：
```bash
python scripts/quickstart_ak.py
```

**选项 B — 使用自己的 parquet/csv 数据**：

把数据按 [`data/README.md`](../data/README.md) 规范命名放到 `data/`，然后：

```bash
python scripts/wf_backtest.py \
    --panel data/your_panel.parquet \
    --strategies V8,V10 \
    --windows 7 --window-size 20
```

列名不对？用 `--column-map`：
```bash
--column-map "涨跌幅:change_pct,换手:turnover_rate,板块名:name"
```

---

## 第 4 步（1 分钟）：看可视化

```bash
streamlit run streamlit_app/app.py
```

浏览器打开 http://localhost:8501，交互式选择策略、股票、参数。

---

## 第 5 步（1 分钟）：挑一个策略

打开 [`docs/strategy-guide.md`](strategy-guide.md) 挑策略。**建议起步：V8** —— 在真实 WF 验证中 Sharpe 2.86，7/7 正窗口，稳定性最好。

```python
from real_ai_r.macro import ZepingMacroStrategyV8

strat = ZepingMacroStrategyV8()
result = strat.predict(
    board_df=today_board_df,  # 当日截面
    top_n=10,
    tech_history=[...],       # 过去 N 天科技板块日均涨
    cycle_history=[...],      # 过去 N 天周期板块日均涨
)
for bs in result.predictions:
    print(bs.board_name, bs.total_score)
```

---

## 常见问题

**Q: 我没有板块面板数据怎么办？**
A: 用 `scripts/gen_synthetic_panel.py` 生成合成数据做方法论验证，再上真实数据。

**Q: V5 在我的数据上返回空？**
A: V5 对红线板块过滤严格，如果你的板块名多数命中红线关键词，会被过滤光。用 V8 / V10 更鲁棒。

**Q: 怎么知道我该用哪个 V 版本？**
A: 看 [strategy-guide.md](strategy-guide.md) 的决策树。绝大多数场景用 V8。

**Q: V9.3 需要 warmup 是什么意思？**
A: V9.3 的在线学习需要先 "吃" 2+ 年历史数据预热，然后才能准确打分。CLI 用 `--warmup-panel` 传入历史数据。

---

## 下一步

- 📖 [strategy-guide.md](strategy-guide.md) — 策略对比与选型
- 📊 [data/README.md](../data/README.md) — 数据准备规范
- 🔬 运行 `python scripts/wf_backtest.py --help` 看所有选项
