"""泽平宏观 V9 — 在线学习量化框架。

核心创新（全面超越 V8）：
1. 原子因子 + 在线 IC 动态加权（替代静态 40/40/20）
2. Gaussian HMM 隐制度识别（替代 regime_score 硬算）
3. 板块相关性图传播（捕捉主题联动）
4. QP 组合优化（替代 Top-N 截断）
5. Hedge 元学习集成（多版本软融合防过拟合）
6. 走步回测框架（样本外验证）

典型用法:
    from real_ai_r.v9 import V9Engine, V9BacktestRunner

    engine = V9Engine()
    result = engine.predict(board_df=..., history=...)

    runner = V9BacktestRunner(engine=engine)
    perf = runner.run(history_df=..., start="2024-01-01", end="2024-12-31")
"""

from real_ai_r.v9.engine import V9Config, V9Engine, V9Prediction
from real_ai_r.v9.backtest import V9BacktestResult, V9BacktestRunner

__all__ = [
    "V9Config",
    "V9Engine",
    "V9Prediction",
    "V9BacktestResult",
    "V9BacktestRunner",
]
