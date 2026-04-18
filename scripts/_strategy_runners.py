"""策略运行器 — 把 V5/V7/V8/V9.2/V9.3/V10/V11(LGB) 统一到同一个 predict 接口。

每个 runner 是一个 callable:
    runner.predict(board_df, top_n, tech_history, cycle_history) -> list[board_name]
    runner.record_day(realized_returns, excess_per_board, daily_excess) -> None
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import pandas as pd


class StrategyRunner(Protocol):
    name: str

    def predict(
        self,
        board_df: pd.DataFrame,
        top_n: int,
        tech_history: list[float],
        cycle_history: list[float],
    ) -> list[str]: ...

    def record_day(
        self,
        realized_returns: dict[str, float],
        excess_per_board: dict[str, float],
        daily_excess: float,
    ) -> None: ...


@dataclass
class ZepingRunner:
    """V5 / V7 / V8 / V10 — ZepingMacroStrategy 家族通用 runner。"""

    name: str
    strategy: Any  # ZepingMacroStrategy-like

    def predict(self, board_df, top_n, tech_history, cycle_history):
        # V5 基类 predict() 不接受 tech_history/cycle_history，
        # V7/V8/V10 重写了 predict() 才接受这些参数。
        import inspect
        sig = inspect.signature(self.strategy.predict)
        if "tech_history" in sig.parameters:
            result = self.strategy.predict(
                board_df=board_df,
                top_n=top_n,
                tech_history=tech_history,
                cycle_history=cycle_history,
            )
        else:
            result = self.strategy.predict(
                board_df=board_df,
                top_n=top_n,
            )
        return [p.board_name for p in result.predictions]

    def record_day(self, realized_returns, excess_per_board, daily_excess):
        if hasattr(self.strategy, "record_excess"):
            try:
                self.strategy.record_excess(daily_excess)
            except Exception:
                pass
        if hasattr(self.strategy, "record_board_performance"):
            for b, e in excess_per_board.items():
                try:
                    self.strategy.record_board_performance(b, e)
                except Exception:
                    pass


@dataclass
class V92Runner:
    """V9.2 Hedge 集成专用 runner（额外调 observe_realized_returns）。"""

    name: str
    strategy: Any

    def predict(self, board_df, top_n, tech_history, cycle_history):
        result = self.strategy.predict(
            board_df=board_df,
            top_n=top_n,
            tech_history=tech_history,
            cycle_history=cycle_history,
        )
        return [p.board_name for p in result.predictions]

    def record_day(self, realized_returns, excess_per_board, daily_excess):
        # V9.2 核心驱动 - 每日 observe
        try:
            self.strategy.observe_realized_returns(realized_returns)
        except Exception:
            pass
        # V8 回调也转发
        try:
            self.strategy.record_excess(daily_excess)
        except Exception:
            pass


@dataclass
class V93Runner:
    """V9.3 Warm-Start — 需要 fit_warmup 预热。"""

    name: str
    strategy: Any
    warmed: bool = False

    def predict(self, board_df, top_n, tech_history, cycle_history):
        result = self.strategy.predict(
            board_df=board_df,
            top_n=top_n,
            tech_history=tech_history,
            cycle_history=cycle_history,
        )
        return [p.board_name for p in result.predictions]

    def record_day(self, realized_returns, excess_per_board, daily_excess):
        try:
            self.strategy.observe_realized_returns(realized_returns)
        except Exception:
            pass


def build_runners(
    strategies: list[str] | None = None,
    warmup_panel: pd.DataFrame | None = None,
) -> list[StrategyRunner]:
    """实例化各策略并包装为统一 runner。

    参数:
        strategies: ['V5','V7','V8','V9.2','V9.3','V10','V11','V12']，None=全部
        warmup_panel: V9.3/V11/V12 需要；若 None，这些策略跳过
    """
    chosen = strategies or ["V5", "V7", "V8", "V9.2", "V9.3", "V10", "V11", "V12"]
    runners: list[StrategyRunner] = []

    if "V5" in chosen:
        from real_ai_r.macro.zeping_strategy_v5 import ZepingMacroStrategyV5
        runners.append(ZepingRunner("V5", ZepingMacroStrategyV5()))

    if "V7" in chosen:
        from real_ai_r.macro.zeping_strategy_v7 import ZepingMacroStrategyV7
        runners.append(ZepingRunner("V7", ZepingMacroStrategyV7()))

    if "V8" in chosen:
        from real_ai_r.macro.zeping_strategy_v8 import ZepingMacroStrategyV8
        runners.append(ZepingRunner("V8", ZepingMacroStrategyV8()))

    if "V9.2" in chosen:
        from real_ai_r.macro.meta_ensemble_v92 import (
            MetaEnsembleStrategyV92,
            V92Params,
        )
        v92 = MetaEnsembleStrategyV92(
            v92_params=V92Params(
                hedge_eta=1.5, hedge_warmup=5, hedge_floor=0.05,
            ),
        )
        runners.append(V92Runner("V9.2", v92))

    if "V9.3" in chosen and warmup_panel is not None and len(warmup_panel) > 0:
        from real_ai_r.macro.v9_3_warmstart import V9_3Strategy
        v93 = V9_3Strategy()
        try:
            v93.fit_warmup(warmup_panel)
            runners.append(V93Runner("V9.3", v93, warmed=True))
        except Exception as e:
            print(f"[warn] V9.3 warmup failed: {e}; skipping V9.3")

    if "V10" in chosen:
        from real_ai_r.macro.zeping_strategy_v10 import ZepingMacroStrategyV10
        runners.append(ZepingRunner("V10", ZepingMacroStrategyV10()))

    if "V11" in chosen and warmup_panel is not None and len(warmup_panel) > 0:
        from real_ai_r.macro.zeping_strategy_v11_lgbm import ZepingLGBMStrategy
        v11 = ZepingLGBMStrategy()
        try:
            v11.fit(warmup_panel)
            runners.append(ZepingRunner("V11(LGB)", v11))
        except Exception as e:
            print(f"[warn] V11 LGB training failed: {e}; skipping V11")

    if "V12" in chosen and warmup_panel is not None and len(warmup_panel) > 0:
        from real_ai_r.macro.zeping_strategy_v12_lgbm import ZepingLGBMStrategyV12
        v12 = ZepingLGBMStrategyV12()
        try:
            v12.fit(warmup_panel)
            runners.append(ZepingRunner("V12(LGB+)", v12))
        except Exception as e:
            print(f"[warn] V12 LGB+ training failed: {e}; skipping V12")

    return runners
