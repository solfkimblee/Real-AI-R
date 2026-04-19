"""策略运行器 — 把 V5/V7/V8/V9.2/V9.3/V10/V11/V12/V12a/V12b/V12c/V13/V14 统一到同一个 predict 接口。

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
        strategies: ['V5','V7','V8','V9.2','V9.3','V10','V11','V12',
                     'V12a','V12b','V12c','V13','V14']，None=全部（不含消融变体）
        warmup_panel: V9.3/V11/V12/V13/V14 需要；若 None，这些策略跳过
    """
    chosen = strategies or ["V5", "V7", "V8", "V9.2", "V9.3", "V10", "V11", "V12", "V13", "V14"]
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

    if "V13" in chosen and warmup_panel is not None and len(warmup_panel) > 0:
        from real_ai_r.macro.zeping_strategy_v13_lgbm import ZepingLGBMStrategyV13
        v13 = ZepingLGBMStrategyV13()
        try:
            v13.fit(warmup_panel)
            runners.append(ZepingRunner("V13(LGB+ZP)", v13))
        except Exception as e:
            print(f"[warn] V13 LGB+ZP training failed: {e}; skipping V13")

    if "V14" in chosen and warmup_panel is not None and len(warmup_panel) > 0:
        from real_ai_r.macro.zeping_strategy_v14_lgbm import ZepingLGBMStrategyV14
        v14 = ZepingLGBMStrategyV14()
        try:
            v14.fit(warmup_panel)
            runners.append(ZepingRunner("V14(LGB+ZP+Live)", v14))
        except Exception as e:
            print(f"[warn] V14 LGB+ZP+Live training failed: {e}; skipping V14")

    # --- V12 消融变体 ---
    # Note: All V12 variants include linkage features (V12 always builds them).
    #   retrain_every=999999 effectively disables rolling retrain.
    #   horizons=(1,) uses single horizon (no multi-horizon).
    #   V12a = retrain + linkage (no multi-horizon)
    #   V12b = multi-horizon + linkage (no retrain)
    #   V12c = linkage only (single horizon, no retrain)
    if "V12a" in chosen and warmup_panel is not None and len(warmup_panel) > 0:
        try:
            from real_ai_r.macro.zeping_strategy_v12_lgbm import ZepingLGBMStrategyV12
            # V12a: 滚动再训练 + 联动特征（单horizon, 无多horizon融合）
            v12a = ZepingLGBMStrategyV12(
                horizons=(1,), horizon_weights=(1.0,),
                retrain_every=20, retrain_window=250,
            )
            v12a.fit(warmup_panel)
            runners.append(ZepingRunner("V12a(Retrain)", v12a))
        except Exception as e:
            print(f"[warn] V12a training failed: {e}; skipping V12a")

    if "V12b" in chosen and warmup_panel is not None and len(warmup_panel) > 0:
        try:
            from real_ai_r.macro.zeping_strategy_v12_lgbm import ZepingLGBMStrategyV12
            # V12b: 多horizon融合 + 联动特征（无滚动再训练）
            v12b = ZepingLGBMStrategyV12(
                horizons=(1, 3, 5), horizon_weights=(0.5, 0.3, 0.2),
                retrain_every=999999,
            )
            v12b.fit(warmup_panel)
            runners.append(ZepingRunner("V12b(MultiH)", v12b))
        except Exception as e:
            print(f"[warn] V12b training failed: {e}; skipping V12b")

    if "V12c" in chosen and warmup_panel is not None and len(warmup_panel) > 0:
        try:
            from real_ai_r.macro.zeping_strategy_v12_lgbm import ZepingLGBMStrategyV12
            # V12c: 联动特征 only（单horizon, 无滚动再训练）
            v12c = ZepingLGBMStrategyV12(
                horizons=(1,), horizon_weights=(1.0,),
                retrain_every=999999,
            )
            v12c.fit(warmup_panel)
            runners.append(ZepingRunner("V12c(Linkage)", v12c))
        except Exception as e:
            print(f"[warn] V12c training failed: {e}; skipping V12c")

    return runners
