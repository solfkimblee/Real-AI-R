"""回测引擎 — 事件驱动架构

支持：
- A股 T+1 交易规则
- 手续费（佣金 + 印花税）
- 滑点模拟
- 仓位管理
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from real_ai_r.strategies.base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """单笔交易记录。"""

    date: pd.Timestamp
    direction: str  # "BUY" | "SELL"
    price: float
    shares: int
    commission: float
    amount: float  # price * shares


@dataclass
class BacktestResult:
    """回测结果。"""

    # 基础信息
    strategy_name: str
    initial_capital: float
    final_capital: float

    # 时间序列
    portfolio_values: pd.Series  # 每日组合净值
    positions: pd.Series  # 每日持仓数量
    cash_series: pd.Series  # 每日现金

    # 交易记录
    trades: list[Trade] = field(default_factory=list)

    # 绩效指标（由 analysis 模块计算后填充）
    metrics: dict = field(default_factory=dict)

    def summary(self) -> str:
        """返回回测结果摘要字符串。"""
        total_return = (self.final_capital - self.initial_capital) / self.initial_capital
        lines = [
            f"策略: {self.strategy_name}",
            f"初始资金: {self.initial_capital:,.2f}",
            f"最终资金: {self.final_capital:,.2f}",
            f"总收益率: {total_return:.2%}",
            f"交易次数: {len(self.trades)}",
        ]
        if self.metrics:
            for k, v in self.metrics.items():
                if isinstance(v, float):
                    lines.append(f"{k}: {v:.4f}")
                else:
                    lines.append(f"{k}: {v}")
        return "\n".join(lines)


class BacktestEngine:
    """回测引擎。

    Parameters
    ----------
    data : pd.DataFrame
        标准化行情数据（需包含 date, open, high, low, close, volume 列）。
    strategy : BaseStrategy
        交易策略实例。
    initial_capital : float
        初始资金，默认 100,000。
    commission_rate : float
        佣金费率（单边），默认万三 (0.0003)。
    stamp_tax_rate : float
        印花税率（卖出时收取），默认千一 (0.001)。
    slippage : float
        滑点比例，默认 0.001 (0.1%)。
    min_commission : float
        最低佣金，默认 5 元。
    trade_unit : int
        最小交易单位（手），A股1手=100股。
    max_position_pct : float
        最大仓位比例，默认 0.95（留5%现金缓冲）。
    """

    def __init__(
        self,
        data: pd.DataFrame,
        strategy: BaseStrategy,
        initial_capital: float = 100_000.0,
        commission_rate: float = 0.0003,
        stamp_tax_rate: float = 0.001,
        slippage: float = 0.001,
        min_commission: float = 5.0,
        trade_unit: int = 100,
        max_position_pct: float = 0.95,
    ) -> None:
        self.data = data.copy().reset_index(drop=True)
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.stamp_tax_rate = stamp_tax_rate
        self.slippage = slippage
        self.min_commission = min_commission
        self.trade_unit = trade_unit
        self.max_position_pct = max_position_pct

        # 状态
        self._cash = initial_capital
        self._position = 0  # 持仓股数
        self._trades: list[Trade] = []
        self._can_sell = False  # T+1: 当日买入次日才能卖出

    def run(self) -> BacktestResult:
        """执行回测。

        Returns
        -------
        BacktestResult
            回测结果，包含净值曲线、交易记录等。
        """
        data = self.data
        signals = self.strategy.generate_signals(data)

        n = len(data)
        portfolio_values = np.zeros(n)
        positions = np.zeros(n, dtype=int)
        cash_arr = np.zeros(n)

        for i in range(n):
            price = data.loc[i, "close"]
            signal = signals.iloc[i]

            # T+1: 如果昨天买入，今天才可卖出
            if i > 0 and positions[i - 1] > 0 and not self._can_sell:
                self._can_sell = True

            # 处理信号
            if signal == Signal.BUY.value and self._position == 0:
                self._execute_buy(data.loc[i])
            elif signal == Signal.SELL.value and self._position > 0 and self._can_sell:
                self._execute_sell(data.loc[i])

            # 记录每日状态
            positions[i] = self._position
            cash_arr[i] = self._cash
            portfolio_values[i] = self._cash + self._position * price

        dates = data["date"]
        result = BacktestResult(
            strategy_name=self.strategy.name,
            initial_capital=self.initial_capital,
            final_capital=portfolio_values[-1] if n > 0 else self.initial_capital,
            portfolio_values=pd.Series(portfolio_values, index=dates),
            positions=pd.Series(positions, index=dates),
            cash_series=pd.Series(cash_arr, index=dates),
            trades=self._trades,
        )

        logger.info(
            "回测完成: %s | 收益: %.2f%% | 交易: %d笔",
            self.strategy.name,
            (result.final_capital / self.initial_capital - 1) * 100,
            len(self._trades),
        )

        return result

    # ------------------------------------------------------------------
    # 交易执行
    # ------------------------------------------------------------------

    def _execute_buy(self, bar: pd.Series) -> None:
        """执行买入。"""
        price = bar["close"]
        # 加滑点
        buy_price = price * (1 + self.slippage)

        # 计算最大可买手数
        available = self._cash * self.max_position_pct
        max_shares = int(available / (buy_price * self.trade_unit)) * self.trade_unit

        if max_shares <= 0:
            return

        # 计算佣金
        amount = buy_price * max_shares
        commission = max(amount * self.commission_rate, self.min_commission)
        total_cost = amount + commission

        if total_cost > self._cash:
            # 减少一手
            max_shares -= self.trade_unit
            if max_shares <= 0:
                return
            amount = buy_price * max_shares
            commission = max(amount * self.commission_rate, self.min_commission)
            total_cost = amount + commission

        self._cash -= total_cost
        self._position = max_shares
        self._can_sell = False  # T+1: 当天买入不能卖

        self._trades.append(
            Trade(
                date=bar["date"],
                direction="BUY",
                price=buy_price,
                shares=max_shares,
                commission=commission,
                amount=amount,
            )
        )

    def _execute_sell(self, bar: pd.Series) -> None:
        """执行卖出。"""
        if self._position <= 0:
            return

        price = bar["close"]
        # 减滑点
        sell_price = price * (1 - self.slippage)

        amount = sell_price * self._position
        commission = max(amount * self.commission_rate, self.min_commission)
        stamp_tax = amount * self.stamp_tax_rate
        net_amount = amount - commission - stamp_tax

        self._cash += net_amount

        self._trades.append(
            Trade(
                date=bar["date"],
                direction="SELL",
                price=sell_price,
                shares=self._position,
                commission=commission + stamp_tax,
                amount=amount,
            )
        )

        self._position = 0
        self._can_sell = False
