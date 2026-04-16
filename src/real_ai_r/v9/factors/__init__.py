"""V9 原子因子库。

每个因子是一个 `Factor` 对象，输入 board_df + context，输出
每个板块的 zscore 标准化截面分数（pd.Series, index=board_name）。

所有内置因子在 import 时自动注册到全局 registry。
"""

from real_ai_r.v9.factors.base import Factor, FactorContext, FactorRegistry, registry
from real_ai_r.v9.factors import breadth, event, fund_flow, price  # noqa: F401 (register)

__all__ = [
    "Factor",
    "FactorContext",
    "FactorRegistry",
    "registry",
]
