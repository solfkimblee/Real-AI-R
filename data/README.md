# 回测数据目录

本目录存放真实回测数据文件。**不提交到 main 分支 — 推 feature 分支即可。**

## 期望文件

```
data/
├── panel_train.parquet      # 训练集：~2023-01 ~ 2024-06 (400 天左右)
├── panel_val.parquet        # 验证集：~2024-07 ~ 2024-09 (60 天)
├── panel_test.parquet       # 测试集：~2024-10 ~ 2025-03 (149 天) — 最终才跑一次
└── benchmark.csv            # 可选：基准 (如上证指数)
```

**为什么 train/val/test 三分？**
- `train`：Claude 用来做参数扫描、调敏感性矩阵、扩关键词
- `val`：Claude **只看最终数字**不做调参。Claude 说"val Sharpe 多少"，你决定是否进 test
- `test`：**锁死**。只跑一次，结果就是定论。防止 p-hacking。

## panel_*.parquet 列规范

**必需列:**
| 列名 | 类型 | 单位 | 说明 |
|------|------|------|------|
| `date` | datetime | - | 交易日 |
| `name` | string | - | 板块名（如"半导体"）|
| `change_pct` | float | % | 当日涨跌幅（如 1.5 表示 +1.5%）|

**可选列（给越多，V10/V8 信号越丰富）：**
| 列名 | 类型 | 单位 | 备注 |
|------|------|------|------|
| `code` | string | - | 板块代码 |
| `turnover_rate` | float | % | 当日换手率 |
| `rise_count` | int | 个 | 板块内上涨家数 |
| `fall_count` | int | 个 | 板块内下跌家数 |
| `lead_stock_pct` | float | % | 领涨股涨幅 |
| `momentum_5d` | float | % | 5 日累计涨幅 |
| `main_net_inflow` | float | 亿 | 主力净流入额 |

**列名不一致没关系**：`scripts/wf_backtest.py` 支持 `--column-map` 重命名。例如：
```
python scripts/wf_backtest.py --column-map "涨跌幅:change_pct,换手:turnover_rate"
```

## benchmark.csv 格式

```csv
date,change_pct
2024-01-02,0.45
2024-01-03,-0.12
...
```

若不提供，`wf_backtest.py` 会用"全板块等权平均"作为基准。

## 文件大小参考

- 500 板块 × 400 天 × 10 列 parquet ≈ 5-10 MB（无压力）
- 如超过 50 MB 考虑 git-lfs，或直接 gzip

## 数据隐私

- 本项目只需**公开市场数据**（板块行情、指数）
- 不要提交：账户信息、持仓、私有研报
