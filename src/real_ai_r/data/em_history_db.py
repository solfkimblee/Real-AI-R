"""EM历史自建库 — 每日盘后自动采集496板块快照

设计理念：
    EM的496个细分板块实时API可用，但历史API被限流。
    解决方案：每日盘后自动采集实时快照存入SQLite，积累自己的历史库。
    10-20个交易日后即可支撑V2多因子回测。

数据库Schema：
    board_daily — 板块日线快照
    collection_log — 采集日志

使用方法：
    # 采集当日数据
    db = EMHistoryDB()
    db.collect_today()

    # 查询历史
    df = db.get_board_history("快递", days=20)

    # 获取某日全市场快照
    snapshot = db.get_market_snapshot("2026-04-15")
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# 默认数据库路径
DEFAULT_DB_PATH = Path.home() / ".real_ai_r" / "em_history.db"


class EMHistoryDB:
    """EM板块历史自建数据库。

    每日盘后调用 collect_today() 自动采集496个板块的实时快照，
    存入本地SQLite数据库，逐步积累历史数据。
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # 数据库初始化
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """创建数据库表（如不存在）。"""
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS board_daily (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,              -- 交易日 YYYY-MM-DD
                    board_code TEXT NOT NULL,         -- 板块代码 (如 BK1489)
                    board_name TEXT NOT NULL,         -- 板块名称 (如 快递)
                    close_price REAL,                -- 收盘价/最新价
                    change_amount REAL,              -- 涨跌额
                    change_pct REAL,                 -- 涨跌幅%
                    total_market_cap REAL,           -- 总市值
                    turnover_rate REAL,              -- 换手率%
                    rise_count INTEGER,              -- 上涨家数
                    fall_count INTEGER,              -- 下跌家数
                    lead_stock TEXT,                 -- 领涨股名称
                    lead_stock_change REAL,          -- 领涨股涨幅%
                    collected_at TEXT NOT NULL,       -- 采集时间 ISO格式
                    UNIQUE(date, board_code)          -- 每板块每天一条
                );

                CREATE INDEX IF NOT EXISTS idx_board_daily_date
                    ON board_daily(date);
                CREATE INDEX IF NOT EXISTS idx_board_daily_name
                    ON board_daily(board_name);
                CREATE INDEX IF NOT EXISTS idx_board_daily_code
                    ON board_daily(board_code);

                CREATE TABLE IF NOT EXISTS collection_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    board_type TEXT NOT NULL,         -- 'industry' or 'concept'
                    total_boards INTEGER,
                    success_count INTEGER,
                    failed_count INTEGER,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    status TEXT DEFAULT 'running',    -- running/success/failed
                    error_message TEXT
                );
            """)

    def _conn(self) -> sqlite3.Connection:
        """获取数据库连接。"""
        return sqlite3.connect(str(self.db_path))

    # ------------------------------------------------------------------
    # 数据采集
    # ------------------------------------------------------------------

    def collect_today(
        self,
        board_type: str = "industry",
        trade_date: str | None = None,
    ) -> dict:
        """采集当日EM板块数据并存入数据库。

        Parameters
        ----------
        board_type : str
            板块类型: "industry" (行业, 496个) 或 "concept" (概念, 493个)
        trade_date : str | None
            交易日期 (YYYY-MM-DD)。默认为今天。

        Returns
        -------
        dict
            采集结果: {total, success, failed, date}
        """
        import akshare as ak

        if trade_date is None:
            trade_date = datetime.now().strftime("%Y-%m-%d")

        started_at = datetime.now().isoformat()
        log_id = self._start_log(trade_date, board_type, started_at)

        try:
            # 获取实时数据
            if board_type == "industry":
                df = ak.stock_board_industry_name_em()
            else:
                df = ak.stock_board_concept_name_em()

            total = len(df)
            success = 0
            failed = 0

            # 列名映射
            col_map = {
                "排名": "rank", "板块名称": "board_name", "板块代码": "board_code",
                "最新价": "close_price", "涨跌额": "change_amount",
                "涨跌幅": "change_pct", "总市值": "total_market_cap",
                "换手率": "turnover_rate", "上涨家数": "rise_count",
                "下跌家数": "fall_count", "领涨股票": "lead_stock",
                "领涨股票-涨跌幅": "lead_stock_change",
            }
            df = df.rename(columns=col_map)

            collected_at = datetime.now().isoformat()

            with self._conn() as conn:
                for _, row in df.iterrows():
                    try:
                        conn.execute("""
                            INSERT OR REPLACE INTO board_daily
                            (date, board_code, board_name, close_price,
                             change_amount, change_pct, total_market_cap,
                             turnover_rate, rise_count, fall_count,
                             lead_stock, lead_stock_change, collected_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            trade_date,
                            str(row.get("board_code", "")),
                            str(row.get("board_name", "")),
                            _safe_float(row.get("close_price")),
                            _safe_float(row.get("change_amount")),
                            _safe_float(row.get("change_pct")),
                            _safe_float(row.get("total_market_cap")),
                            _safe_float(row.get("turnover_rate")),
                            _safe_int(row.get("rise_count")),
                            _safe_int(row.get("fall_count")),
                            str(row.get("lead_stock", "")),
                            _safe_float(row.get("lead_stock_change")),
                            collected_at,
                        ))
                        success += 1
                    except Exception as e:
                        failed += 1
                        logger.warning("板块 %s 写入失败: %s", row.get("board_name"), e)

            self._finish_log(log_id, total, success, failed, "success")
            result = {
                "date": trade_date, "total": total,
                "success": success, "failed": failed,
            }
            logger.info("EM采集完成: %s", result)
            return result

        except Exception as e:
            self._finish_log(log_id, 0, 0, 0, "failed", str(e))
            logger.error("EM采集失败: %s", e)
            return {"date": trade_date, "total": 0, "success": 0, "failed": 0, "error": str(e)}

    # ------------------------------------------------------------------
    # 数据查询
    # ------------------------------------------------------------------

    def get_board_history(
        self,
        board_name: str,
        days: int = 20,
    ) -> pd.DataFrame:
        """获取某板块的历史数据。"""
        with self._conn() as conn:
            df = pd.read_sql_query("""
                SELECT date, board_code, board_name, close_price,
                       change_pct, turnover_rate, rise_count, fall_count,
                       lead_stock, lead_stock_change, total_market_cap
                FROM board_daily
                WHERE board_name = ?
                ORDER BY date DESC
                LIMIT ?
            """, conn, params=(board_name, days))
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
        return df

    def get_market_snapshot(
        self,
        date: str,
    ) -> pd.DataFrame:
        """获取某日全市场板块快照。"""
        with self._conn() as conn:
            df = pd.read_sql_query("""
                SELECT date, board_code, board_name, close_price,
                       change_pct, turnover_rate, rise_count, fall_count,
                       lead_stock, lead_stock_change, total_market_cap
                FROM board_daily
                WHERE date = ?
                ORDER BY change_pct DESC
            """, conn, params=(date,))
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return df

    def get_available_dates(self) -> list[str]:
        """获取已采集的所有交易日列表。"""
        with self._conn() as conn:
            cursor = conn.execute("""
                SELECT DISTINCT date FROM board_daily ORDER BY date
            """)
            return [row[0] for row in cursor.fetchall()]

    def get_board_count(self, date: str | None = None) -> int:
        """获取板块数量。"""
        with self._conn() as conn:
            if date:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM board_daily WHERE date = ?", (date,)
                )
            else:
                cursor = conn.execute(
                    "SELECT COUNT(DISTINCT board_name) FROM board_daily"
                )
            return cursor.fetchone()[0]

    def get_collection_logs(self, limit: int = 10) -> pd.DataFrame:
        """获取采集日志。"""
        with self._conn() as conn:
            return pd.read_sql_query("""
                SELECT * FROM collection_log
                ORDER BY id DESC LIMIT ?
            """, conn, params=(limit,))

    def get_market_avg_history(self, days: int = 30) -> pd.DataFrame:
        """获取市场平均涨跌幅历史（用于反转保护状态检测）。"""
        with self._conn() as conn:
            df = pd.read_sql_query("""
                SELECT date, AVG(change_pct) as market_avg,
                       COUNT(*) as board_count
                FROM board_daily
                GROUP BY date
                ORDER BY date DESC
                LIMIT ?
            """, conn, params=(days,))
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # 日志管理
    # ------------------------------------------------------------------

    def _start_log(self, date: str, board_type: str, started_at: str) -> int:
        with self._conn() as conn:
            cursor = conn.execute("""
                INSERT INTO collection_log (date, board_type, total_boards,
                    success_count, failed_count, started_at, status)
                VALUES (?, ?, 0, 0, 0, ?, 'running')
            """, (date, board_type, started_at))
            return cursor.lastrowid or 0

    def _finish_log(
        self, log_id: int, total: int, success: int,
        failed: int, status: str, error: str | None = None,
    ) -> None:
        with self._conn() as conn:
            conn.execute("""
                UPDATE collection_log
                SET total_boards = ?, success_count = ?, failed_count = ?,
                    finished_at = ?, status = ?, error_message = ?
                WHERE id = ?
            """, (total, success, failed, datetime.now().isoformat(),
                  status, error, log_id))


# ======================================================================
# 盘后自动采集脚本入口
# ======================================================================

def run_daily_collection(db_path: str | None = None) -> None:
    """盘后自动采集入口（可用cron定时调用）。

    Usage:
        python -m real_ai_r.data.em_history_db

    Or via cron (每个交易日16:00执行):
        0 16 * * 1-5 cd /path/to/repo && python -m real_ai_r.data.em_history_db
    """
    db = EMHistoryDB(db_path=db_path)

    print(f"{'=' * 60}")
    print(f"  EM板块历史数据采集 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  数据库: {db.db_path}")
    print(f"{'=' * 60}")

    # 采集行业板块
    print("\n[1/2] 采集行业板块...")
    result1 = db.collect_today(board_type="industry")
    print(f"  行业板块: 成功={result1['success']}, 失败={result1['failed']}")

    # 采集概念板块（可选）
    print("\n[2/2] 采集概念板块...")
    result2 = db.collect_today(board_type="concept")
    print(f"  概念板块: 成功={result2['success']}, 失败={result2['failed']}")

    # 统计
    dates = db.get_available_dates()
    print(f"\n{'=' * 60}")
    print(f"  采集完成!")
    print(f"  已积累 {len(dates)} 个交易日数据")
    print(f"  最早日期: {dates[0] if dates else 'N/A'}")
    print(f"  最新日期: {dates[-1] if dates else 'N/A'}")
    print(f"{'=' * 60}")


# ======================================================================
# 工具函数
# ======================================================================

def _safe_float(val) -> float:
    if val is None:
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _safe_int(val) -> int:
    if val is None:
        return 0
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0


if __name__ == "__main__":
    run_daily_collection()
