"""模型版本管理与注册表

提供模型的持久化存储、版本管理、对比分析功能：
- 训练完自动保存模型（含元数据：时间、参数、指标、特征）
- 列出所有已保存模型版本
- 加载指定版本用于预测或回测
- 删除旧版本
- 对比多个版本的性能指标
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib

from real_ai_r.ml.model import HotBoardModel, ModelMetrics

logger = logging.getLogger(__name__)

# 默认模型存储目录
DEFAULT_REGISTRY_DIR = Path.home() / ".real_ai_r" / "models"


@dataclass
class ModelVersion:
    """模型版本元数据。"""

    version_id: str                          # 唯一版本号 (时间戳)
    created_at: str                          # 创建时间 ISO
    board_type: str                          # industry / concept
    train_days: int = 0                      # 训练数据天数
    max_boards: int = 0                      # 采集板块数
    sample_count: int = 0                    # 训练样本数
    feature_count: int = 0                   # 特征维度
    feature_columns: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)
    # 性能指标
    auc: float = 0.0
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    accuracy: float = 0.0
    # 可选备注
    note: str = ""

    @property
    def display_name(self) -> str:
        bt = "行业" if self.board_type == "industry" else "概念"
        return f"v{self.version_id} | {bt} | AUC={self.auc:.4f}"


class ModelRegistry:
    """模型注册表 — 管理所有已保存的模型版本。

    目录结构::

        registry_dir/
          ├── index.json          # 版本索引
          ├── v_20260413_143000/
          │   ├── model.joblib    # 模型文件
          │   └── meta.json       # 元数据
          └── v_20260413_150000/
              ├── model.joblib
              └── meta.json
    """

    def __init__(self, registry_dir: str | Path | None = None) -> None:
        self.registry_dir = Path(registry_dir) if registry_dir else DEFAULT_REGISTRY_DIR
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.registry_dir / "index.json"
        self._versions: list[ModelVersion] = []
        self._load_index()

    # ------------------------------------------------------------------
    # 索引管理
    # ------------------------------------------------------------------

    def _load_index(self) -> None:
        """从磁盘加载版本索引。"""
        if self._index_path.exists():
            try:
                data = json.loads(self._index_path.read_text(encoding="utf-8"))
                self._versions = [ModelVersion(**v) for v in data]
            except Exception as e:
                logger.warning("索引加载失败，重建: %s", e)
                self._rebuild_index()
        else:
            self._rebuild_index()

    def _rebuild_index(self) -> None:
        """扫描目录重建索引。"""
        self._versions = []
        for d in sorted(self.registry_dir.iterdir()):
            meta_path = d / "meta.json"
            if d.is_dir() and meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    self._versions.append(ModelVersion(**meta))
                except Exception:
                    continue
        self._save_index()

    def _save_index(self) -> None:
        """保存版本索引到磁盘。"""
        data = [asdict(v) for v in self._versions]
        self._index_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def save_model(
        self,
        model: HotBoardModel,
        board_type: str,
        train_days: int = 0,
        max_boards: int = 0,
        sample_count: int = 0,
        note: str = "",
    ) -> ModelVersion:
        """保存训练好的模型到注册表。

        Parameters
        ----------
        model : HotBoardModel
            已训练的模型实例。
        board_type : str
            "industry" 或 "concept"。
        train_days : int
            训练数据天数。
        max_boards : int
            采集板块数量。
        sample_count : int
            训练样本数。
        note : str
            可选备注。

        Returns
        -------
        ModelVersion
            保存后的版本元数据。
        """
        now = datetime.now(tz=timezone.utc)
        version_id = now.strftime("%Y%m%d_%H%M%S")
        version_dir = self.registry_dir / f"v_{version_id}"
        version_dir.mkdir(parents=True, exist_ok=True)

        # 保存模型文件
        model_path = version_dir / "model.joblib"
        joblib.dump({
            "model": model.model,
            "feature_columns": model.feature_columns,
            "params": model.params,
            "metrics": model._metrics,
        }, model_path)

        # 构建元数据
        metrics = model.metrics or ModelMetrics()
        version = ModelVersion(
            version_id=version_id,
            created_at=now.isoformat(),
            board_type=board_type,
            train_days=train_days,
            max_boards=max_boards,
            sample_count=sample_count,
            feature_count=len(model.feature_columns),
            feature_columns=list(model.feature_columns),
            params={k: v for k, v in model.params.items()
                    if isinstance(v, (int, float, str, bool))},
            auc=metrics.auc,
            f1=metrics.f1,
            precision=metrics.precision,
            recall=metrics.recall,
            accuracy=metrics.accuracy,
            note=note,
        )

        # 保存元数据
        meta_path = version_dir / "meta.json"
        meta_path.write_text(
            json.dumps(asdict(version), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # 更新索引
        self._versions.append(version)
        self._save_index()

        logger.info("模型已保存: %s (AUC=%.4f)", version_id, version.auc)
        return version

    def list_versions(
        self,
        board_type: str | None = None,
    ) -> list[ModelVersion]:
        """列出所有模型版本。

        Parameters
        ----------
        board_type : str, optional
            按板块类型过滤 ("industry" / "concept")。

        Returns
        -------
        list[ModelVersion]
            按创建时间倒序排列的版本列表。
        """
        versions = self._versions
        if board_type:
            versions = [v for v in versions if v.board_type == board_type]
        return sorted(versions, key=lambda v: v.created_at, reverse=True)

    def load_model(self, version_id: str) -> HotBoardModel | None:
        """加载指定版本的模型。

        Parameters
        ----------
        version_id : str
            版本ID。

        Returns
        -------
        HotBoardModel | None
            加载后的模型，如果不存在返回 None。
        """
        version_dir = self.registry_dir / f"v_{version_id}"
        model_path = version_dir / "model.joblib"

        if not model_path.exists():
            logger.warning("模型文件不存在: %s", model_path)
            return None

        try:
            data = joblib.load(model_path)
            model = HotBoardModel(
                feature_columns=data.get("feature_columns"),
                params=data.get("params"),
            )
            model.model = data["model"]
            model._metrics = data.get("metrics")
            model.is_trained = model.model is not None
            logger.info("模型已加载: %s", version_id)
            return model
        except Exception as e:
            logger.error("模型加载失败: %s", e)
            return None

    def get_version(self, version_id: str) -> ModelVersion | None:
        """获取指定版本的元数据。"""
        for v in self._versions:
            if v.version_id == version_id:
                return v
        return None

    def delete_version(self, version_id: str) -> bool:
        """删除指定版本。

        Parameters
        ----------
        version_id : str
            要删除的版本ID。

        Returns
        -------
        bool
            删除成功返回 True。
        """
        version_dir = self.registry_dir / f"v_{version_id}"
        if version_dir.exists():
            shutil.rmtree(version_dir)

        self._versions = [v for v in self._versions if v.version_id != version_id]
        self._save_index()
        logger.info("模型已删除: %s", version_id)
        return True

    def compare_versions(
        self,
        version_ids: list[str],
    ) -> list[dict]:
        """对比多个版本的性能指标。

        Parameters
        ----------
        version_ids : list[str]
            要对比的版本ID列表。

        Returns
        -------
        list[dict]
            每个版本的对比数据。
        """
        result = []
        for vid in version_ids:
            v = self.get_version(vid)
            if v:
                bt_display = "行业板块" if v.board_type == "industry" else "概念板块"
                result.append({
                    "版本": v.version_id,
                    "创建时间": v.created_at[:19].replace("T", " "),
                    "板块类型": bt_display,
                    "训练天数": v.train_days,
                    "样本数": v.sample_count,
                    "特征数": v.feature_count,
                    "AUC": round(v.auc, 4),
                    "F1": round(v.f1, 4),
                    "精确率": round(v.precision, 4),
                    "召回率": round(v.recall, 4),
                    "备注": v.note,
                })
        return result

    @property
    def version_count(self) -> int:
        return len(self._versions)
