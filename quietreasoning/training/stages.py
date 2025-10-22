"""Training stage scheduler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from quietreasoning.config import StageSwitch, TrainingConfig


@dataclass
class StageState:
    stage: StageSwitch
    index: int
    consumed_tokens: float
    total_tokens: float
    features: Dict[str, bool]


class StageScheduler:
    def __init__(self, cfg: TrainingConfig) -> None:
        self.cfg = cfg
        self.boundaries = []
        total = 0.0
        for stage in cfg.stages:
            total += stage.tokens
            self.boundaries.append(total)
        self._default_features: Dict[str, bool] = {
            "workspace": True,
            "pkm": True,
            "adapters": True,
            "ssm": True,
            "retrieval": True,
            "knn": True,
        }

    def _features_for_stage(self, stage: StageSwitch) -> Dict[str, bool]:
        features = self._default_features.copy()
        for key, value in stage.enable.items():
            features[key] = bool(value)
        if "knn" not in stage.enable and "retrieval" in stage.enable and not stage.enable["retrieval"]:
            features["knn"] = False
        return features

    def stage_at(self, seen_tokens: float) -> StageState:
        total = 0.0
        for idx, stage in enumerate(self.cfg.stages):
            total += stage.tokens
            if seen_tokens < total or stage.tokens == 0:
                consumed = seen_tokens - (total - stage.tokens)
                consumed = max(consumed, 0.0)
                return StageState(
                    stage=stage,
                    index=idx,
                    consumed_tokens=consumed,
                    total_tokens=stage.tokens,
                    features=self._features_for_stage(stage),
                )
        last = self.cfg.stages[-1]
        return StageState(
            stage=last,
            index=len(self.cfg.stages) - 1,
            consumed_tokens=last.tokens,
            total_tokens=last.tokens,
            features=self._features_for_stage(last),
        )

    def learning_rate_scale(self, global_step_tokens: float) -> float:
        state = self.stage_at(global_step_tokens)
        if state.stage.lr_peak is None or state.total_tokens == 0:
            return 1.0
        progress = state.consumed_tokens / state.total_tokens
        return min(progress, 1.0) * (state.stage.lr_peak / self.cfg.optimizer.learning_rate)
