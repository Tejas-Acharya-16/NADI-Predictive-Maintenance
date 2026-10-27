"""Plain deterministic RUL/anomaly triage with idempotent mock work orders."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TriageConfig:
    warning_rul_cycles: float = 50.0
    critical_rul_cycles: float = 20.0
    anomaly_persistence_cycles: int = 3
    low_confidence_interval_width: float = 50.0


class RuleBasedTriage:
    """Use explainable rules; this is a simulated maintenance-demo component."""

    def __init__(self, config: TriageConfig = TriageConfig()) -> None:
        self.config = config
        self._anomaly_streak: dict[int, int] = {}
        self._orders: dict[str, dict[str, Any]] = {}

    def assess(
        self,
        *,
        engine_id: int,
        cycle: int,
        predicted_rul: float,
        anomaly: bool,
        anomaly_score: float,
        anomaly_threshold: float,
        data_quality: str,
        model_version: str,
        interval_width: float | None = None,
    ) -> dict[str, Any]:
        streak = self._anomaly_streak.get(engine_id, 0) + 1 if anomaly else 0
        self._anomaly_streak[engine_id] = streak
        persistent = streak >= self.config.anomaly_persistence_cycles
        reasons: list[str] = []
        if data_quality != "good":
            severity, action = "EXCEPTION", "human data-quality review"
            reasons.append(f"data_quality={data_quality}")
        elif interval_width is not None and interval_width > self.config.low_confidence_interval_width:
            severity, action = "EXCEPTION", "human model-confidence review"
            reasons.append(f"interval_width={interval_width:.1f}")
        elif predicted_rul <= self.config.critical_rul_cycles and persistent:
            severity, action = "L3", "high-priority inspection"
            reasons.extend([f"predicted_rul={predicted_rul:.1f}", "persistent anomaly"])
        elif predicted_rul <= self.config.critical_rul_cycles:
            severity, action = "L2", "planned inspection"
            reasons.append(f"predicted_rul={predicted_rul:.1f}")
        elif predicted_rul <= self.config.warning_rul_cycles or persistent:
            severity, action = "L1", "inspection recommendation"
            reasons.append(f"predicted_rul={predicted_rul:.1f}" if predicted_rul <= self.config.warning_rul_cycles else "persistent anomaly")
        else:
            severity, action = "L0", "continue monitoring"
            reasons.append("rul and anomaly state within configured demo thresholds")
        # A repeated replay event updates the same open recommendation instead
        # of producing a work order every few cycles.
        work_order_key = f"engine_{engine_id}:{severity}"
        result = {
            "work_order_key": work_order_key,
            "asset_id": f"engine_{engine_id:03d}",
            "timestamp": f"cycle_{cycle}",
            "severity": severity,
            "recommended_action": action,
            "predicted_rul": float(predicted_rul),
            "anomaly_score": float(anomaly_score),
            "anomaly_threshold": float(anomaly_threshold),
            "anomaly_persistence": streak,
            "data_quality": data_quality,
            "model_version": model_version,
            "trigger_reason": "; ".join(reasons),
            "triage_config": asdict(self.config),
        }
        if severity != "L0":
            self._orders.setdefault(work_order_key, result)
        return result

    @property
    def work_orders(self) -> list[dict[str, Any]]:
        return list(self._orders.values())
