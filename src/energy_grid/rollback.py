from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromotionResult:
    promote: bool
    reason: str


def promotion_gate(
    *,
    champion_error: float,
    challenger_error: float,
    slice_regressions: list[float],
    minimum_improvement: float = 0.02,
    maximum_slice_regression: float = 0.10,
) -> PromotionResult:
    if champion_error <= 0:
        return PromotionResult(False, "champion error must be positive")
    improvement = (champion_error - challenger_error) / champion_error
    if improvement < minimum_improvement:
        return PromotionResult(False, "challenger does not improve primary error by 2%")
    if any(regression > maximum_slice_regression for regression in slice_regressions):
        return PromotionResult(False, "challenger regresses a critical slice by more than 10%")
    return PromotionResult(True, "promotion criteria satisfied")


@dataclass
class RollbackState:
    consecutive_breaches: int = 0


class RollbackController:
    def __init__(self, threshold: float = 0.20, required_breaches: int = 3) -> None:
        self.threshold = threshold
        self.required_breaches = required_breaches

    def evaluate(
        self, state: RollbackState, *, champion_error: float, reference_error: float
    ) -> bool:
        if reference_error <= 0:
            raise ValueError("reference error must be positive")
        degraded = champion_error > reference_error * (1 + self.threshold)
        state.consecutive_breaches = state.consecutive_breaches + 1 if degraded else 0
        return state.consecutive_breaches >= self.required_breaches


def rollback_mlflow_alias(model_name: str, previous_version: str, tracking_uri: str) -> None:
    from mlflow import MlflowClient, set_tracking_uri

    set_tracking_uri(tracking_uri)
    client = MlflowClient()
    client.set_registered_model_alias(model_name, "champion", previous_version)
