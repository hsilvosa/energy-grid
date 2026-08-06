import pytest

from energy_grid.rollback import RollbackController, RollbackState, promotion_gate


def test_promotion_requires_improvement_and_slice_safety() -> None:
    assert promotion_gate(
        champion_error=100, challenger_error=97, slice_regressions=[0.05]
    ).promote
    assert not promotion_gate(
        champion_error=100, challenger_error=99, slice_regressions=[]
    ).promote
    assert not promotion_gate(
        champion_error=100, challenger_error=95, slice_regressions=[0.11]
    ).promote


def test_rollback_requires_three_consecutive_breaches() -> None:
    controller = RollbackController()
    state = RollbackState()
    assert not controller.evaluate(state, champion_error=125, reference_error=100)
    assert not controller.evaluate(state, champion_error=121, reference_error=100)
    assert controller.evaluate(state, champion_error=130, reference_error=100)
    assert state.consecutive_breaches == 3


def test_healthy_window_resets_rollback_counter() -> None:
    controller = RollbackController()
    state = RollbackState(2)
    assert not controller.evaluate(state, champion_error=110, reference_error=100)
    assert state.consecutive_breaches == 0
    with pytest.raises(ValueError):
        controller.evaluate(state, champion_error=1, reference_error=0)

