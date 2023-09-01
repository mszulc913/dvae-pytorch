import re

import pytest

from dvae_pytorch.training.params_schedule import LinearScheduler


def test_linear_scheduler() -> None:
    scheduler = LinearScheduler(start=0.0, end=0.3, steps=3, warmup=0.0, cooldown=0.0)

    assert pytest.approx(scheduler.get_value()) == 0.0
    scheduler.step()
    assert pytest.approx(scheduler.get_value()) == 0.1
    scheduler.step()
    assert pytest.approx(scheduler.get_value()) == 0.2
    scheduler.step()
    assert pytest.approx(scheduler.get_value()) == 0.3
    scheduler.step()
    assert pytest.approx(scheduler.get_value()) == 0.3


def test_linear_scheduler_with_warmup_and_cooldown() -> None:
    scheduler = LinearScheduler(start=0.0, end=0.3, steps=5, warmup=0.3, cooldown=0.3)

    assert pytest.approx(scheduler.get_value()) == 0.0
    scheduler.step()
    assert pytest.approx(scheduler.get_value()) == 0.0
    scheduler.step()
    assert pytest.approx(scheduler.get_value()) == 0.1
    scheduler.step()
    assert pytest.approx(scheduler.get_value()) == 0.2
    scheduler.step()
    assert pytest.approx(scheduler.get_value()) == 0.3
    scheduler.step()
    assert pytest.approx(scheduler.get_value()) == 0.3


@pytest.mark.parametrize(
    (
        "start",
        "end",
        "steps",
        "warmup",
        "cooldown",
        "expected_message",
    ),
    [
        (0.0, 0.3, 3, 0.5, 0.6, "`warmup` + `cooldown` must be less than 1."),
        (0.0, 0.3, 3, 0.5, 0.5, "`warmup` + `cooldown` must be less than 1."),
        (0.0, 0.3, 3, 1.1, 0.0, "`warmup` must be between 0 and 1."),
        (0.0, 0.3, 3, 0.0, 1.1, "`cooldown` must be between 0 and 1."),
        (0.0, 0.3, 3, -0.1, 0.0, "`warmup` must be between 0 and 1."),
        (0.0, 0.3, 3, 0.0, -0.1, "`cooldown` must be between 0 and 1."),
        (0.0, 0.3, 0, 0.0, 0.0, "`steps` must be at least 1."),
        (0.0, 0.3, -1, 0.0, 0.0, "`steps` must be at least 1."),
    ],
)
def test_linear_scheduler_with_invalid_params(
    start: float,
    end: float,
    steps: int,
    warmup: float,
    cooldown: float,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=re.escape(expected_message)):
        LinearScheduler(start=start, end=end, steps=steps, warmup=warmup, cooldown=cooldown)
