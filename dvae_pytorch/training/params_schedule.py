import abc


class ParameterScheduler(abc.ABC):
    """Base class for parameter schedulers."""

    @abc.abstractmethod
    def step(self) -> None:
        """Update the parameters of the scheduler."""

    @abc.abstractmethod
    def get_value(self) -> float:
        """Get the current value of the parameter."""


class LinearScheduler(ParameterScheduler):
    """A scheduler that linearly interpolates between two values."""

    def __init__(
        self, start: float, end: float, steps: int, warmup: float = 0, cooldown: float = 0
    ) -> None:
        """Init the linear scheduler.

        Args:
            start: The start value of the parameter.
            end: The end value of the parameter.
            steps: The number of steps to take.
            warmup: Fraction of steps to warm up. The scheduler will start
                interpolating from `start` to `end` at `int(warmup * steps)` step.
            cooldown: Fraction of steps to cool down. The scheduler will be
                interpolating from `end` to `start` till `1 - int(cooldown * steps)` step.
        """
        super().__init__()
        if warmup < 0 or warmup > 1:
            raise ValueError("`warmup` must be between 0 and 1.")
        if cooldown < 0 or cooldown > 1:
            raise ValueError("`cooldown` must be between 0 and 1.")
        if warmup + cooldown >= 1:
            raise ValueError("`warmup` + `cooldown` must be less than 1.")
        if steps < 1:
            raise ValueError("`steps` must be at least 1.")
        self._value = start
        self._start_step = int(warmup * steps)
        self._end_step = steps - int(cooldown * steps)
        self._step_size = (end - start) / (self._end_step - self._start_step)
        self._step = 0

    def step(self) -> None:  # noqa: D102
        if self._start_step <= self._step < self._end_step:
            self._value += self._step_size
        self._step += 1

    def get_value(self) -> float:  # noqa: D102
        return self._value
