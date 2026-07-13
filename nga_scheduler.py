import math

class BaseScheduler:
    def __init__(self, *, net_growth_cap: int, net_growth_min: int, drop_cap: int, reentry_penalty: float):
        self.net_growth_cap = net_growth_cap
        self.net_growth_min = net_growth_min
        self.drop_cap = drop_cap
        self.reentry_penalty = reentry_penalty
        self._check_validness()

    def _check_validness(self):
        assert self.net_growth_min >= 0
        assert self.net_growth_cap >= self.net_growth_min
        assert self.drop_cap >= 0
        assert 0 <= self.reentry_penalty <= 1

    def update(self, runner):
        return

    def to_dict(self):
        return {
            'type': type(self).__name__,
            **self.__dict__,
        }

class RateScheduler(BaseScheduler):
    def __init__(
        self,
        *,
        net_growth_cap_base: int,
        net_growth_cap_rate: float,
        net_growth_min: int,
        drop_cap_base: int,
        drop_cap_rate: float,
        reentry_penalty: float,
    ):
        self.net_growth_cap = None
        self.net_growth_min = net_growth_min
        self.drop_cap = None
        self.reentry_penalty = reentry_penalty

        self.net_growth_cap_base = net_growth_cap_base
        self.net_growth_cap_rate = net_growth_cap_rate
        self.drop_cap_base = drop_cap_base
        self.drop_cap_rate = drop_cap_rate

    def update(self, runner):
        basis_size = len(runner.basis_reprs)
        self.net_growth_cap = max(
            self.net_growth_cap_base,
            math.ceil(self.net_growth_cap_rate * basis_size),
        )
        self.drop_cap = max(
            self.drop_cap_base,
            math.ceil(self.drop_cap_rate * basis_size),
        )
        self._check_validness()

class DecayScheduler(BaseScheduler):
    def __init__(
        self,
        *,
        net_growth_cap_base: int,
        net_growth_cap_rate: float,
        net_growth_min: int,
        drop_cap_base: int,
        drop_cap_rate: float,
        drop_decay_start: int,
        drop_decay_end: int,
        reentry_penalty: float,
    ):
        assert drop_decay_end > drop_decay_start
        self.net_growth_cap = None
        self.net_growth_min = net_growth_min
        self.drop_cap = None
        self.reentry_penalty = reentry_penalty

        self.net_growth_cap_base = net_growth_cap_base
        self.net_growth_cap_rate = net_growth_cap_rate
        self.drop_cap_base = drop_cap_base
        self.drop_cap_rate = drop_cap_rate
        self.drop_decay_start = drop_decay_start
        self.drop_decay_end = drop_decay_end

    def update(self, runner):
        basis_size = len(runner.basis_reprs)
        self.net_growth_cap = max(
            self.net_growth_cap_base,
            math.ceil(self.net_growth_cap_rate * basis_size),
        )

        if basis_size < self.drop_decay_start:
            self.drop_cap = max(
                self.drop_cap_base,
                math.ceil(self.drop_cap_rate * basis_size),
            )
        else:
            drop_cap_max = max(
                self.drop_cap_base,
                math.ceil(self.drop_cap_rate * self.drop_decay_start),
            )
            decay_factor = 1 - (basis_size - self.drop_decay_start) / (self.drop_decay_end - self.drop_decay_start)
            decay_factor = max(0, decay_factor)
            self.drop_cap = math.ceil(
                self.drop_cap_base + decay_factor * (drop_cap_max - self.drop_cap_base)
            )

        self._check_validness()
