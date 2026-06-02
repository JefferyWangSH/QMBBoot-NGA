from dataclasses import dataclass, fields
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


@dataclass(slots=True)
class BaseBeamScheduler:
    growth_cap: int
    replace_num: int
    replace_cap: int
    grow_temperature: float
    drop_temperature: float
    reentry_penalty: float

    def __post_init__(self):
        assert self.growth_cap >= 1
        assert self.replace_num >= 0
        assert self.replace_cap >= 0
        assert self.grow_temperature >= 0
        assert self.drop_temperature >= 0
        assert 0 <= self.reentry_penalty <= 1

    def to_dict(self):
        data = {'type': type(self).__name__}
        for field in fields(self):
            data[field.name] = getattr(self, field.name)
        return data

    def update(self, runner):
        return

class RateBeamScheduler:
    def __init__(
        self,
        *,
        growth_cap_base: int,
        growth_cap_rate: float,
        replace_num: int,
        replace_cap_base: int,
        replace_cap_rate: float,
        grow_temperature: float,
        drop_temperature: float,
        reentry_penalty: float,
    ):
        self.growth_cap = None
        self.replace_cap = None
        self.replace_num = replace_num
        self.grow_temperature = grow_temperature
        self.drop_temperature = drop_temperature
        self.reentry_penalty = reentry_penalty

        self.growth_cap_base = growth_cap_base
        self.growth_cap_rate = growth_cap_rate
        self.replace_cap_base = replace_cap_base
        self.replace_cap_rate = replace_cap_rate

    def to_dict(self):
        return {'type': type(self).__name__, **self.__dict__}

    def update(self, runner):
        basis_size = len(runner.basis_reprs)
        self.growth_cap = max(
            self.growth_cap_base,
            math.ceil(self.growth_cap_rate * basis_size),
        )
        self.replace_cap = max(
            self.replace_cap_base,
            math.ceil(self.replace_cap_rate * basis_size),
        )
