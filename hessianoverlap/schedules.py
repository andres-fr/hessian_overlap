#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
This module implements several generators that, when iterated, produce
sequences of scalars following different schedules which may be parametrized
at initialization.
"""


import numpy as np

import torch


# ##############################################################################
# # SCHEDULERS
# ##############################################################################
class ConstantSchedule:
    """ """

    def __init__(self, value=1.0):
        """ """
        self.value = value
        self.reset()

    def __iter__(self):
        """ """
        return self

    def __next__(self):
        """ """
        result = self()
        return result

    def reset(self):
        """ """
        self.schedule = self.schedule_generator()

    def __call__(self):
        """ """
        result = next(self.schedule)
        return result

    # override this for more schedules
    def schedule_generator(self):
        """ """
        while True:
            yield self.value


class CosineSchedule(ConstantSchedule):
    """ """

    PI_HALF = torch.pi / 2

    def __init__(self, maximum=1.0, num_steps=1000, warmup=0.1):
        """ """
        # initial values
        self.maximum = maximum
        self.num_steps = num_steps
        self.warmup = warmup
        #
        self.reset()

    def schedule_generator(self):
        """ """
        assert 0 < self.warmup < 1, "Warmup must be between 0 and 1!"
        warmup_steps = round(self.num_steps * self.warmup)
        cosine_steps = self.num_steps - warmup_steps
        #
        step = 0
        while step < warmup_steps:
            value = self.maximum * (step / warmup_steps)
            #
            step += 1
            yield value
        #
        step = 0
        while step < cosine_steps:
            value = (
                self.maximum
                * torch.cos(
                    torch.tensor(self.PI_HALF * (step / (cosine_steps - 1)))
                ).item()
            )
            #
            step += 1
            yield value


class TrapezoidalSchedule(ConstantSchedule):
    """ """

    def __init__(self, maximum=1.0, num_steps=1000, warmup=0.1, cooldown=0.1):
        """ """
        # initial values
        self.maximum = maximum
        self.num_steps = num_steps
        self.warmup = warmup
        self.cooldown = cooldown
        #
        self.reset()

    def schedule_generator(self):
        """ """

        assert 0 <= self.warmup <= 1, "Warmup must be between 0 and 1!"
        assert 0 <= self.cooldown <= 1, "Cooldown must be between 0 and 1!"
        assert (
            self.warmup + self.cooldown
        ) <= 1, "Warmup+cooldown must be <= 1!"
        #
        warmup_steps = round(self.num_steps * self.warmup)
        cooldown_steps = round(self.num_steps * self.cooldown)
        plateau_steps = self.num_steps - (warmup_steps + cooldown_steps)
        # warmup
        step = 0
        while step < warmup_steps:
            value = self.maximum * (step / warmup_steps)
            #
            step += 1
            yield value
        # plateau
        step = 0
        while step < plateau_steps:
            value = self.maximum
            #
            step += 1
            yield value
        # cooldown
        step = cooldown_steps - 1
        while step >= 0:
            value = self.maximum * (step / (cooldown_steps - 1))
            #
            step -= 1
            yield value


class ExpTrapezoidalSchedule(ConstantSchedule):
    """ """

    def __init__(self, maximum=1.0, num_steps=1000, warmup=0.1, cooldown=0.1):
        """ """
        # initial values
        self.maximum = maximum
        self.num_steps = num_steps
        self.warmup = warmup
        self.cooldown = cooldown
        #
        self.reset()

    def schedule_generator(self):
        """ """
        assert 0 <= self.warmup <= 1, "Warmup must be between 0 and 1!"
        assert 0 <= self.cooldown <= 1, "Cooldown must be between 0 and 1!"
        assert (
            self.warmup + self.cooldown
        ) <= 1, "Warmup+cooldown must be <= 1!"
        #
        warmup_steps = round(self.num_steps * self.warmup)
        cooldown_steps = round(self.num_steps * self.cooldown)
        plateau_steps = self.num_steps - (warmup_steps + cooldown_steps)
        # warmup
        step = 0
        while step < warmup_steps:
            lin_value = np.log(self.maximum + 1) * (step / warmup_steps)
            value = np.exp(lin_value) - 1
            #
            step += 1
            yield value
        # plateau
        step = 0
        while step < plateau_steps:
            value = self.maximum
            #
            step += 1
            yield value
        # cooldown
        step = cooldown_steps - 1
        while step >= 0:
            lin_value = np.log(self.maximum + 1) * (step / (cooldown_steps - 1))
            value = np.exp(lin_value) - 1
            # value = self.maximum * (step / (cooldown_steps - 1))
            #
            step -= 1
            yield value


class CosineWrSchedule(CosineSchedule):
    """ """

    def __init__(
        self, maximum=1.0, period=1000, decay=1.0, slowdown=1.0, warmup=0.1
    ):
        """ """
        # initial values
        self._initial_maximum = maximum
        self._initial_period = period
        self._decay = decay
        self._slowdown = slowdown
        self.warmup = warmup
        #
        self.reset()

    def cycle_generator(self, maxval, period):
        """ """
        step = 0
        while step < period:
            value = (
                maxval
                * torch.cos(
                    torch.tensor(self.PI_HALF * (step / (period - 1)))
                ).item()
            )
            #
            step += 1
            yield value

    def schedule_generator(self):
        """ """
        assert 0 < self.warmup < 1, "Warmup must be between 0 and 1!"
        warmup_steps = round(self._initial_period * self.warmup)
        # warmup
        step = 0
        while step < warmup_steps:
            value = self._initial_maximum * (step / warmup_steps)
            #
            step += 1
            yield value
        # cosine cycles
        maxval = self._initial_maximum
        period = self._initial_period
        while True:
            yield from self.cycle_generator(maxval, period)
            maxval *= self._decay
            period *= self._slowdown
