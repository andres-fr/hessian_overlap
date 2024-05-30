#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
This module extends PyTorch optimizers with custom scheduled functionality
from the ``schedules`` module.
"""


import inspect

#
import torch


# ##############################################################################
# # SCHEDULED OPTIMIZERS
# ##############################################################################
class ScheduledOptimizer:
    """ """

    def set_lr(self, lr_val):
        """ """
        for g in self.param_groups:
            g["lr"] = lr_val

    def get_lr(self):
        """ """
        lr_list = [g["lr"] for g in self.param_groups]
        result = lr_list[0]
        assert all(
            lr == result for lr in lr_list
        ), "Different learning rates per group not supported"
        return result


class ScheduledSGD(torch.optim.SGD, ScheduledOptimizer):
    """ """

    def __init__(self, params, **kwargs):
        """ """
        # we expect possibly an extra "lr_sched" argument: grab it
        self.lr_sched = kwargs.pop("lr_sched", None)
        # pass the rest to the super class optimizer
        super(self.__class__, self).__init__(params, **kwargs)

    def step(self, *args, **kwargs):
        """ """
        # if we provided a scheduler, apply/update it it
        if self.lr_sched is not None:
            self.set_lr(self.lr_sched())
        # regular opt step
        super(self.__class__, self).step(*args, **kwargs)


class ScheduledAdam(torch.optim.Adam, ScheduledOptimizer):
    """ """

    def __init__(self, params, **kwargs):
        """ """
        # we expect possibly an extra "lr_sched" argument: grab it
        self.lr_sched = kwargs.pop("lr_sched", None)
        try:
            super(self.__class__, self).__init__(params, **kwargs)
        except TypeError:
            # we assume that the issue is that we gave beta1 and beta2 instead
            # of both in a "betas" tuple.
            # first, get default betas
            default_b1, default_b2 = (
                inspect.signature(torch.optim.Adam).parameters["betas"].default
            )
            # then, fetch any given betas
            b1 = kwargs.pop("beta1", default_b1)
            b2 = kwargs.pop("beta2", default_b2)
            kwargs["betas"] = (b1, b2)
            # now retry the super call
            super(self.__class__, self).__init__(params, **kwargs)

    def step(self, *args, **kwargs):
        """ """
        # if we provided a scheduler, apply/update it it
        if self.lr_sched is not None:
            self.set_lr(self.lr_sched())
        # regular opt step
        super(self.__class__, self).step(*args, **kwargs)
