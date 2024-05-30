#!/usr/bin/env python
# -*-coding:utf-8-*-


"""
This module handes typechecking+loading YAML config files, used to configure
or tune a Deep Learning optimization run (e.g. optimizer hyperparameters,
batch size...).
"""


import sys
# for omegaconf
from dataclasses import dataclass

#
from omegaconf import MISSING, OmegaConf


# ##############################################################################
# # OPTIMIZER SETTING SCHEMAS
# ##############################################################################
@dataclass
class SGDHyperparams:
    lr: float = MISSING


@dataclass
class AdamHyperparams:
    lr: float = MISSING
    # betas: Tuple[float] = MISSING
    beta1: float = MISSING
    beta2: float = MISSING
    eps: float = MISSING


# ##############################################################################
# # SCHEDULE SETTING SCHEMAS
# ##############################################################################
@dataclass
class CosineSchedule:
    """
    :cvar warmup: Ratio of the total training to ramp up from zero to target
      value.
    """

    warmup: float = 0


@dataclass
class CosineWrSchedule:
    """
    :cvar warmup: Number of steps needed to linearly increase from zero to
      target value.
    :var cycle: Duration of the first cosine decay cycle to zero after target
      value was reached (in number of steps)
    :cvar decay: Each consecutive cycle is multiplied by this value.
    :cvar slowdown: The duration of each consecutive cycle is multiplied by
      this value.
    """

    warmup: float = 0
    cycle: int = MISSING
    decay: float = 1.0
    slowdown: float = 1.0


@dataclass
class TrapezoidalSchedule:
    """
    :cvar warmup: Ratio of the total training to linearly increase from zero to
      target value, at beginning.
    :cvar cooldown: Ratio of the total training to linearly decrease from
      target value to zero, at end.
    """

    warmup: float = MISSING
    cooldown: float = MISSING


# ##############################################################################
# # CONVENIENCE FUNCTIONS
# ##############################################################################
def get_config(yaml_path, problem="cifar10_3c3d", optimizer="SGD"):
    """ """
    problem_sett = OmegaConf.load(yaml_path)[problem]
    sett = problem_sett[optimizer]
    # get BS and num epochs from problem
    bs, epochs = problem_sett["batch_size"], problem_sett["num_epochs"]
    # get class and hpars from requested optimizer
    opt_hpar_class = getattr(sys.modules[__name__], optimizer + "Hyperparams")
    opt_hpars = dict(
        OmegaConf.merge(OmegaConf.structured(opt_hpar_class), sett["opt_hpars"])
    )
    # get LR schedule and its values
    sched = sett["lr_schedule"]
    if sched is None:  # "constant" schedule
        lr_sched = None
    else:
        lr_sched_hpar_class = getattr(sys.modules[__name__], sched)
        lr_sched = dict(
            OmegaConf.merge(
                OmegaConf.structured(lr_sched_hpar_class),
                sett["lr_schedule_hpars"],
            )
        )
    # gather and return
    return (bs, epochs), (optimizer, opt_hpars), (sched, lr_sched)
