#!/usr/bin/env python
# -*-coding:utf-8-*-


"""
Some DeepOBS problems trigger CurvLinOps errors, e.g. cifar10_3c3d triggers
check for deterministic loss because it contains non-deterministic data
augmentations.

To allow debugging this, this script starts the setup and attempts to create a
Hessian.
"""


import random
from dataclasses import dataclass
# for omegaconf
from typing import Optional

import torch
from curvlinops import HessianLinearOperator
#
from deepobs.config import set_data_dir
#
from lth_hessian.config_io import get_config
from lth_hessian.obs_extensions import get_obs_testproblem
from lth_hessian.obs_extensions.samplers import SubsetSampler
#
from omegaconf import MISSING, OmegaConf


# ##############################################################################
# # CLI
# ##############################################################################
@dataclass
class ConfDef:
    """ """

    OBS_DATASETS: str = MISSING  # /shared/datasets/DeepOBS
    #
    TUNING_CONF: str = MISSING
    OPTIMIZER: str = MISSING
    PROBLEM: str = "mnist_mini"
    #
    RANDOM_SEED: Optional[int] = None
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    #
    NUM_HESSIAN_DATAPOINTS: int = MISSING


# ##############################################################################
# # MAIN ROUTINE
# ##############################################################################
if __name__ == "__main__":
    conf = OmegaConf.structured(ConfDef())
    cli_conf = OmegaConf.from_cli()
    conf = OmegaConf.merge(conf, cli_conf)

    # if no seed is given, take a random one
    if conf.RANDOM_SEED is None:
        conf.RANDOM_SEED = random.randint(0, 1e7)

    # Avoid re-downloading dataset on local dir
    set_data_dir(conf.OBS_DATASETS)

    # retrieve setting from YAML file
    (
        (batch_size, num_epochs),
        (opt_name, opt_hpars),
        (sched_name, sched_hpars),
    ) = get_config(conf.TUNING_CONF, conf.PROBLEM, conf.OPTIMIZER)

    # Set up problem and model
    (
        model,
        train_loader,
        loss_fn,
        losses_fn,
        eval_fn,
        tproblem,
    ) = get_obs_testproblem(conf.PROBLEM, batch_size, conf.RANDOM_SEED)

    model.eval()  # for deterministic behavior
    model = model.to(conf.DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]

    # extend DeepOBS default dataloaders with custom samplers, to provide
    # custom, fixed subsets for Hessians and rule-based seeds for training
    train_ds = tproblem.data._make_train_and_valid_dataloader()[0].dataset
    H_train_sampler = SubsetSampler.get_balanced(
        train_ds, random=False, size=conf.NUM_HESSIAN_DATAPOINTS
    )
    H_train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=False, sampler=H_train_sampler
    )
    #
    test_ds = tproblem.data._make_test_dataloader().dataset
    H_test_sampler = SubsetSampler.get_balanced(
        test_ds, random=False, size=conf.NUM_HESSIAN_DATAPOINTS
    )
    H_test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, sampler=H_test_sampler
    )

    #
    params = [p for p in model.parameters() if p.requires_grad]
    H_train = HessianLinearOperator(model, loss_fn, params, H_train_dl)
    H_test = HessianLinearOperator(model, loss_fn, params, H_test_dl)
    print("If H didn't crash, we are good")
    breakpoint()
