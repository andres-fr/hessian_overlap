#!/usr/bin/env python
# -*-coding:utf-8-*-


"""
This script trains a DeepOBS problem and logs parameters and gradients
along the way, together with other relevant metadata and training statistics
like batch loss and validation accuracy.
"""


import os
import random
from copy import deepcopy
from dataclasses import dataclass

# for omegaconf
from typing import List, Optional

import numpy as np

import hessianoverlap.optimizers
import torch

#
from hessianoverlap.config_io import get_config
from hessianoverlap.incremental_hdf5 import IncrementalHDF5
from hessianoverlap.logging import JsonColorLogger, make_timestamp
from hessianoverlap.obs_extensions import get_obs_testproblem
from hessianoverlap.obs_extensions.samplers import SubsetSampler

#
from deepobs.config import set_data_dir

#
from omegaconf import MISSING, OmegaConf


# ##############################################################################
# # HELPERS
# ##############################################################################
def create_hdf5(path, height, dtype=np.float32):
    """ """
    result = IncrementalHDF5(
        path,
        height,
        dtype=np.float32,
        compression="lzf",
        data_chunk_length=1,
        metadata_chunk_length=1,
        err_if_exists=True,
    )
    return result


def clear_model_gradients(model):
    """ """
    for p in model.parameters():
        p.grad = None


# ##############################################################################
# # CLI
# ##############################################################################
@dataclass
class ConfDef:
    """
    :cvar MAX_STEPS: If given, train for only this many batches
    """

    OBS_DATASETS: str = MISSING  # /shared/datasets/DeepOBS
    #
    TUNING_CONF: str = MISSING
    OPTIMIZER: str = MISSING
    PROBLEM: str = "mnist_mini"
    #
    RANDOM_SEED: Optional[int] = None
    MAX_STEPS: Optional[int] = None
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    #
    OUTPUT_DIR: str = "output"
    OUTDIR_SUFFIX: Optional[str] = None
    RECORD_STEPS: Optional[List[int]] = None
    NUM_HESSIAN_DATAPOINTS: int = MISSING
    STORE_GRADS: bool = False
    WITH_EVAL: bool = True


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

    timestamp = make_timestamp(timezone="Europe/Berlin", with_tz_output=False)

    # create directory to store results for this run
    run_dir = os.path.join(
        conf.OUTPUT_DIR,
        "__".join([timestamp, conf.PROBLEM, str(conf.RANDOM_SEED)]),
    )
    if conf.OUTDIR_SUFFIX is not None:
        run_dir += "__" + conf.OUTDIR_SUFFIX

    os.makedirs(run_dir, exist_ok=False)

    txt_logger = JsonColorLogger(f"[{os.path.basename(__file__)}]", run_dir)
    txt_logger.loj("CONFIG", OmegaConf.to_container(conf))

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

    # set up optimizer
    opt_class = getattr(hessianoverlap.optimizers, "Scheduled" + conf.OPTIMIZER)
    opt = opt_class(model.parameters(), **opt_hpars, lr_sched=None)

    # prepare hessian globals and dataloaders
    record_steps = conf.RECORD_STEPS
    record_steps = set() if record_steps is None else set(record_steps)
    param_shapes = [tuple(p.shape) for p in params]
    num_params = sum([p.numel() for p in params])
    steps_per_epoch = len(train_loader)
    max_steps = (
        conf.MAX_STEPS
        if conf.MAX_STEPS is not None
        else (num_epochs * steps_per_epoch)
    )
    record_steps = {x for x in record_steps if x <= max_steps}

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
    txt_logger.loj(
        "GLOBALS",
        {
            "run_dir": run_dir,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "max_steps": max_steps,
            "steps_per_epoch": steps_per_epoch,
            "opt_name": opt_name,
            "opt_hpars": opt_hpars,
            "sched_name": sched_name,
            "sched_hpars": sched_hpars,
            "param_shapes": param_shapes,
            "num_params": num_params,
            "H_train_idxs": H_train_dl.sampler.indices,
            "H_test_idxs": H_test_dl.sampler.indices,
            "record_steps": list(sorted(record_steps)),
        },
    )

    #
    rundir_base = os.path.basename(run_dir)
    params_h5 = create_hdf5(
        os.path.join(run_dir, f"{rundir_base}__params.h5"), num_params
    )
    if conf.STORE_GRADS:
        g_train_h5 = create_hdf5(
            os.path.join(run_dir, f"{rundir_base}__gradients_train.h5"),
            num_params,
        )
        g_test_h5 = create_hdf5(
            os.path.join(run_dir, f"{rundir_base}__gradients_test.h5"),
            num_params,
        )

    # ##########################################################################
    # # MAIN LOOP
    # ##########################################################################
    global_step = 0
    for epoch in range(1, num_epochs + 1):
        if global_step > max_steps:
            break
        if conf.WITH_EVAL:
            # At beg of each epoch, evaluate and log to Cockpit
            model.eval()
            (
                train_loss,
                train_acc,
                xv_loss,
                xv_acc,
                test_loss,
                test_acc,
            ) = eval_fn()
            txt_logger.loj(
                f"EVAL ROUND",
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "xv_loss": xv_loss,
                    "xv_acc": xv_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                },
            )
        # Training loop
        model.train()
        for i, (inputs, targets) in enumerate(iter(train_loader), 1):
            if global_step > max_steps:
                break

            # Note that targets.shape=[32], outputs.shape=[32, 100], but the
            # loss_fn applies "mean" reduction, so the 100 outptus are avgd.
            inputs, targets = inputs.to(conf.DEVICE), targets.to(conf.DEVICE)
            opt.zero_grad()
            # fwpass
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            losses = losses_fn(outputs, targets)

            # ##################################################################
            # # RECORDING VALUES
            # ##################################################################
            # add individual loss data to logdict
            if (record_steps is None) or (global_step in record_steps):
                # clone model
                model_clone = deepcopy(model)
                model_clone.load_state_dict(
                    {k: v.clone() for k, v in model.state_dict().items()}
                )
                model_clone.eval()
                params = list(model_clone.parameters())

                # save cloned parameters
                p_flat = (
                    torch.cat([p.flatten() for p in params])
                    .unsqueeze(1)
                    .cpu()
                    .detach()
                    .numpy()
                )
                params_h5.append(p_flat, str({"step": global_step}))
                txt_logger.loj(
                    "HDF5",
                    {
                        "global_step": global_step,
                        "matrix": "batch_parameters",
                        "split": "batch",
                        "status": "saved",
                    },
                )

                # compute average gradients over Hessian train set
                # https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/20
                clear_model_gradients(model_clone)
                minitrain_loss = 0
                for mini_x, mini_y in H_train_dl:
                    mini_out = model_clone(mini_x.to(conf.DEVICE))
                    mnls = loss_fn(mini_out, mini_y.to(conf.DEVICE))
                    mnls.backward()  # accumulate gradients
                    minitrain_loss += mnls.item() * len(mini_y)
                minitrain_loss /= len(H_train_dl.sampler.indices)
                #
                if conf.STORE_GRADS:
                    minitrain_grads = (
                        torch.cat(
                            [p.grad.flatten() for p in model_clone.parameters()]
                        )
                        .unsqueeze(1)
                        .cpu()
                        .detach()
                        .numpy()
                    )
                    g_train_h5.append(
                        minitrain_grads, str({"step": global_step})
                    )
                    txt_logger.loj(
                        "HDF5",
                        {
                            "global_step": global_step,
                            "matrix": "train_gradient",
                            "split": "batch",
                            "status": "saved",
                        },
                    )

                # remove model_clone gradients
                clear_model_gradients(model_clone)

                # compute average gradients over hessian test set
                minitest_loss = 0
                for mini_x, mini_y in H_test_dl:
                    mini_out = model_clone(mini_x.to(conf.DEVICE))
                    mnls = loss_fn(mini_out, mini_y.to(conf.DEVICE))
                    mnls.backward()  # accumulate gradients
                    minitest_loss += mnls.item() * len(mini_y)
                minitest_loss /= len(H_test_dl.sampler.indices)
                #
                if conf.STORE_GRADS:
                    minitest_grads = (
                        torch.cat(
                            [p.grad.flatten() for p in model_clone.parameters()]
                        )
                        .unsqueeze(1)
                        .cpu()
                        .detach()
                        .numpy()
                    )
                    g_test_h5.append(minitest_grads, str({"step": global_step}))
                    txt_logger.loj(
                        "HDF5",
                        {
                            "global_step": global_step,
                            "matrix": "test_gradient",
                            "split": "batch",
                            "status": "saved",
                        },
                    )

                # log all scalars from main training and from this
                current_lr = opt.param_groups[0]["lr"]
                txt_logger.loj(
                    "TRAINING BATCH",
                    {
                        "global_step": global_step,
                        "epoch": epoch,
                        "batch_loss": loss.item(),
                        "H_train_loss": minitrain_loss,
                        "H_test_loss": minitest_loss,
                        "lr": current_lr,
                    },
                )

            loss.backward()
            opt.step()
            global_step += 1
    if conf.WITH_EVAL:
        # final evaluation
        train_loss, train_acc, xv_loss, xv_acc, test_loss, test_acc = eval_fn()
        txt_logger.loj(
            f"FINAL EVALUATION",
            {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "xv_loss": xv_loss,
                "xv_acc": xv_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
            },
        )
    params_h5.close()
    if conf.STORE_GRADS:
        g_train_h5.close()
        g_test_h5.close()
