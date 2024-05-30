#!/usr/bin/env python
# -*-coding:utf-8-*-


"""
This script trains mnist_mini, and, for the requested steps, computes both the
sketched and the true EIGH in-RAM, providing a log with format similar to 10d.
"""

import os
import random
from copy import deepcopy
from dataclasses import dataclass

# for omegaconf
from typing import List, Optional

import numpy as np

import hessianoverlap.optimizers
import scipy
import torch

#
from hessianoverlap.config_io import get_config
from hessianoverlap.logging import JsonColorLogger, make_timestamp
from hessianoverlap.obs_extensions import get_obs_testproblem
from hessianoverlap.obs_extensions.samplers import SubsetSampler
from curvlinops import HessianLinearOperator

#
from deepobs.config import set_data_dir

#
from omegaconf import MISSING, OmegaConf
from skerch.a_posteriori import (
    a_posteriori_error,
    a_posteriori_error_bounds,
    scree_bounds,
)
from skerch.decompositions import seigh
from skerch.linops import CompositeLinOp, DiagonalLinOp, TorchLinOpWrapper


# ##############################################################################
# # HELPERS
# ##############################################################################
class TorchHessianLinearOperator(TorchLinOpWrapper, HessianLinearOperator):
    pass


def clear_model_gradients(model):
    """ """
    for p in model.parameters():
        p.grad = None


def truncated_a_posteriori(
    M, Q, core_U, core_S, n, seed, dtype, device, trunc_dims=None
):
    """
    :param n: Number of a-posteriori measurements
    """
    dims = len(core_S) if trunc_dims is None else trunc_dims
    M_lowrank = CompositeLinOp(
        [
            ("Q", Q),
            ("U", core_U[:, :dims]),
            ("S", DiagonalLinOp(core_S[:dims])),
            ("Ut", core_U[:, :dims].T),
            ("Qt", Q.T),
        ]
    )
    #
    f1, f2, res_estimate = a_posteriori_error(
        M,
        M_lowrank,
        n,
        seed=seed,
        dtype=dtype,
        device=device,
        adjoint=False,
    )[0]
    #
    return f1, f2, res_estimate


def hessian_analysis(
    step,
    split,
    logger,
    model,
    loss_fn,
    params,
    paramvec,
    H_dl,
    model_dtype,
    conf,
):
    """ """
    #  EIGH SPECTRA AND EIGVEC ERRORS
    logger.loj(f"{split} INFO", [f"Performing full EIGH (step {step})..."])
    H = HessianLinearOperator(model, loss_fn, params, H_dl)
    H_explicit = H @ np.eye(H.shape[0])
    H_ew, H_ev = scipy.linalg.eigh(H_explicit)
    # H_explicit = torch.zeros(H.shape, dtype=model_dtype).numpy()
    # H_ew, H_ev = np.zeros_like(H_explicit[0]), np.zeros_like(H_explicit)
    #
    logger.loj(f"{split} INFO", [f"Performing Sketched EIGH (step {step})..."])
    TH = TorchHessianLinearOperator(model, loss_fn, params, H_dl)
    Q, core_U, core_S = seigh(
        TH,
        conf.DEVICE,
        model_dtype,
        conf.NUM_OUTER,
        conf.NUM_INNER,
        conf.RANDOM_SEED,
    )
    logger.loj(f"{split} RESULT", {"step": step, "H_eigvals": H_ew.tolist()})
    logger.loj(f"{split} RESULT", {"step": step, "core_S": core_S.tolist()})
    # error per eigenvector, by descending eigval magnitude
    ev_residuals = ((Q @ core_U).cpu() - H_ev[:, : conf.NUM_OUTER]).norm(dim=0)
    logger.loj(
        f"{split} RESULT",
        {"step": step, "eigvec_resnorm": ev_residuals.tolist()},
    )
    #  ESTIMATE A-POSTERIORI ERROR AND SCREE BOUNDS
    if conf.WITH_A_POSTERIORI:
        logger.loj(
            f"{split} INFO",
            [f"A-posteriori error estimation and scree bounds (step {step})"],
        )
        trunc_50 = int(np.ceil(0.5 * conf.NUM_OUTER))
        trunc_05 = int(np.ceil(0.05 * conf.NUM_OUTER))

        f1, f2, res = truncated_a_posteriori(
            TH,
            Q,
            core_U,
            core_S,
            conf.NUM_A_POSTERIORI,
            conf.RANDOM_SEED + 2,
            model_dtype,
            conf.DEVICE,
            trunc_dims=conf.NUM_OUTER,
        )
        _, f2_50, res_50 = truncated_a_posteriori(
            TH,
            Q,
            core_U,
            core_S,
            conf.NUM_A_POSTERIORI,
            conf.RANDOM_SEED + 2,
            model_dtype,
            conf.DEVICE,
            trunc_dims=trunc_50,
        )
        _, f2_05, res_05 = truncated_a_posteriori(
            TH,
            Q,
            core_U,
            core_S,
            conf.NUM_A_POSTERIORI,
            conf.RANDOM_SEED + 2,
            model_dtype,
            conf.DEVICE,
            trunc_dims=trunc_05,
        )
        logger.loj(
            f"{split} RESULT",
            {
                "info": (
                    "Frob^2 of Hessian, corresponding k-rank reconstruction "
                    + "H_k and residual (H - H_k)"
                ),
                "step": step,
                "Frob(H)^2": f1,
                f"Frob(H_{conf.NUM_OUTER})^2": f2,
                f"Frob(H_{trunc_50})^2": f2_50,
                f"Frob(H_{trunc_05})^2": f2_05,
                f"Frob(H-H_{conf.NUM_OUTER})^2": res,
                f"Frob(H-H_{trunc_50})^2": res_50,
                f"Frob(H-H_{trunc_05})^2": res_05,
            },
        )

        logger.loj(
            f"{split} RESULT",
            {
                "step": step,
                **a_posteriori_error_bounds(
                    conf.NUM_A_POSTERIORI, rel_err=0.25
                ),
                **a_posteriori_error_bounds(conf.NUM_A_POSTERIORI, rel_err=0.5),
                **a_posteriori_error_bounds(conf.NUM_A_POSTERIORI, rel_err=1),
            },
        )
        #
        lo_scree, hi_scree = scree_bounds(core_S, f1**0.5, res**0.5)
        logger.loj(
            f"{split} RESULT",
            {
                "step": step,
                "info": "Scree lower and upper bounds by increasing k",
                "scree_lower": lo_scree.tolist(),
                "scree_upper": hi_scree.tolist(),
            },
        )
    # OVERLAP GRASSMANNIAN METRIC
    frob_by_mag = []
    for k in range(1, conf.NUM_OUTER + 1):
        mask_idxs = paramvec.abs().topk(k, largest=True, sorted=False).indices
        frob = ((Q[mask_idxs] @ core_U[:, :k]) ** 2).sum().item()
        frob_by_mag.append(frob)
    logger.loj(
        f"{split} RESULT",
        {
            "step": step,
            "frob_Qtop^2": frob_by_mag,
            "info": "Frob(Q[largest_absparams,:k])^2 ordered by increasing k",
        },
    )
    logger.loj(
        f"{split} RESULT",
        {
            "step": step,
            "largest_absparams": mask_idxs.tolist(),
            "info": "Indices of k largest parameters, by descending magnitude",
        },
    )


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
    RECORD_STEPS: List[int] = MISSING
    NUM_HESSIAN_DATAPOINTS: int = MISSING
    #
    WITH_TEST_HESSIAN: bool = True
    WITH_TRAIN_HESSIAN: bool = True
    #
    NUM_OUTER: int = 355
    NUM_INNER: int = 700
    WITH_A_POSTERIORI: bool = True
    NUM_A_POSTERIORI: int = 30


# ##############################################################################
# # MAIN ROUTINE
# ##############################################################################
if __name__ == "__main__":
    conf = OmegaConf.structured(ConfDef())
    cli_conf = OmegaConf.from_cli()
    conf = OmegaConf.merge(conf, cli_conf)

    assert (
        conf.WITH_TEST_HESSIAN or conf.WITH_TRAIN_HESSIAN
    ), "At least one of (test, train) Hessian must be active!"

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
    model_dtype = params[0].dtype

    # set up optimizer
    opt_class = getattr(hessianoverlap.optimizers, "Scheduled" + conf.OPTIMIZER)
    opt = opt_class(model.parameters(), **opt_hpars, lr_sched=None)

    # prepare hessian globals and dataloaders
    param_shapes = [tuple(p.shape) for p in params]
    num_params = sum([p.numel() for p in params])
    steps_per_epoch = len(train_loader)
    max_steps = (
        conf.MAX_STEPS
        if conf.MAX_STEPS is not None
        else (num_epochs * steps_per_epoch)
    )
    record_steps = {x for x in conf.RECORD_STEPS if x <= max_steps}

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

    # ##########################################################################
    # # MAIN LOOP
    # ##########################################################################
    global_step = 0
    for epoch in range(1, num_epochs + 1):
        if global_step > max_steps:
            break
        # At beg of each epoch, evaluate and log to Cockpit
        model.eval()
        train_loss, train_acc, xv_loss, xv_acc, test_loss, test_acc = eval_fn()
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
                # instantiate Hessian linop
                params = [p for p in model.parameters() if p.requires_grad]
                paramvec = torch.nn.utils.parameters_to_vector(params)
                if conf.WITH_TEST_HESSIAN:
                    hessian_analysis(
                        global_step,
                        "H_TEST",
                        txt_logger,
                        model,
                        loss_fn,
                        params,
                        paramvec,
                        H_test_dl,
                        model_dtype,
                        conf,
                    )

                if conf.WITH_TRAIN_HESSIAN:
                    hessian_analysis(
                        global_step,
                        "H_TRAIN",
                        txt_logger,
                        model,
                        loss_fn,
                        params,
                        paramvec,
                        H_train_dl,
                        model_dtype,
                        conf,
                    )

                # ##############################################################
                # # DONE WITH HESSIANS. CONTINUE TRAINING LOOP
                # ##############################################################
                # clone model
                model_clone = deepcopy(model)
                model_clone.load_state_dict(
                    {k: v.clone() for k, v in model.state_dict().items()}
                )
                model_clone.eval()
                params = list(model_clone.parameters())

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
