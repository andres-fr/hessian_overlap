#!/usr/bin/env python
# -*-coding:utf-8-*-


"""
This script loads the results from ``10a_gather_params_grads.py``, i.e. the
recorded parameters and logged metadata from a DeepOBS problem, and computes
the Hessian-Vector-Product ``H_t @ Phi_i``, where ``H_t`` is the Hessian at
training step ``t``, and ``Phi_i`` is the ``ith`` column of the noisy
measurement operator ``Phi``.

Results are written into a ``MEASUREMENTS_DIR`` distributed HDF5 array, that
has been previously created via ``10b_prepare_hdf5_measurements.slurm``.

This interface allows to naturally distribute the computation of Hessians
across different devices. Usage example: see homonymous slurm script.
"""


import json
import os
from dataclasses import dataclass

# for omegaconf
from typing import List

import h5py
import torch
from hessianoverlap.incremental_hdf5 import IncrementalHDF5
from hessianoverlap.logging import JsonColorLogger, make_timestamp
from hessianoverlap.obs_extensions import get_obs_testproblem
from hessianoverlap.obs_extensions.samplers import SubsetSampler
from curvlinops import HessianLinearOperator

#
from deepobs.config import set_data_dir

#
from omegaconf import MISSING, OmegaConf
from skerch import INNER_FMT, LO_FMT
from skerch.distributed_measurements import DistributedHDF5, innermeas_idx_torch
from skerch.linops import TorchLinOpWrapper


# ##############################################################################
# # HELPERS
# ##############################################################################
class TorchHessianLinearOperator(TorchLinOpWrapper, HessianLinearOperator):
    pass


# ##############################################################################
# # CLI
# ##############################################################################
@dataclass
class ConfDef:
    """
    :cvar TEST_HESSIAN: If True, the Hessian is computed on the test subsplit,
      otherwise the train subsplit is used.
    """

    # specify these, the rest will be grabbed from the log
    OBS_DATASETS: str = MISSING  # /shared/datasets/DeepOBS
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    #
    PARAMS_PATH: str = MISSING
    TRAINING_LOGPATH: str = MISSING
    MEASUREMENTS_DIR: str = MISSING
    NUM_INNER: int = MISSING
    NUM_OUTER: int = MISSING
    SUCCESS_FLAG: str = "measured"
    #
    TEST_HESSIAN: bool = False
    STEP: int = MISSING
    MEASUREMENT_IDXS: List[int] = MISSING


# ##############################################################################
# # MAIN ROUTINE
# ##############################################################################
if __name__ == "__main__":
    conf = OmegaConf.structured(ConfDef())
    cli_conf = OmegaConf.from_cli()
    conf = OmegaConf.merge(conf, cli_conf)

    # load log
    log_batch = JsonColorLogger.read_file(
        conf.TRAINING_LOGPATH, filter_headers={"TRAINING BATCH"}
    )
    log_eval = JsonColorLogger.read_file(
        conf.TRAINING_LOGPATH, filter_headers={"EVAL ROUND", "FINAL EVALUATION"}
    )
    log_h5 = JsonColorLogger.read_file(
        conf.TRAINING_LOGPATH, filter_headers={"HDF5"}
    )
    log_conf = JsonColorLogger.read_file(
        conf.TRAINING_LOGPATH, filter_headers={"CONFIG"}
    )
    log_glob = JsonColorLogger.read_file(
        conf.TRAINING_LOGPATH, filter_headers={"GLOBALS"}
    )
    #
    run_dir = log_glob["run_dir"][0]
    problem = log_conf["PROBLEM"][0]
    batch_size = int(log_glob["batch_size"][0])
    record_steps = set(log_h5["global_step"])
    num_hessian_datapoints = int(log_conf["NUM_HESSIAN_DATAPOINTS"][0])
    random_seed = int(log_conf["RANDOM_SEED"][0])
    li_seed, ri_seed = random_seed + 1, random_seed + 2
    H_train_idxs = log_glob["H_train_idxs"][0]
    H_test_idxs = log_glob["H_test_idxs"][0]
    num_params = int(log_glob["num_params"][0])
    # load parameters, and map from step to h5 idx
    step_to_h5 = {}
    params_h5 = h5py.File(conf.PARAMS_PATH, "r")
    for i in range(IncrementalHDF5.get_num_elements(params_h5)):
        _, meta = IncrementalHDF5.get_element(
            params_h5, i, with_data=False, with_metadata=True
        )
        meta = json.loads(meta.replace("'", '"'))
        step = meta["step"]
        assert (
            step in record_steps
        ), "Index mismatch between stored params and log!"
        step_to_h5[step] = i
    # also check that required step is available
    assert conf.STEP in record_steps, f"Step {conf.STEP} not available!"
    # create log for this process, inside of same folder
    timestamp = make_timestamp(timezone="Europe/Berlin", with_tz_output=False)

    abbrev_idxs = "...".join(
        str(idx)
        for idx in conf.MEASUREMENT_IDXS[0:1] + conf.MEASUREMENT_IDXS[-1:]
    )
    txt_logger = JsonColorLogger(
        f"[{os.path.basename(__file__)}]__idxs=<{abbrev_idxs}>", run_dir
    )
    txt_logger.loj("CONFIG", OmegaConf.to_container(conf))
    txt_logger.loj(
        "GLOBALS",
        {
            "run_dir": run_dir,
            "problem": problem,
            "batch_size": batch_size,
            "record_steps": sorted(record_steps),
            "num_hessian_datapoints": num_hessian_datapoints,
            "random_seed": random_seed,
            "li_seed": li_seed,
            "ri_seed": ri_seed,
            "H_train_idxs": H_train_idxs,
            "H_test_idxs": H_test_idxs,
            "num_params": num_params,
            "step_to_h5": step_to_h5,
        },
    )

    # Avoid re-downloading dataset on local dir
    set_data_dir(conf.OBS_DATASETS)

    # Set up problem
    txt_logger.loj("INFO", ["Setting up problem..."])
    (
        model,
        train_loader,
        loss_fn,
        losses_fn,
        eval_fn,
        tproblem,
    ) = get_obs_testproblem(problem, batch_size, random_seed)
    model.eval()  # for deterministic behavior

    # prepare Hessian dataloader (fixed, deterministic):
    txt_logger.loj("INFO", ["Preparing Hessian dataloader..."])
    if conf.TEST_HESSIAN:
        H_dataset = tproblem.data._make_test_dataloader().dataset
    else:
        H_dataset = tproblem.data._make_train_and_valid_dataloader()[0].dataset
    H_sampler = SubsetSampler.get_balanced(
        H_dataset, random=False, size=num_hessian_datapoints
    )
    H_dl = torch.utils.data.DataLoader(
        H_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=H_sampler,
    )

    # load parameters from HDF5 for this step
    txt_logger.loj("INFO", ["Loading model parameters and creating Hessian..."])
    data, _ = IncrementalHDF5.get_element(
        params_h5, step_to_h5[conf.STEP], with_data=True
    )
    data = torch.from_numpy(data.flatten())
    torch.nn.utils.vector_to_parameters(data, model.parameters())
    model = model.to(conf.DEVICE)

    # instantiate Hessian linop
    params = [p for p in model.parameters() if p.requires_grad]
    H = TorchHessianLinearOperator(model, loss_fn, params, H_dl)

    # figure out paths for distributed HDF5 inner+outer elements
    for m_idx in conf.MEASUREMENT_IDXS:
        idx_str = DistributedHDF5.SUBPATHS_FORMAT.format(m_idx)
        inner_subpath = os.path.join(
            conf.MEASUREMENTS_DIR, INNER_FMT.format(idx_str)
        )
        outer_subpath = os.path.join(
            conf.MEASUREMENTS_DIR, LO_FMT.format(idx_str)
        )
        # perform inner (and possibly recycled outer) random measurement
        txt_logger.loj(
            "INFO",
            [
                "Performing measurement",
                {
                    "step": conf.STEP,
                    "idx": m_idx,
                    "meas_idx": m_idx,
                    "H_test": conf.TEST_HESSIAN,
                },
            ],
        )
        inn_vals, inn_flag, inn_h5 = DistributedHDF5.load(inner_subpath)
        out_buff = innermeas_idx_torch(
            m_idx,
            conf.NUM_INNER,
            H,
            "cpu",  # Hessian is computed in same device as the model anyway.
            torch.float32,
            inn_vals,
            li_seed,
            ri_seed,
            flag=inn_flag,
            processing_flag="processing measurement...",
            success_flag=conf.SUCCESS_FLAG,
        )
        # check successful flags
        assert (
            inn_flag[0].decode() == conf.SUCCESS_FLAG
        ), f"Bad inner measurement? {i}"
        inn_h5.close()
        # recycle outer measurements
        if m_idx < conf.NUM_OUTER:
            out_vals, out_flag, out_h5 = DistributedHDF5.load(outer_subpath)
            out_vals[:] = out_buff
            out_flag[0] = conf.SUCCESS_FLAG
            assert (
                out_flag[0].decode() == conf.SUCCESS_FLAG
            ), f"Bad outer measurement? {i}"
            out_h5.close()
        # measurements are done, close remaining open files and finish
        params_h5.close()
        txt_logger.loj(
            "INFO",
            [
                "Measurement successful!",
                {
                    "with_outer": m_idx < conf.NUM_OUTER,
                    "step": conf.STEP,
                    "idx": m_idx,
                    "meas_idx": m_idx,
                    "H_test": conf.TEST_HESSIAN,
                },
            ],
        )
