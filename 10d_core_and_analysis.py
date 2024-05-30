#!/usr/bin/env python
# -*-coding:utf-8-*-


"""
This script merges performed Hessian measurements into the monolithic HDF5
array, and solves the *core sketch* in order to obtain the sketched
eigendecomposition in its final form. Then performs the numerical analysis
(e.g. computing quantities needed to measure the *overlap*, see paper), and
logs them into a file.
"""


import json
import os

# for omegaconf
from dataclasses import dataclass

import numpy as np

import h5py
import torch
from hessianoverlap.incremental_hdf5 import IncrementalHDF5

#
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
from skerch.distributed_decompositions import orthogonalize, solve_core_seigh
from skerch.distributed_measurements import DistributedHDF5
from skerch.linops import CompositeLinOp, DiagonalLinOp, TorchLinOpWrapper
from skerch.ssrft import SSRFT


# ##############################################################################
# # HELPERS
# ##############################################################################
class TorchHessianLinearOperator(TorchLinOpWrapper, HessianLinearOperator):
    pass


class TorchCompositeLinOp(TorchLinOpWrapper, CompositeLinOp):
    pass


def copy_virtual_to_monolithic(
    virtual_path,
    monolithic_path,
    success_flag=None,
    delete_merged_subfiles=False,
):
    """ """
    # open monolithic file to write into
    h5_merged = h5py.File(monolithic_path, "r+")
    # analyze virtual file's chunks
    (
        data_shape,
        data_dtype,
        data_subshapes,
        data_map,
        flags_shape,
        flags_dtype,
        flag_subshapes,
        flag_map,
        filedim_idx,
    ) = DistributedHDF5.analyze_virtual(virtual_path, success_flag)
    data_subshape = data_subshapes[0]
    # iterate over virtual subfiles
    sorted_data = sorted(data_map, key=lambda x: x[filedim_idx][0])
    for begs_ends in sorted_data:
        # grab subfile data and check flag
        subpath = data_map[begs_ends]
        subdata, subflag, h5 = DistributedHDF5.load(subpath, filemode="r")
        if success_flag is not None:
            assert (
                subflag[0].decode() == success_flag
            ), f"Subfile flag not equal {success_flag}!"
        # write subdata and flag to h5f, flush and close subfile
        flag_idx = begs_ends[filedim_idx][0]
        target_slices = tuple(slice(*be) for be in begs_ends)
        h5_merged[DistributedHDF5.DATA_NAME][target_slices] = subdata[
            :
        ].reshape(data_subshape)
        h5_merged[DistributedHDF5.FLAG_NAME][flag_idx : flag_idx + 1] = subflag[
            :
        ]
        h5.close()
        # optionally, delete subfile
        if delete_merged_subfiles:
            os.remove(subpath)
    # done!
    h5_merged.close()


def truncated_a_posteriori(
    M, Q, core_U, core_S, n, seed, dtype, device, trunc_dims=None
):
    """
    :param n: Number of a-posteriori measurements
    """
    dims = len(core_S) if trunc_dims is None else trunc_dims
    M_lowrank = TorchCompositeLinOp(
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


# ##############################################################################
# # CLI
# ##############################################################################
@dataclass
class ConfDef:
    """
    :cvar STEP: Since we do post-hoc error estimation, and the original matrix
      uses the network parameters, we need to know the step.
    :cvar TEST_HESSIAN: Since we do post-hoc error estimation, and the original
      matrix uses a dataloader, we need to know which split is being used.
      True for test split, false for train split.
    """

    # to reproduce DL linop
    OBS_DATASETS: str = MISSING  # /shared/datasets/DeepOBS
    PARAMS_PATH: str = MISSING
    TEST_HESSIAN: bool = False
    STEP: int = MISSING
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    TRAINING_LOGPATH: str = MISSING
    #
    INNER_VIRTUAL: str = MISSING
    OUTER_VIRTUAL: str = MISSING
    INNER_MONOLITHIC: str = MISSING
    OUTER_MONOLITHIC: str = MISSING
    #
    SUCCESS_FLAG: str = "measured"
    #
    WITH_A_POSTERIORI: bool = True
    NUM_A_POSTERIORI: int = 30
    DELETE_MERGED_SUBFILES: bool = False


# ##############################################################################
# # MAIN ROUTINE
# ##############################################################################
if __name__ == "__main__":
    conf = OmegaConf.structured(ConfDef())
    cli_conf = OmegaConf.from_cli()
    conf = OmegaConf.merge(conf, cli_conf)

    # load log
    log_conf = JsonColorLogger.read_file(
        conf.TRAINING_LOGPATH, filter_headers={"CONFIG"}
    )
    log_glob = JsonColorLogger.read_file(
        conf.TRAINING_LOGPATH, filter_headers={"GLOBALS"}
    )
    log_h5 = JsonColorLogger.read_file(
        conf.TRAINING_LOGPATH, filter_headers={"HDF5"}
    )
    run_dir = log_glob["run_dir"][0]
    problem = log_conf["PROBLEM"][0]
    batch_size = int(log_glob["batch_size"][0])
    record_steps = set(log_h5["global_step"])
    num_hessian_datapoints = int(log_conf["NUM_HESSIAN_DATAPOINTS"][0])
    random_seed = int(log_conf["RANDOM_SEED"][0])
    li_seed, ri_seed = random_seed + 1, random_seed + 2
    num_params = int(log_glob["num_params"][0])

    # create log for this process, inside of same folder
    timestamp = make_timestamp(timezone="Europe/Berlin", with_tz_output=False)
    txt_logger = JsonColorLogger(
        f"[{os.path.basename(__file__)}]"
        + f"__step={conf.STEP}__test={conf.TEST_HESSIAN}",
        run_dir,
    )
    txt_logger.loj("CONFIG", OmegaConf.to_container(conf))

    # ##########################################################################
    # # GETTING THE HESSIAN (for a-posteriori error and scree)
    # ##########################################################################
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
    paramvec, _ = IncrementalHDF5.get_element(
        params_h5, step_to_h5[conf.STEP], with_data=True
    )
    paramvec = torch.from_numpy(paramvec.flatten())
    torch.nn.utils.vector_to_parameters(paramvec, model.parameters())
    model = model.to(conf.DEVICE)
    params_h5.close()

    # instantiate Hessian linop
    params = [p for p in model.parameters() if p.requires_grad]
    H = TorchHessianLinearOperator(model, loss_fn, params, H_dl)

    # ##########################################################################
    # # COPY VIRTUAL HDF5 CONTENTS INTO MONOLITHIC
    # ##########################################################################
    txt_logger.loj("INFO", ["Checking inner flags and copying to monolithic"])
    try:
        copy_virtual_to_monolithic(
            conf.INNER_VIRTUAL,
            conf.INNER_MONOLITHIC,
            conf.SUCCESS_FLAG,
            conf.DELETE_MERGED_SUBFILES,
        )
        txt_logger.loj(
            "INFO", ["Checking outer flags and copying to monolithic"]
        )
        copy_virtual_to_monolithic(
            conf.OUTER_VIRTUAL,
            conf.OUTER_MONOLITHIC,
            conf.SUCCESS_FLAG,
            conf.DELETE_MERGED_SUBFILES,
        )
        txt_logger.loj("INFO", ["Done copying to monolithic"])
    except AssertionError as ae:
        txt_logger.loj(
            "ERROR", [f"Exception during copying to monolithic! {ae}"]
        )

    # ##########################################################################
    # # FINALIZING EIGENDECOMPOSITION
    # ##########################################################################

    # load all merged components and check that all flags are OK
    outr, outr_flags, o_h5 = DistributedHDF5.load(
        conf.OUTER_MONOLITHIC, filemode="r"
    )
    innr, innr_flags, i_h5 = DistributedHDF5.load(
        conf.INNER_MONOLITHIC, filemode="r"
    )
    num_innr, num_outr = len(innr_flags), len(outr_flags)

    try:
        assert all(
            f.decode() == conf.SUCCESS_FLAG for f in outr_flags
        ), "Bad outer flags!"
        assert all(
            f.decode() == conf.SUCCESS_FLAG for f in innr_flags
        ), "Bad inner flags!"
    except AssertionError as ae:
        txt_logger.loj("ERROR", [f"Bad flags in monolithic! {ae}"])
        # breakpoint()

    assert innr.shape == (num_innr, num_innr), "Unexpected inner shape!"
    assert outr.shape == (num_params, num_outr), "Unexpected outer shape!"
    # orthogonalize outer measurements (not in-place ATM, maybe pytables?)
    txt_logger.loj("INFO", [f"Orthogonalizing outer measurements {outr.shape}"])
    Q = orthogonalize(outr, overwrite=False)
    assert np.shares_memory(Q, Q.T), "Transposing creates a copy?"
    o_h5.close()
    # Solve core op and decompose via eigh
    txt_logger.loj("INFO", ["Solving SEIGH core"])
    li_ssrft = SSRFT((num_innr, num_params), seed=li_seed)
    ri_ssrft = SSRFT((num_innr, num_params), seed=ri_seed)
    core_U, core_S = solve_core_seigh(Q, innr, li_ssrft, ri_ssrft)
    i_h5.close()
    txt_logger.loj("RESULT", {"core_S": core_S.tolist()})

    # ##########################################################################
    # #  ESTIMATE A-POSTERIORI ERROR AND SCREE BOUNDS
    # ##########################################################################
    if conf.WITH_A_POSTERIORI and (conf.NUM_A_POSTERIORI > 0):
        txt_logger.loj(
            "INFO", ["A-posteriori error estimation and scree bounds"]
        )
        trunc_50 = int(np.ceil(0.5 * num_outr))
        trunc_05 = int(np.ceil(0.05 * num_outr))
        f1, f2, res = truncated_a_posteriori(
            H,
            Q,
            core_U,
            core_S,
            conf.NUM_A_POSTERIORI,
            random_seed + 2,
            params[0].dtype,
            conf.DEVICE,
            trunc_dims=num_outr,
        )
        _, f2_50, res_50 = truncated_a_posteriori(
            H,
            Q,
            core_U,
            core_S,
            conf.NUM_A_POSTERIORI,
            random_seed + 2,
            params[0].dtype,
            conf.DEVICE,
            trunc_dims=trunc_50,
        )
        _, f2_05, res_05 = truncated_a_posteriori(
            H,
            Q,
            core_U,
            core_S,
            conf.NUM_A_POSTERIORI,
            random_seed + 2,
            params[0].dtype,
            conf.DEVICE,
            trunc_dims=trunc_05,
        )
        txt_logger.loj(
            "RESULT",
            {
                "info": (
                    "Frob^2 of Hessian, corresponding k-rank reconstruction "
                    + "H_k and residual (H - H_k)"
                ),
                "Frob(H)^2": f1,
                f"Frob(H_{num_outr})^2": f2,
                f"Frob(H_{trunc_50})^2": f2_50,
                f"Frob(H_{trunc_05})^2": f2_05,
                f"Frob(H-H_{num_outr})^2": res,
                f"Frob(H-H_{trunc_50})^2": res_50,
                f"Frob(H-H_{trunc_05})^2": res_05,
            },
        )

        txt_logger.loj(
            "RESULT",
            {
                **a_posteriori_error_bounds(
                    conf.NUM_A_POSTERIORI, rel_err=0.25
                ),
                **a_posteriori_error_bounds(conf.NUM_A_POSTERIORI, rel_err=0.5),
                **a_posteriori_error_bounds(conf.NUM_A_POSTERIORI, rel_err=1),
            },
        )
        #
        lo_scree, hi_scree = scree_bounds(
            torch.from_numpy(core_S), f1**0.5, res**0.5
        )
        txt_logger.loj(
            "RESULT",
            {
                "info": "Scree lower and upper bounds by increasing k",
                "scree_lower": lo_scree.tolist(),
                "scree_upper": hi_scree.tolist(),
            },
        )

    # ##########################################################################
    # # OVERLAP GRASSMANNIAN METRIC
    # ##########################################################################
    frob_by_mag = []
    for k in range(1, num_outr + 1):
        mask_idxs = paramvec.abs().topk(k, largest=True, sorted=False).indices
        # our reconstructed eigenbasis is W = (Q @ core_U)
        # we want to extract W[mask_idxs, :k] = (Q[mask_idxs] @ core_U)
        frob = ((Q[mask_idxs] @ core_U[:, :k]) ** 2).sum().item()
        frob_by_mag.append(frob)
    txt_logger.loj(
        "RESULT",
        {
            "frob_Qtop^2": frob_by_mag,
            "info": "Frob(Q[largest_absparams,:k])^2 ordered by increasing k",
        },
    )
    txt_logger.loj(
        "RESULT",
        {
            "largest_absparams": mask_idxs.tolist(),
            "info": "Indices of k largest parameters, by descending magnitude",
        },
    )
