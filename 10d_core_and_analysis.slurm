#!/bin/bash


# ##############################################################################
# # Slurm submission file, allows to run large arrays of jobs defined via
# # nested loops (see bottom)
# #
# # Note that SLURM will stop reading SBATCH directives as soon as there
# # is executable code (this includes defining variables and any non-comment),
# # so make sure to leave that for the end.
# # More info: https://slurm.schedmd.com/sbatch.html
# ##############################################################################

# # Job identifier
#SBATCH --job-name="hessov_v2.0_10d"


# ##############################################################################
# # LOGGING
# ##############################################################################

# # Notify via email
# # https://slurm.schedmd.com/sbatch.html#OPT_mail-type
# # NONE,BEGIN,END,FAIL,ALL,REQUEUE,TIME_LIMIT,TIME_LIMIT_80, ARRAY_TASKS
#SBATCH --mail-type=END,REQUEUE,TIME_LIMIT_80
#SBATCH --mail-user=[REDACTED]

# # output pathing for logs. The directory must exist, otherwise no logs
# # More info: https://slurm.schedmd.com/sbatch.html#lbAH

#SBATCH --error  [REDACTED]
#SBATCH --output [REDACTED]


# ##############################################################################
# # RESOURCES
# # Overview: https://stackoverflow.com/a/54845007
# # The following info can be useful to decide what to request
# # list partitions: `sinfo -s`
# # partition details: `scontrol show partitions`
# # partition info example: `sinfo --long --partition=gpu-2080ti`
# # further inspection: `scontrol show partition gpu-2080ti-interactive`
# # States are:
# #   * alloc (allocated to one or more jobs)
# #   * mix (some of its CPUs allocated, others are IDLE)
# #   * idle (no jobs allocated)
# #   * down/drain/draining (no touchy)
# # check `man sinfo` for comprehensive info on nodes
# #
# # Also check defaults for a given partition, e.g.:
# # general max defaults: scontrol show config | grep -i "max"
# # partition defaults: scontrol show partition gpu-2080ti
# #
# # Billing weights: scontrol show config | grep -i "priorityweight"
# ##############################################################################

# fixed
# ntasks is hardcoded to 1. Use arrays for parallelism
# 1 node means all cores on same machine
#SBATCH --nodes=1
#SBATCH --ntasks=1

# change these
# DUE TO VERY FEW CPU NODES, PARTITION IS PARAMETRIZED OUTSIDE
#SBATCH --time=0-24:00  # D-HH:MM
#SBATCH --cpus-per-task=16
# DUE TO POTENTIALLY LARGE REQUIREMENTS, MEMORY IS PARAMETRIZED OUTSIDE


# #############################################################################
# # MAIN ROUTINE
# #############################################################################
python -u 10d_core_and_analysis.py \
       OBS_DATASETS=$OBS_DATASETS_PATH \
       PARAMS_PATH=$PARAMS_PATH \
       TRAINING_LOGPATH=$LOG_PATH \
       INNER_VIRTUAL=$INNER_VIRTUAL \
       OUTER_VIRTUAL=$OUTER_VIRTUAL \
       INNER_MONOLITHIC=$INNER_MONOLITHIC \
       OUTER_MONOLITHIC=$OUTER_MONOLITHIC \
       SUCCESS_FLAG=$MEAS_FLAG \
       STEP=$STEP \
       TEST_HESSIAN=$TEST_HESSIAN \
       NUM_A_POSTERIORI=$NUM_A_POSTERIORI \
       WITH_A_POSTERIORI=$WITH_A_POSTERIORI
