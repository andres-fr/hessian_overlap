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
#SBATCH --job-name="hessov_v2.0_12b"


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
#SBATCH --partition=2080-galvani  # sinfo -s
#SBATCH --time=0-10:00  # D-HH:MM
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G


# ##############################################################################
# # MAIN ROUTINE
# # note the following array syntax:
# # --array=0-9  --> 10 jobs, from 0 to 9 (both included)
# # --array=4,7,123  --> 3 jobs with numbers 4, 7 and 123
# # --array=0-4:2  --> 3 jobs with indices 0,2 and 4
# #
# # If you call sbatch via "arg1=val1 arg2=val2 sbatch test.sh", you can then
# # access those variables here via "./myprog $arg1 $arg2"
# ##############################################################################


RECORD_BEGINNING=0
RECORD_EVERY=100
MAX_RECORD_STEP=1000
SEQUENCE="["`seq -s, $RECORD_BEGINNING $RECORD_EVERY $MAX_RECORD_STEP`"]"

SUFFIX="$SLURM_JOB_ID"

python -u 12a_mnist_mini_exhaustive_test.py \
       OBS_DATASETS="${WORK}/datasets/DeepOBS" \
       TUNING_CONF="config/basic_config.yaml" \
       OPTIMIZER="SGD" \
       PROBLEM="mnist_mini" \
       RANDOM_SEED=12345 \
       OUTPUT_DIR="output" \
       OUTDIR_SUFFIX=$SUFFIX \
       RECORD_STEPS="$SEQUENCE" \
       MAX_STEPS=16001 \
       NUM_HESSIAN_DATAPOINTS=500 \
       NUM_OUTER=355 \
       NUM_INNER=710 \
       WITH_A_POSTERIORI=0
