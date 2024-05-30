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
#SBATCH --job-name="hessov_v2.0_10c"


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
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G


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

# array is being called from script to use variables and avoid
# "Invalid job array specification" error

# note this loop does measurements for A SINGLE HESSIAN at given $STEP, and
# for a given data split (boolean $TEST_HESSIAN). Each runner in this for loop
# takes care of a subset of measurements, and saves them to HDF5.
for((i=1; i<=$MEAS_ARRSIZE; i++))
do
    # Each array task loops for the whole range, but only one
    # particular combination gets executed when matching TASK_ID
    if [ "$SLURM_ARRAY_TASK_ID" -eq "$i" ]
    then
        beg=$((($i - 1) * $MEAS_PER_JOB))
        end=$(($beg + $MEAS_PER_JOB))
        meas_idxs="["`python -c "print(','.join(str(x) for x in range($beg, min($NUM_INNER, $end))))"`"]"
        python -u 10c_hessian_hdf5_measurements.py \
             OBS_DATASETS=$OBS_DATASETS_PATH \
             PARAMS_PATH=$PARAMS_PATH \
             TRAINING_LOGPATH=$LOG_PATH \
             MEASUREMENTS_DIR=$MEAS_DIR \
             MEASUREMENT_IDXS=$meas_idxs \
             NUM_INNER=$NUM_INNER \
             NUM_OUTER=$NUM_OUTER \
             SUCCESS_FLAG=$MEAS_FLAG \
             STEP=$STEP \
             TEST_HESSIAN=$TEST_HESSIAN
    fi
done
