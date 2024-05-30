#!/bin/bash

# activate conda env. `conda activate` doesn't work inside scripts
source $WORK/miniconda3/bin/activate hessov

# globals
OBS_DATASETS_PATH="${WORK}/datasets/DeepOBS"
PROBLEM="cifar100_allcnnc"  # NOTE WE ONLY COMPUTE TEST MATRICES
RANDOM_SEED=12345
OUTPUT_DIR="output"
MEASDIR_NAME="measurements"
MEAS_FLAG="measured"
NUM_HESSIAN_DATAPOINTS=1000
MAX_RECORD_STEP=8001
MAX_TRAIN_STEPS=8001
RECORD_BEGINNING=0
RECORD_EVERY=2000  # 100
NUM_OUTER=1000
NUM_INNER=2000
#
MEAS_PER_JOB=340
MEAS_JOB_DURATION="0-02:00"
#
TRAIN_PARTITION="2080-galvani"  # overrides script
OVERLAP_RAM=200
OVERLAP_PARTITION="2080-galvani"
NUM_A_POSTERIORI=20
WITH_A_POSTERIORI=0
#
START_TESTHESSIAN_ITER=1 # NOTE WE ONLY COMPUTE TEST MATRICES

# train NN, gather parameters and gradients into HDF5 and grab job ID
echo ""
echo `date +"[%Y-%m-%d %H:%M:%S]"`
echo "starting training job for" $PROBLEM
echo sbatch --wait --partition=${TRAIN_PARTITION} --export=ALL,OBS_DATASETS_PATH=$OBS_DATASETS_PATH,OUTPUT_DIR=$OUTPUT_DIR,PROBLEM=$PROBLEM,RANDOM_SEED=$RANDOM_SEED,NUM_HESSIAN_DATAPOINTS=$NUM_HESSIAN_DATAPOINTS,MAX_RECORD_STEP=$MAX_RECORD_STEP,RECORD_EVERY=$RECORD_EVERY,RECORD_BEGINNING=$RECORD_BEGINNING,MAX_STEPS=$MAX_TRAIN_STEPS 10a_gather_params_grads.slurm
jobstring_1=$(sbatch --wait --partition=${TRAIN_PARTITION} --export=ALL,OBS_DATASETS_PATH=$OBS_DATASETS_PATH,OUTPUT_DIR=$OUTPUT_DIR,PROBLEM=$PROBLEM,RANDOM_SEED=$RANDOM_SEED,NUM_HESSIAN_DATAPOINTS=$NUM_HESSIAN_DATAPOINTS,MAX_RECORD_STEP=$MAX_RECORD_STEP,RECORD_EVERY=$RECORD_EVERY,RECORD_BEGINNING=$RECORD_BEGINNING,MAX_STEPS=$MAX_TRAIN_STEPS 10a_gather_params_grads.slurm) && \
    echo "training job finished"
jobid_1=${jobstring_1##* }
echo "training job ID:" $jobid_1

# since we waited, we can now parse the training log and get relevant infos
run_dir=`find $OUTPUT_DIR -type d -name "*__$jobid_1"`
meas_dir=${run_dir}/${MEASDIR_NAME}
log_path=`ls ${run_dir}/*10a_gather_params_grads*.log`
num_params=`grep "num_params" $log_path | jq -c ".[1][1].num_params"`
record_steps=`grep "record_steps" $log_path | jq -c ".[1][1].record_steps"`
record_steps_arr=`python -c "print(' '.join(str(x) for x in $record_steps))"`
# we can also get the params, needed for the measurements
params_path=`ls ${run_dir}/*params.h5`
# arr size for a single Hessian is ceil(num_inner / float(meas_per_job))
meas_arrsize=$((($NUM_INNER + ($MEAS_PER_JOB - 1)) / $MEAS_PER_JOB))
echo "processed training log"

# create virtual HDF5 layout
echo ""
echo `date +"[%Y-%m-%d %H:%M:%S]"`
echo "creating virtual HDF5 measurement layout in" $meas_dir
mkdir $meas_dir && \
    python -u -m skerch create_hdf5_layout \
           --hdf5dir $meas_dir \
           --dtype float32 \
           --shape $num_params,$num_params \
           --outer $NUM_OUTER \
           --inner $NUM_INNER \
           --sym  && \
    echo "created virtual HDF5 layout"

# create monolithic HDF5 layout
echo ""
echo `date +"[%Y-%m-%d %H:%M:%S]"`
echo "creating monolithic HDF5 measurement layout in" $meas_dir
outer_all=$meas_dir/`python -c "import skerch as pssvd; print(pssvd.LO_FMT.format('ALL'))"`
outer_merged=$meas_dir/`python -c "import skerch as pssvd; print(pssvd.LO_FMT.format('MERGED'))"`
inner_all=$meas_dir/`python -c "import skerch as pssvd; print(pssvd.INNER_FMT.format('ALL'))"`
inner_merged=$meas_dir/`python -c "import skerch as pssvd; print(pssvd.INNER_FMT.format('MERGED'))"`
python -u -m skerch merge_hdf5 --in_path $outer_all \
       --out_path $outer_merged && \
    python -u -m skerch merge_hdf5 --in_path $inner_all \
           --out_path $inner_merged && \
    echo "created monolithic HDF5 layout"

# we have one train and one test Hessian per step. each Hessian is a single
# array GPU job for the measurements, followed by the big-RAM CPU job.
for step in $record_steps_arr
do
    for((test_hessian=$START_TESTHESSIAN_ITER; test_hessian<2; test_hessian++))
    do
        # job array for random measurements, write to virtual HDF5
        echo ""
        echo `date +"[%Y-%m-%d %H:%M:%S]"`
        echo sbatch --wait --array=1-$meas_arrsize --time=$MEAS_JOB_DURATION --export=ALL,MEAS_ARRSIZE=$meas_arrsize,MEAS_PER_JOB=$MEAS_PER_JOB,OBS_DATASETS_PATH=$OBS_DATASETS_PATH,PARAMS_PATH=$params_path,LOG_PATH=$log_path,MEAS_DIR=$meas_dir,NUM_INNER=$NUM_INNER,NUM_OUTER=$NUM_OUTER,MEAS_FLAG=$MEAS_FLAG,STEP=$step,TEST_HESSIAN=$test_hessian 10c_hessian_hdf5_measurements.slurm
        sbatch --wait --array=1-$meas_arrsize --time=$MEAS_JOB_DURATION --export=ALL,MEAS_ARRSIZE=$meas_arrsize,MEAS_PER_JOB=$MEAS_PER_JOB,OBS_DATASETS_PATH=$OBS_DATASETS_PATH,PARAMS_PATH=$params_path,LOG_PATH=$log_path,MEAS_DIR=$meas_dir,NUM_INNER=$NUM_INNER,NUM_OUTER=$NUM_OUTER,MEAS_FLAG=$MEAS_FLAG,STEP=$step,TEST_HESSIAN=$test_hessian 10c_hessian_hdf5_measurements.slurm

        # single CPU job with large RAM for final analysis
        echo ""
        echo `date +"[%Y-%m-%d %H:%M:%S]"`
        echo sbatch --wait --mem=${OVERLAP_RAM}G --partition=${OVERLAP_PARTITION} --export=ALL,WITH_A_POSTERIORI=$WITH_A_POSTERIORI,OBS_DATASETS_PATH=$OBS_DATASETS_PATH,PARAMS_PATH=$params_path,LOG_PATH=$log_path,MEAS_DIR=$meas_dir,INNER_VIRTUAL=$inner_all,INNER_MONOLITHIC=$inner_merge,OUTER_VIRTUAL=$outer_all,OUTER_MONOLITHIC=$outer_merged,MEAS_FLAG=$MEAS_FLAG,STEP=$step,TEST_HESSIAN=$test_hessian,NUM_A_POSTERIORI=$NUM_A_POSTERIORI 10d_core_and_analysis.slurm
        sbatch --wait --mem=${OVERLAP_RAM}G --partition=${OVERLAP_PARTITION} --export=ALL,WITH_A_POSTERIORI=$WITH_A_POSTERIORI,OBS_DATASETS_PATH=$OBS_DATASETS_PATH,PARAMS_PATH=$params_path,LOG_PATH=$log_path,MEAS_DIR=$meas_dir,INNER_VIRTUAL=$inner_all,INNER_MONOLITHIC=$inner_merged,OUTER_VIRTUAL=$outer_all,OUTER_MONOLITHIC=$outer_merged,MEAS_FLAG=$MEAS_FLAG,STEP=$step,TEST_HESSIAN=$test_hessian,NUM_A_POSTERIORI=$NUM_A_POSTERIORI 10d_core_and_analysis.slurm
   done
done

# wrap up
echo ""
echo "Done! deleting" $meas_dir
rm -rf $meas_dir
