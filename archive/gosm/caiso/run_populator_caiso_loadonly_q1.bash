#! /bin/bash

export PRESCIENT_VERSION=1.0
export PRESCIENT_PATH=../release/Prescient_${PRESCIENT_VERSION}
export PYTHONPATH=$PRESCIENT_PATH:$PYTHONPATH

python $PRESCIENT_PATH/exec/prescient.py \
       --model-directory=$PRESCIENT_PATH/models/latorre \
       --run-populator \
       --data-to-fit=rolling \
       --cutpoints-names-filename=$PRESCIENT_PATH/controls/cutpoints/cpts_names.dat \
       --single-category-width=0.1 \
       --output-directory=foo \
       --ruc-horizon=48 \
       --scenarios-populate-datesfile=$PRESCIENT_PATH/data/CAISO/caiso_dates_september_2014.dat \
       --pyspgen-base-file=$PRESCIENT_PATH/data/CAISO/generation/CAISO_skeleton48_wecc.dat \
       --L1Linf-solver=gurobi \
       --traceback \
       --loads-dps-cuts-filename=$PRESCIENT_PATH/controls/cutpoints/TailsLimited.dat \
       --loads-input-directory=$PRESCIENT_PATH/data/CAISO/Load \
       --loads-input-filename=caiso_load_070113_063015.csv \
       --load-scaling-factor=1.0 \
       --sources-file=sources_none.csv \
       --pattern-binds \
       --work-with-pattern=absolute 
