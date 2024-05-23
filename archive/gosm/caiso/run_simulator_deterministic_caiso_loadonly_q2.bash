#! /bin/bash

export PRESCIENT_VERSION=1.0
export PRESCIENT_PATH=../release/Prescient_${PRESCIENT_VERSION}
export PYTHONPATH=$PRESCIENT_PATH:$PYTHONPATH

python $PRESCIENT_PATH/exec/prescient.py \
       --run-simulator \
       --data-directory=caiso_simple_loadonly_december_2014 \
       --model-directory=$PRESCIENT_PATH/models/latorre \
       --output-directory=caiso_simple_loadonly_deterministic_december_2014 \
       --run-deterministic-ruc \
       --start-date=11-26-2014 \
       --num-days=36 \
       --traceback \
       --random-seed=10 \
       --output-sced-initial-conditions \
       --output-sced-demands \
       --output-sced-solutions \
       --output-ruc-initial-conditions \
       --output-ruc-solutions \
       --output-ruc-dispatches \
       --relax-ramping-if-infeasible \
       --ruc-mipgap=0.0001 \
       --sced-mipgap=0.0 \
       --reserve-factor=0.05 \
       --solver=gurobi \
       --deterministic-ruc-solver-options="RINS=100 Heuristics=0.40 Threads=4 TimeLimit=1800" \
       --output-solver-logs \
       --write-sced-instances \
       --simulate-out-of-sample
