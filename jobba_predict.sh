#!/bin/bash
#
echo SWITCH OFF DISPLAY
#export DISPLAY=
echo DISPLAY = $DISPLAY

source /mnt/t3nfs01/data01/shome/krgedia/pyenv.sh

echo JOBBA: Predicting on $1 present in file $2 and tree is $3 for Y = $4  

python /mnt/t3nfs01/data01/shome/krgedia/Workspace/lxplus/qreg_pt_norm/predict_quantileRegression_Batch.py $1 $2 $3 $4 

echo JOBBA Done: Predicting on $1 present in file $2 and tree is $3 for Y = $4
