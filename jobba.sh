#!/bin/bash
#
echo SWITCH OFF DISPLAY
#export DISPLAY=
echo DISPLAY = $DISPLAY

source /mnt/t3nfs01/data01/shome/krgedia/pyenv.sh

echo JOBBA: Training on $1 present in file $2 and tree is $3 for Y = $4 , quantile = $5, maxDepth = $6, minLeaf = $7

python /mnt/t3nfs01/data01/shome/krgedia/Workspace/lxplus/qreg_pt_norm/train_quantileRegression_Batch.py $1 $2 $3 $4 $5 $6 $7 

echo JOBBA Done: Training on $1 present in file $2 and tree is $3 for Y = $4 , quantile = $5, maxDepth = $6, minLeaf = $7
#echo JOBBA DONE: Training on $1 for Y = $2 , quantile = $3,  startEvt = $4, stopEvt = $5, maxDepth = $6, minLeaf = $7, EBEE = $8
