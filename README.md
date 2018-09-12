# data-MC_morphing

'quantileRegression.py' : quantile regression morphing + stochastic morphing class 

'runna.sh' -> 'jobba.sh' -> 'train_quantileRegression_Batch.py' : train conditional quantile (output in 'weights' folder) and conditional class probability functions (output in 'class' folder)

'runna_predict.sh' -> 'jobba_predict.sh' -> 'predict_quantileRegression_Batch.py' : gives corrected MC for each variables (output in 'h5_corr_files' folder)

'merge_h5.py' : merges intial uncorrected MC file with all the correcetd MC files in 'h5_corr_files' folder.

'NN_MC_file.py', 'NN_data_file.py' : prepares 
