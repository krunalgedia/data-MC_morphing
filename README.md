# data-MC_morphing

'quantileRegression.py' : quantile regression morphing + stochastic morphing class 

'runna.sh' -> 'jobba.sh' -> 'train_quantileRegression_Batch.py' : train conditional quantile (output in 'weights' folder) and conditional class probability functions (output in 'class' folder)

'runna_predict.sh' -> 'jobba_predict.sh' -> 'predict_quantileRegression_Batch.py' : gives corrected MC for each variables (output in 'h5_corr_files' folder)

'merge_h5.py' : merges intial uncorrected MC file with all the correcetd MC files in 'h5_corr_files' folder.

'NN_MC_file.py', 'NN_data_file.py' : prepares tuple for NN regression.

'predict_fit_ffwd_MC_uncorr.py', 'predict_fit_ffwd_MC_corr.py', 'predict_fit_ffwd_data.py' : NN regression code for uncorrected MC, corrected MC and data.

'config_2016_updated_MC_uncorr.json', 'config_2016_updated_MC_corr.json', 'config_2016_updated_data.json' : names of input varaiables required by NN regression by the above files.

-----------------------------------------------------------------------------------------------------------------------------

Notation : sWeighted MC -> MC
           sWeighted data -> data
           MC truth -> MC truth

Initially you have 'qreg_pt_norm.root' which consists of 't_dataset_sig' (MC with weight = 'p_sw_mc_norm' * 'yield_b_sw'), 't_r_dataset_sig' (data with weight = 'p_sw_data_norm' * 'r_yield_b_sw'), MC truth (weight = 'rho_weight').

Step 1 : 'root_to_h5_files.ipynb' - gives training MC ('t_dataset_sig_train.h5'), test MC ('t_dataset_sig_test.h5), training MC truth ('mc_sig_train.h5'), trainig data (t_r_dataset_sig_train.h5) tuples. Use this code only once in the project else you would change you trainig and test sets.

Step 2: 'runna.sh' -> 'jobba.sh' -> 'train_quantileRegression_Batch.py' - Trains the conditional quantile functions for each quantile value per variable for MC, data and MC truth and saves it in 'weights' folder. Also, for stochastic corrections, saves conditional class probabilities function in 'class' folder.

Step 3: 'runna_predict.sh' -> 'jobba_predict.sh' -> 'predict_quantileRegression_Batch.py' - Gives final corrected MC variables (saved in 'h5_corr_files' folder). (Also saves class probabilities, if stochastic case)

Step 4: 'merge_h5.py' - Merges uncorrected MC (t_dataset_sig_test.h5) with all files in 'h5_corr_files' folder to give 't_dataset_sig_test_new.h5'. It also gives 't_dataset_sig_test_plots.h5' where it just computes energy fraction of rings for plots and regression as 't_dataset_sig_test_new.h5' contains raw energy of the rings.

Step 5 : 'NN_MC_file.py', 'NN_data_file.py' - Prepares MC (t_dataset_sig_test_plots_NN.h5) and data (t_r_dataset_sig_train_NN.h5) files for NN regression.

Step 6 : 'predict_fit_ffwd_MC_uncorr.py', 'predict_fit_ffwd_MC_corr.py', 'predict_fit_ffwd_data.py' - Gives NN regression output for uncorrected MC , corrected MC and data. For MC, first run 'predict_fit_ffwd_MC_uncorr.py' and then ''predict_fit_ffwd_MC_corr.py' so that final file has outputs from both uncorrected and corrected MC. Final MC is 'applied_res_2018-08-08_applied_res_2018-08-08_t_dataset_sig_test_plots_NN.h5' while for data is 'applied_res_2018-08-08_t_r_dataset_sig_train_NN.h5'. 

'Save Dataframe with two random numbers.ipynb' : Used only once in the project. Gives dataframe with two random variables (random_numbers.h5) to be used during stochastic morphing.

---------------------------------------------------------------------

For data/MC plots I used: MC (t_dataset_sig_test_plots.h5) and data (t_r_dataset_sig_train.h5) and MC truth (mc_sig_train.h5).

For NN regression plots, I used : MC (applied_res_2018-08-08_applied_res_2018-08-08_t_dataset_sig_test_plots_NN.h5) and data (applied_res_2018-08-08_t_r_dataset_sig_train_NN.h5)

---------------------------------------------------------------------

'final_datasets.ipynb' : For MC, I merged 't_dataset_sig_test_plots.h5' with 'applied_res_2018-08-08_applied_res_2018-08-08_t_dataset_sig_test_plots_NN.h5' to give 't_dataset_sig_test_final.h5' with weight 'weight'. For data I merged 't_r_dataset_sig_train.h5' with 'applied_res_2018-08-08_t_r_dataset_sig_train_NN.h5' with weight 'weight'. For MC truth, we use old file itself i.e. mc_sig_train.h5 with weight 'rho_weight'. 

----------------------------------------------------------------------

Jupyter Notebooks used for plots:

'splots.ipynb' and 'splots-inclusive.ipynb': splots of pt-inclusive and pt-exclusive samples.

'splots_verify.ipynb' and 'splots_verify-inclusive.ipynb': splots verification plots for pt_inclusive and pt_exclusive samples.

'qreg.ipynb' : data/MC morphing plots.

'NN_plots.ipynb' and 'NN_profile_plots.ipynb' : NN regression plots.

'corr_matrix.ipynb' : correlation matrices

'class.ipynb', 'Jet_pt_splot_comparision.ipynb ', 'qreg_morphing_discontinuity.ipynb' : random plots for my thesis.


















 


           
