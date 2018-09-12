# # Characterize FFWD NN output
#load libs
import keras.models
import os
#import bregnn.io as io
## import matplotlib.pyplot as plt
import sys
import json
import datetime
from optparse import OptionParser, make_option
#sys.path.insert(0, '/users/nchernya/HHbbgg_ETH/bregression/python/')
#import plotting_utils as plotting
import pandas as pd

parser = OptionParser(option_list=[
    make_option("--training",type='string',dest="training",default='HybridLoss'),
    make_option("--inp-dir",type='string',dest="inp_dir",default='/mnt/t3nfs01/data01/shome/krgedia/Workspace/lxplus/qreg_pt_norm/'),
    make_option("--target-dir",type='string',dest="target_dir",default='/mnt/t3nfs01/data01/shome/krgedia/breg_training_model_2016_updated.hdf5'),
    make_option("--inp-file",type='string',dest='inp_file',default='t_r_dataset_sig_train_NN.h5'),
    make_option("--out-dir",type='string',dest="out_dir",default='/mnt/t3nfs01/data01/shome/krgedia/Workspace/lxplus/qreg_pt_norm/'),
])

## parse options
(options, args) = parser.parse_args()
input_trainings = options.training.split(',')

# ## Read test data and model
# load data
data = pd.read_hdf('%s%s'%(options.inp_dir,options.inp_file))#,columns=None)

'''
if not ('Jet_pt_raw' in data.columns):
    data['Jet_pt_raw'] = data['Jet_pt']*data['Jet_rawEnergy']/data['Jet_e']
    data['Jet_mt_raw'] = data['Jet_mt']*data['Jet_rawEnergy']/data['Jet_e']
    data['Jet_mass']=data['Jet_mass']*data['Jet_rawEnergy']/data['Jet_e']
    data['Jet_leptonPtRelInv']=data['Jet_leptonPtRelInv']*data['Jet_rawEnergy']/data['Jet_e']
'''

#for idx,name in enumerate(input_trainings):
    # list all model files in the training folder
#    target='/users/nchernya/HHbbgg_ETH/bregression/notebooks/'+input_trainings[idx]
target=options.target_dir
  #  target=options.target_dir
imodel = '/mnt/t3nfs01/data01/shome/krgedia/breg_training_model_2016_updated.hdf5'
## print type(imodel) 

    # read training configuration
import json
#    with open('%s/config_2016_updated.json' % target) as fin:
with open('/mnt/t3nfs01/data01/shome/krgedia/Workspace/lxplus/qreg_pt_norm/config_2016_updated_data.json') as fin:
    config = json.loads(fin.read())
config
  
    # ## Compute predictions
features = config['options']['features'].split(',')
    #for i,f in enumerate(features): 
    #    if f == 'Jet_pt' or f == 'Jet_mt'  : features[i] = features[i]+'_raw'

X = data[features].values
print(X)
print "I just printed X[features]"
    
model = keras.models.load_model(imodel,compile=False)
y_pred = model.predict(X)
  
    # *Note*: the target is typically normalized in the training y = (y-mu)/sigma
    # ## Convert raw prediction into actual scale and resolution estimation

if y_pred.shape[1] == 1: # with one output we only have a scale correction
    corr = y_pred
    res = None
elif y_pred.shape[1] == 2: # with two outputs first is mean and second is sigma
    corr = y_pred[:,0]
    res = y_pred[:,1]
elif y_pred.shape[1] == 3: # assume that 3 outputs are mean + 2 quantile
    corr = y_pred[:,0]
    res = 0.5*(y_pred[:,2] - y_pred[:,1])

    # normalize back to energy scale
if config['options']['normalize_target']:
    corr *= config['y_std']
    corr += config['y_mean']
    
    if res is not None:
        res *= config['y_std']
  
# Add new prediction to data frame
data = data.assign(newNNreg=y_pred[:,0])
data = data.rename(columns={'newNNreg':'Jet_pt_reg_NN'})
if res is not None :
    data = data.assign(newNNreg_res=res)
    data = data.rename(columns={'newNNreg_res':'Jet_resolution_NN'})

print(data.columns,data.shape)

# save dataframe with added corrections
now = datetime.datetime.now()
now = str(now).split(' ')[0] 
#outfilename = options.out_dir + 'applied_res_%s_'%str(now) + options.inp_file
outfile =  options.out_dir + 'applied_res_%s_'%str(now) + options.inp_file
data.to_hdf(outfile,'hdf',mode='w')

