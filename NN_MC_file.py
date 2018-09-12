import numpy as np
import matplotlib.pyplot as plt
from root_numpy import root2array, tree2array
from root_numpy import fill_hist
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import root_pandas as rpd
from   root_pandas import read_root
import time
import pickle
import gzip
import bisect
# from joblib import Parallel, delayed
from sklearn.externals.joblib import Parallel, parallel_backend, register_parallel_backend
from joblib import delayed
import os
import ROOT as rt
import copy as cp
from sklearn.ensemble import RandomForestRegressor
from ROOT import TLorentzVector


h5_dir = '/mnt/t3nfs01/data01/shome/krgedia/Workspace/lxplus/qreg_pt_norm/t_dataset_sig_test_plots'

mc =  pd.read_hdf(h5_dir+'.h5', 'hdf')

df=mc

df['isEle']=np.zeros(df.shape[0])
df['isMu']=np.zeros(df.shape[0])
df['isOther']=np.zeros(df.shape[0])

for i in range(0, len(df)):

    jl_pdgid = abs(df.loc[i,'Jet_leptonPdgId'])

    if (jl_pdgid==11): 
        df.loc[i,'isEle'] = 1
              
    elif (jl_pdgid == 13.0):
        df.loc[i,'isMu'] = 1
            
    else: 
        df.loc[i,'isOther'] = 1

    df.loc[i,'Jet_vtxMass'] = max(0,df.loc[i,'Jet_vtxMass'])
    df.loc[i,'Jet_vtxMass_corr'] = max(0,df.loc[i,'Jet_vtxMass_corr'])
    df.loc[i,'Jet_vtxPt'] = max(0,df.loc[i,'Jet_vtxPt'])
    df.loc[i,'Jet_vtxPt_corr'] = max(0,df.loc[i,'Jet_vtxPt_corr'])
    df.loc[i,'Jet_vtx3DSig'] = max(0,df.loc[i,'Jet_vtx3DSig'])
    df.loc[i,'Jet_vtx3DSig_corr'] = max(0,df.loc[i,'Jet_vtx3DSig_corr'])
    df.loc[i,'Jet_vtx3DVal'] = max(0,df.loc[i,'Jet_vtx3DVal'])
    df.loc[i,'Jet_vtx3DVal_corr'] = max(0,df.loc[i,'Jet_vtx3DVal_corr'])
    df.loc[i,'Jet_leptonDeltaR'] = max(0,df.loc[i,'Jet_leptonDeltaR'])
    df.loc[i,'Jet_leptonDeltaR_corr'] = max(0,df.loc[i,'Jet_leptonDeltaR_corr'])
    df.loc[i,'Jet_leptonPt'] = max(0,df.loc[i,'Jet_leptonPt'])
    df.loc[i,'Jet_leptonPt_corr'] = max(0,df.loc[i,'Jet_leptonPt_corr'])
    df.loc[i,'Jet_leptonPtRel'] = max(0,df.loc[i,'Jet_leptonPtRel'])
    df.loc[i,'Jet_leptonPtRel_corr'] = max(0,df.loc[i,'Jet_leptonPtRel_corr'])
    df.loc[i,'Jet_leptonPtRelInv'] = max(0,df.loc[i,'Jet_leptonPtRelInv'])
    df.loc[i,'Jet_leptonPtRelInv_corr'] = max(0,df.loc[i,'Jet_leptonPtRelInv_corr'])
 
    jet_vector = rt.TLorentzVector()
    jet_vector_corr = rt.TLorentzVector()
    jet_vector.SetPtEtaPhiM(df.loc[i,'Jet_pt'],df.loc[i,'Jet_eta'],df.loc[i,'Jet_phi'],df.loc[i,'Jet_mass'])
    jet_vector_corr.SetPtEtaPhiM(df.loc[i,'Jet_pt'],df.loc[i,'Jet_eta'],df.loc[i,'Jet_phi'],df.loc[i,'Jet_mass_corr'])        

    jet_rescale = df.loc[i,'Jet_rawEnergy']/jet_vector.E()
    jet_rescale_corr = df.loc[i,'Jet_rawEnergy']/jet_vector_corr.E()
        
    df.loc[i,'Jet_mass_raw'] = df.loc[i,'Jet_mass']*jet_rescale
    df.loc[i,'Jet_mass_corr_raw'] = df.loc[i,'Jet_mass_corr']*jet_rescale_corr
    df.loc[i,'Jet_leptonPtRelInv_raw'] = df.loc[i,'Jet_leptonPtRelInv']*jet_rescale
    df.loc[i,'Jet_leptonPtRelInv_corr_raw'] = df.loc[i,'Jet_leptonPtRelInv_corr']*jet_rescale_corr
    df.loc[i,'Jet_mt_raw'] = jet_vector.Mt()*jet_rescale
    df.loc[i,'Jet_mt_corr_raw'] = jet_vector_corr.Mt()*jet_rescale_corr
    df.loc[i,'JEC_JER_inv'] = jet_rescale
    df.loc[i,'JEC_JER_inv_corr'] = jet_rescale_corr   


hdf = pd.HDFStore(h5_dir+'_NN.h5')
hdf.put('hdf', df)   # Here you defined 'hdf_new' as a key to your DF. Now, while reading it you have to use the same key
hdf.close()
