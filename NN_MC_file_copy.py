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

df = mc

df['isEle']=np.zeros(df.shape[0])
df['isMu']=np.zeros(df.shape[0])
df['isOther']=np.zeros(df.shape[0])

vtxMass = df['Jet_vtxMass']
vtxMass_corr = df['Jet_vtxMass_corr']
vtxPt = df['Jet_vtxPt']
vtxPt_corr = df['Jet_vtxPt_corr']
vtx3DSig = df['Jet_vtx3DSig']
vtx3DSig_corr = df['Jet_vtx3DSig_corr']
vtx3DVal = df['Jet_vtx3DVal'] 
vtx3DVal_corr = df['Jet_vtx3DVal_corr']
leptonDeltaR = df['Jet_leptonDeltaR']
leptonDeltaR_corr = df['Jet_leptonDeltaR_corr']
leptonPt = df['Jet_leptonPt']
leptonPt_corr = df['Jet_leptonPt_corr']
leptonPtRel = df['Jet_leptonPtRel']
leptonPtRel_corr = df['Jet_leptonPtRel_corr']
leptonPtRelInv = df['Jet_leptonPtRelInv']
leptonPtRelInv_corr = df['Jet_leptonPtRelInv_corr']

pt = df['Jet_pt']
eta = df['Jet_eta']
mass = df['Jet_mass']
phi = df['Jet_phi']
mass_corr = df['Jet_mass_corr']
rawenergy = df['Jet_rawEnergy']

pdgid = abs(df['Jet_leptonPdgId'])

isEle = []
isMu = []
isOther = []

Jet_vtxMass = []
Jet_vtxMass_corr = []
Jet_vtxPt = []
Jet_vtxPt_corr = []
Jet_vtx3DSig = []
Jet_vtx3DSig_corr = []
Jet_vtx3DVal = []
Jet_vtx3DVal_corr = []
Jet_leptonDeltaR = []
Jet_leptonDeltaR_corr = []
Jet_leptonPt = []
Jet_leptonPt_corr = []
Jet_leptonPtRel = []
Jet_leptonPtRel_corr = []
Jet_leptonPtRelInv = []
Jet_leptonPtRelInv_corr = []






for i in range(0, len(df)):

    jl_pdgid = pdgid[i]

    if (jl_pdgid==11): 
        isele = 1
              
    elif (jl_pdgid == 13.0):
        ismu = 1
            
    else: 
        isother = 1

    jet_vtxMass = max(0,vtxMass[i])
    jet_vtxMass_corr = max(0,vtxMass_corr[i])
    jet_vtxPt = max(0,vtxPt[i])
    jet_vtxPt_corr = max(0,vtxPt_corr[i])
    jet_vtx3DSig = max(0,vtx3DSig[i])
    jet_vtx3DSig_corr = max(0,vtx3DSig_corr[i])
    jet_vtx3DVal = max(0,vtx3DVal[i])
    jet_vtx3DVal_corr = max(0,vtx3DVal_corr[i])
    jet_leptonDeltaR = max(0,leptonDeltaR[i])
    jet_leptonDeltaR_corr = max(0,leptonDeltaR_corr[i])
    jet_leptonPt = max(0,leptonPt[i])
    jet_leptonPt_corr = max(0,leptonPt_corr[i])
    jet_leptonPtRel = max(0,leptonPtRel[i])
    jet_leptonPtRel_corr = max(0,leptonPtRel_corr[i])
    jet_leptonPtRelInv = max(0,leptonPtRelInv[i])
    jet_leptonPtRelInv_corr = max(0,leptonPtRelInv_corr[i])
 
    jet_vector = rt.TLorentzVector()
    jet_vector_corr = rt.TLorentzVector()
    jet_vector.SetPtEtaPhiM(pt[i],eta[i],phi[i],mass[i])
    jet_vector_corr.SetPtEtaPhiM(pt[i],eta[i],phi[i],mass_corr[i])        

    jet_rescale = rawenergy[i]/jet_vector.E()
    jet_rescale_corr = rawenergy[i]/jet_vector_corr.E()
        
    mass_raw = mass[i]*jet_rescale
    mass_corr_raw = mass_corr[i]*jet_rescale_corr
    leptonPtRelInv_raw = leptonPtRelInv[i]*jet_rescale
    leptonPtRelInv_corr_raw = leptonPtRelInv_corr[i]*jet_rescale_corr
    mt_raw = jet_vector.Mt()*jet_rescale
    mt_corr_raw = jet_vector_corr.Mt()*jet_rescale_corr
    jec_jer_inv = jet_rescale
    jec_jer_inv_corr = jet_rescale_corr   

        
    isEle.append(isele)     
    isMu.append(ismu)
    isOther.append(isother)

    Jet_vtxMass.append(vtxMass)
    Jet_vtxMass_corr.append(vtxMass_corr)
    Jet_vtxPt.append(vtxPt)
    Jet_vtxPt_corr.append(vtxPt_corr)
    Jet_vtx3DSig.append(vtx3DSig)
    Jet_vtx3DSig_corr.append(vtx3DSig_corr)
    Jet_vtx3DVal.append(vtx3DVal)
    Jet_vtx3DVal_corr.append(vtx3DVal_corr)
    Jet_leptonDeltaR.append(leptonDeltaR)
    Jet_leptonDeltaR_corr.append(leptonDeltaR_corr)
    Jet_leptonPt.append(leptonPt)
    Jet_leptonPt_corr.append(leptonPt_corr)
    Jet_leptonPtRel.append(leptonPtRel)
    Jet_leptonPtRel_corr.append(leptonPtRel_corr)
    Jet_leptonPtRelInv.append(leptonPtRelInv)
    Jet_leptonPtRelInv_corr.append(leptonPtRelInv_corr)
    
    Jet_mass_raw.append(mass_raw) 
    Jet_mass_corr_raw.append(mass_corr_raw)
    Jet_leptonPtRelInv_raw.append(leptonPtRelInv_raw)
    Jet_leptonPtRelInv_corr_raw.append(leptonPtRelInv_corr_raw)
    Jet_mt_raw.append(mt_raw)
    Jet_mt_corr_raw.append(mt_corr_raw)
    JEC_JER_inv.append(jec_jer_inv)
    JEC_JER_inv_corr.append(jec_jer_inv_corr)


df.loc['Jet_vtxMass'].append(vtxMass)
df['Jet_vtxMass_corr'].append(vtxMass_corr)
df['Jet_vtxPt'].append(vtxPt)
df['Jet_vtxPt_corr'].append(vtxPt_corr)
df['Jet_vtx3DSig'].append(vtx3DSig)
df['Jet_vtx3DSig_corr'].append(vtx3DSig_corr)
df['Jet_vtx3DVal'].append(vtx3DVal)
df['Jet_vtx3DVal_corr'].append(vtx3DVal_corr)
df['Jet_leptonDeltaR'].append(leptonDeltaR)
df['Jet_leptonDeltaR_corr'].append(leptonDeltaR_corr)
df['Jet_leptonPt'].append(leptonPt)
df['Jet_leptonPt_corr'].append(leptonPt_corr)
df['Jet_leptonPtRel'].append(leptonPtRel)
df['Jet_leptonPtRel_corr'].append(leptonPtRel_corr)
df['Jet_leptonPtRelInv'].append(leptonPtRelInv)
df['Jet_leptonPtRelInv_corr'].append(leptonPtRelInv_corr)

df['Jet_mass_raw'].append(mass_raw)
df['Jet_mass_corr_raw'].append(mass_corr_raw)
df['Jet_leptonPtRelInv_raw'].append(leptonPtRelInv_raw)
df['Jet_leptonPtRelInv_corr_raw'].append(leptonPtRelInv_corr_raw)
df['Jet_mt_raw'].append(mt_raw)
df['Jet_mt_corr_raw'].append(mt_corr_raw)
df['JEC_JER_inv'].append(jec_jer_inv)
df['JEC_JER_inv_corr']append(jec_jer_inv_corr)












hdf = pd.HDFStore(h5_dir+'_NN.h5')
hdf.put('hdf', df)   # Here you defined 'hdf_new' as a key to your DF. Now, while reading it you have to use the same key
hdf.close()
