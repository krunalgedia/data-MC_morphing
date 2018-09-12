from __future__ import division
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


main = pd.read_hdf('/mnt/t3nfs01/data01/shome/krgedia/Workspace/lxplus/qreg_pt_norm/t_dataset_sig_test.h5', 'hdf')

#variables = ["Jet_leptonDeltaR","Jet_mass","Jet_pt","Jet_eta","Jet_phi","Jet_rawEnergy","Jet_numDaughters_pt03","Jet_numberOfDaughters","rho","rho_copy","Jet_rawPt","Jet_chHEF","Jet_neEmEF","Jet_leadTrackPt","Jet_leptonPt","Jet_leptonPtRel","Jet_leptonPtRelInv","Jet_ptd"]

#variables = ["Jet_mass","Jet_rawEnergy","Jet_rawPt","Jet_chHEF","Jet_neEmEF","Jet_leadTrackPt","Jet_ptd","Jet_leptonPt","Jet_leptonPtRel","Jet_leptonPtRelInv","Jet_leptonDeltaR"]

variables = ["Jet_vtxMass","Jet_vtxPt","Jet_vtx3DSig","Jet_vtx3DVal","Jet_ptd","Jet_neHEF","Jet_muEF","Jet_chEmEF", "Jet_energyRing_dR0_neut", "Jet_energyRing_dR1_neut" ,"Jet_energyRing_dR2_neut", "Jet_energyRing_dR3_neut", "Jet_energyRing_dR4_neut", "Jet_energyRing_dR0_ch", "Jet_energyRing_dR1_ch", "Jet_energyRing_dR2_ch", "Jet_energyRing_dR3_ch", "Jet_energyRing_dR4_ch", "Jet_energyRing_dR0_em", "Jet_energyRing_dR1_em", "Jet_energyRing_dR2_em", "Jet_energyRing_dR3_em", "Jet_energyRing_dR4_em", "Jet_energyRing_dR0_mu", "Jet_energyRing_dR1_mu", "Jet_energyRing_dR2_mu", "Jet_energyRing_dR3_mu", "Jet_energyRing_dR4_mu","Jet_mass", "Jet_rawEnergy", "Jet_rawPt", "Jet_chHEF", "Jet_neEmEF", "Jet_leadTrackPt","Jet_leptonDeltaR", "Jet_leptonPt", "Jet_leptonPtRel", "Jet_leptonPtRelInv"]
#"Jet_energyRing_dR4_mu"
store = pd.DataFrame()

for var in variables:
    print var
    new =  pd.read_hdf('/mnt/t3nfs01/data01/shome/krgedia/Workspace/lxplus/qreg_pt_norm/h5_corr_files/t_dataset_sig'+'_'+var+'_corr'+ '.h5','hdf_corr')

    final = pd.concat([store,new],axis=1)
 
    store = final

final = pd.concat([main,store],axis=1)

print final

hdf_var_new = pd.HDFStore('/mnt/t3nfs01/data01/shome/krgedia/Workspace/lxplus/qreg_pt_norm/t_dataset_sig_test_new.h5')
hdf_var_new.put('hdf',final) #here you defined 'hdf' as a key. So when open this hdf file, please use this key
hdf_var_new.close()

mc = pd.read_hdf('/mnt/t3nfs01/data01/shome/krgedia/Workspace/lxplus/qreg_pt_norm/t_dataset_sig_test_new.h5','hdf')

ringE = 0
ringEnergy_sum = []

for i in range(0,len(mc)):
    ringE = mc.loc[i,'Jet_energyRing_dR0_ch_corr']+mc.loc[i,'Jet_energyRing_dR1_ch_corr']+mc.loc[i,'Jet_energyRing_dR2_ch_corr']+mc.loc[i,'Jet_energyRing_dR3_ch_corr']+mc.loc[i,'Jet_energyRing_dR4_ch_corr']+mc.loc[i,'Jet_energyRing_dR0_mu_corr']+mc.loc[i,'Jet_energyRing_dR1_mu_corr']+mc.loc[i,'Jet_energyRing_dR2_mu_corr']+mc.loc[i,'Jet_energyRing_dR3_mu_corr']+mc.loc[i,'Jet_energyRing_dR4_mu_corr']+mc.loc[i,'Jet_energyRing_dR0_neut_corr']+mc.loc[i,'Jet_energyRing_dR1_neut_corr']+mc.loc[i,'Jet_energyRing_dR2_neut_corr']+mc.loc[i,'Jet_energyRing_dR3_neut_corr']+mc.loc[i,'Jet_energyRing_dR4_neut_corr']+mc.loc[i,'Jet_energyRing_dR0_em_corr']+mc.loc[i,'Jet_energyRing_dR1_em_corr']+mc.loc[i,'Jet_energyRing_dR2_em_corr']+mc.loc[i,'Jet_energyRing_dR3_em_corr']+mc.loc[i,'Jet_energyRing_dR4_em_corr']    
    ringEnergy_sum.append(ringE)
    
mc.loc[:,'ringEnergy_sum'] = ringEnergy_sum

print "I computed the ring energy"

ch_0 = mc['Jet_energyRing_dR0_ch_corr']
ch_1 = mc['Jet_energyRing_dR1_ch_corr']
ch_2 = mc['Jet_energyRing_dR2_ch_corr']
ch_3 = mc['Jet_energyRing_dR3_ch_corr']
ch_4 = mc['Jet_energyRing_dR4_ch_corr']
mu_0 = mc['Jet_energyRing_dR0_mu_corr']
mu_1 = mc['Jet_energyRing_dR1_mu_corr']
mu_2 = mc['Jet_energyRing_dR2_mu_corr']
mu_3 = mc['Jet_energyRing_dR3_mu_corr']
mu_4 = mc['Jet_energyRing_dR4_mu_corr']
neut_0 = mc['Jet_energyRing_dR0_neut_corr']
neut_1 = mc['Jet_energyRing_dR1_neut_corr']
neut_2 = mc['Jet_energyRing_dR2_neut_corr']
neut_3 = mc['Jet_energyRing_dR3_neut_corr']
neut_4 = mc['Jet_energyRing_dR4_neut_corr']
em_0 = mc['Jet_energyRing_dR0_em_corr']
em_1 = mc['Jet_energyRing_dR1_em_corr']
em_2 = mc['Jet_energyRing_dR2_em_corr']
em_3 = mc['Jet_energyRing_dR3_em_corr']
em_4 = mc['Jet_energyRing_dR4_em_corr']

ring_energy  = mc['ringEnergy_sum']

#yield_b_sw = mc["yield_b_sw"]
#p_sw_mc_norm = mc['p_sw_mc_norm']

Jet_energyRing_dR0_ch_corr = []
Jet_energyRing_dR1_ch_corr = []
Jet_energyRing_dR2_ch_corr = []
Jet_energyRing_dR3_ch_corr = []
Jet_energyRing_dR4_ch_corr = []
Jet_energyRing_dR0_em_corr = []
Jet_energyRing_dR1_em_corr = []
Jet_energyRing_dR2_em_corr = []
Jet_energyRing_dR3_em_corr = []
Jet_energyRing_dR4_em_corr = []
Jet_energyRing_dR0_mu_corr = []
Jet_energyRing_dR1_mu_corr = []
Jet_energyRing_dR2_mu_corr = []
Jet_energyRing_dR3_mu_corr = []
Jet_energyRing_dR4_mu_corr = []
Jet_energyRing_dR0_neut_corr = []
Jet_energyRing_dR1_neut_corr = []
Jet_energyRing_dR2_neut_corr = []
Jet_energyRing_dR3_neut_corr = []
Jet_energyRing_dR4_neut_corr = []

print "I am now at ring energy plots"

#total_sw_mc_norm = []

for i in range(0,len(mc)):
    Ch_0 = ch_0[i]/ring_energy[i]
    Ch_1 = ch_1[i]/ring_energy[i]
    Ch_2 = ch_2[i]/ring_energy[i]
    Ch_3 = ch_3[i]/ring_energy[i]
    Ch_4 = ch_4[i]/ring_energy[i]
    Mu_0 = mu_0[i]/ring_energy[i]
    Mu_1 = mu_1[i]/ring_energy[i]
    Mu_2 = mu_2[i]/ring_energy[i]
    Mu_3 = mu_3[i]/ring_energy[i]
    Mu_4 = mu_4[i]/ring_energy[i]
    Em_0 = em_0[i]/ring_energy[i]
    Em_1 = em_1[i]/ring_energy[i]
    Em_2 = em_2[i]/ring_energy[i]
    Em_3 = em_3[i]/ring_energy[i]
    Em_4 = em_4[i]/ring_energy[i]
    Neut_0 = neut_0[i]/ring_energy[i]
    Neut_1 = neut_1[i]/ring_energy[i]
    Neut_2 = neut_2[i]/ring_energy[i]
    Neut_3 = neut_3[i]/ring_energy[i]
    Neut_4 = neut_4[i]/ring_energy[i]
    #ttl_sw_mc_norm = yield_b_sw[i]*p_sw_mc_norm[i]

    Jet_energyRing_dR0_ch_corr.append(Ch_0)
    Jet_energyRing_dR1_ch_corr.append(Ch_1)
    Jet_energyRing_dR2_ch_corr.append(Ch_2)
    Jet_energyRing_dR3_ch_corr.append(Ch_3)
    Jet_energyRing_dR4_ch_corr.append(Ch_4)
    Jet_energyRing_dR0_em_corr.append(Em_0)
    Jet_energyRing_dR1_em_corr.append(Em_1)
    Jet_energyRing_dR2_em_corr.append(Em_2)
    Jet_energyRing_dR3_em_corr.append(Em_3)
    Jet_energyRing_dR4_em_corr.append(Em_4)
    Jet_energyRing_dR0_mu_corr.append(Mu_0)
    Jet_energyRing_dR1_mu_corr.append(Mu_1)
    Jet_energyRing_dR2_mu_corr.append(Mu_2)
    Jet_energyRing_dR3_mu_corr.append(Mu_3)
    Jet_energyRing_dR4_mu_corr.append(Mu_4)
    Jet_energyRing_dR0_neut_corr.append(Neut_0)
    Jet_energyRing_dR1_neut_corr.append(Neut_1)
    Jet_energyRing_dR2_neut_corr.append(Neut_2)
    Jet_energyRing_dR3_neut_corr.append(Neut_3)
    Jet_energyRing_dR4_neut_corr.append(Neut_4)
    #total_sw_mc_norm.append(ttl_sw_mc_norm)

print "done"

mc.loc[:,'Jet_energyRing_dR0_ch_corr'] = Jet_energyRing_dR0_ch_corr
mc.loc[:,'Jet_energyRing_dR1_ch_corr'] = Jet_energyRing_dR1_ch_corr
mc.loc[:,'Jet_energyRing_dR2_ch_corr'] = Jet_energyRing_dR2_ch_corr
mc.loc[:,'Jet_energyRing_dR3_ch_corr'] = Jet_energyRing_dR3_ch_corr
mc.loc[:,'Jet_energyRing_dR4_ch_corr'] = Jet_energyRing_dR4_ch_corr

mc.loc[:,'Jet_energyRing_dR0_mu_corr'] = Jet_energyRing_dR0_mu_corr
mc.loc[:,'Jet_energyRing_dR1_mu_corr'] = Jet_energyRing_dR1_mu_corr
mc.loc[:,'Jet_energyRing_dR2_mu_corr'] = Jet_energyRing_dR2_mu_corr
mc.loc[:,'Jet_energyRing_dR3_mu_corr'] = Jet_energyRing_dR3_mu_corr
mc.loc[:,'Jet_energyRing_dR4_mu_corr'] = Jet_energyRing_dR4_mu_corr

mc.loc[:,'Jet_energyRing_dR0_em_corr'] = Jet_energyRing_dR0_em_corr
mc.loc[:,'Jet_energyRing_dR1_em_corr'] = Jet_energyRing_dR1_em_corr
mc.loc[:,'Jet_energyRing_dR2_em_corr'] = Jet_energyRing_dR2_em_corr
mc.loc[:,'Jet_energyRing_dR3_em_corr'] = Jet_energyRing_dR3_em_corr
mc.loc[:,'Jet_energyRing_dR4_em_corr'] = Jet_energyRing_dR4_em_corr

mc.loc[:,'Jet_energyRing_dR0_neut_corr'] = Jet_energyRing_dR0_neut_corr
mc.loc[:,'Jet_energyRing_dR1_neut_corr'] = Jet_energyRing_dR1_neut_corr
mc.loc[:,'Jet_energyRing_dR2_neut_corr'] = Jet_energyRing_dR2_neut_corr
mc.loc[:,'Jet_energyRing_dR3_neut_corr'] = Jet_energyRing_dR3_neut_corr
mc.loc[:,'Jet_energyRing_dR4_neut_corr'] = Jet_energyRing_dR4_neut_corr

#mc.loc[:,'total_sw_mc_norm'] = total_sw_mc_norm

hdf_plots = pd.HDFStore('/mnt/t3nfs01/data01/shome/krgedia/Workspace/lxplus/qreg_pt_norm/t_dataset_sig_test_plots.h5')
hdf_plots.put('hdf',mc) #here you defined 'hdf' as a key. So when open this hdf file, please use this key
hdf_plots.close()
