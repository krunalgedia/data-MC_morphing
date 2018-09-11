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
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier

class mycolors:
    red = '\033[91m'
    green = '\033[92m'
    blue = '\033[94m'
    yan = '\033[96m'
    cWhite = '\033[97m'
    yellow = '\033[93m'
    magenta = '\033[95m'
    grey = '\033[90m'
    black = '\033[90m'
    default = '\033[0m'

def setupJoblib(ipp_profile='default'):
    
    import ipyparallel as ipp
    from ipyparallel.joblib import IPythonParallelBackend
    global joblib_rc,joblib_view,joblib_be
    joblib_rc = ipp.Client(profile=ipp_profile)
    joblib_view = joblib_rc.load_balanced_view()
    joblib_be = IPythonParallelBackend(view=joblib_view)
    
    register_parallel_backend('ipyparallel',lambda : joblib_be,make_default=True)


class quantileRegression:

    def __init__(self, label, file_name, tree_name):

        self.label   = label
        self.dataMC  = label.split("_",1)[0]

        self.file_name = file_name
        self.tree_name  = tree_name

        self.df_train = 0
        self.df_test  = 0

        self.hdf_train= 0
        self.hdf_test = 0 
              
        self.smcclf = []
        self.sdataclf = []
        self.mcclf = []
        self.snmcclf = []
        
        self.smcclass = 0
        self.sdataclass = 0
        self.mcclass = 0

        self.y_corr = 0
        self.hdf_corr = 0
        self.hdf_corr_add = pd.DataFrame()


    def loadDF(self, rndm = 12345):
      
        df = rpd.read_root(self.file_name,self.tree_name)          
        print "#events in tree "+ str(self.tree_name)+ " :", len(df.index)

        rndseed = rndm
        np.random.seed(rndseed)
        index = list(df.index)
        np.random.shuffle(index)
	df["index"] = np.array(index)
        df.set_index("index",inplace=True)
        df.sort_index(inplace=True)

        print "After reshuffling = ", len(df.index)

        if (self.tree_name == "t_dataset_sig"):      
            df_train = df[0:600000]
            df_test = df[600000:] 

            #self.df = self.df.head(1000)
            #print "Cut DataFrame size = ", len(self.df.index)

            self.df_train = df_train
            self.df_test = df_test
           
            self.df_test.reset_index(drop=True, inplace=True)

            hdf_train = pd.HDFStore('/mnt/t3nfs01/data01/shome/krgedia/Workspace/lxplus/qreg_pt_norm/'+self.tree_name + '_train.h5')
            hdf_train.put('hdf', self.df_train)   # Here you defined 'hdf' as a key to your DF. Now, while reading it you have to use the same key
            self.hdf_train = hdf_train
            hdf_train.close()

            hdf_test = pd.HDFStore('/mnt/t3nfs01/data01/shome/krgedia/Workspace/lxplus/qreg_pt_norm/'+self.tree_name + '_test.h5')
            hdf_test.put('hdf', self.df_test)   # Here you defined 'hdf' as a key to your DF. Now, while reading it you have to use the same key
            self.hdf_test = hdf_test
            hdf_test.close()

            print "#events in train tree during saving"+ str(self.tree_name)+ " :", len(self.df_train.index)
            print "#events in test tree during saving"+ str(self.tree_name)+ " :", len(self.df_test.index)
  
        else:
            self.df_train = df

            hdf_train = pd.HDFStore('/mnt/t3nfs01/data01/shome/krgedia/Workspace/lxplus/qreg_pt_norm/'+self.tree_name + '_train.h5')
            hdf_train.put('hdf', self.df_train)   # Here you defined 'hdf' as a key to your DF. Now, while reading it you have to use the same key
            self.hdf_train = hdf_train
            hdf_train.close()
            
    
    def loadDFh5_train(self):

        hdf_train =  pd.read_hdf('/mnt/t3nfs01/data01/shome/krgedia/Workspace/lxplus/qreg_pt_norm/'+self.tree_name + '_train.h5', 'hdf')
        print "number of events in h5 train file of :"+ str(self.tree_name), len(hdf_train.index)
        self.hdf_train = hdf_train
        #print self.hdf_train
    
    def loadDFh5_test(self):

        hdf_test =  pd.read_hdf('/mnt/t3nfs01/data01/shome/krgedia/Workspace/lxplus/qreg_pt_norm/'+self.tree_name + '_test.h5', 'hdf')
        print "number of events in h5 test file of :"+ str(self.tree_name), len(hdf_test.index)
        hdf_ran = pd.read_hdf('/mnt/t3nfs01/data01/shome/krgedia/Workspace/lxplus/qreg_pt_norm/random_numbers.h5','hdf')
        hdf_test = pd.concat([hdf_test,hdf_ran], axis=1)
        self.hdf_test = hdf_test
    
    def trainDeviance(self,var):
        #if (var[:14] == "Jet_energyRing" or var == "Jet_chEmEF" or var == "Jet_muEF" or var == "Jet_neHEF" or var[:7] == "Jet_vtx"):
        #print self.hdf_train.loc[0:5,var]
        X = self.hdf_train.loc[:,['Jet_pt', 'Jet_eta', 'Jet_phi', 'rho_copy']]
        w = []

        if self.tree_name == "t_dataset_sig":
            w = self.hdf_train['yield_b_sw']*self.hdf_train['p_sw_mc_norm']

        if self.tree_name == "t_r_dataset_sig":
            w = self.hdf_train['r_yield_b_sw']*self.hdf_train['p_sw_data_norm']

        if self.tree_name == "mc_sig":
            w = self.hdf_train['rho_weight']

        cls = []
        cl = 0

        if (var[:14] == "Jet_energyRing" or var == "Jet_chEmEF" or var == "Jet_muEF" or var == "Jet_neHEF" or var[:7] == "Jet_vtx"):     
            for i in range(0, len(self.hdf_train)):
                if (var[:14] == "Jet_energyRing"):
                    if ((self.hdf_train.at[i,var]*self.hdf_train.at[i,'Jet_rawEnergy']) == 0):cl = 0
                    else:cl = 1
                else:
                    if (self.hdf_train.at[i,var] == 0):cl = 0
                    else:cl = 1
                    cls.append(cl)
            
        else:
            for i in range(0, len(self.hdf_train)):
                if (self.hdf_train.at[i,var] == -99):cl = 0
                else:cl = 1
                cls.append(cl)

        print len(self.hdf_train)
        print len(cls)
        self.hdf_train['class'] = cls
        Y_class = self.hdf_train['class']

        clf_deviance = GradientBoostingClassifier(loss='deviance',n_estimators=250, max_depth=3,learning_rate=.1, min_samples_leaf=9,min_samples_split=9)
        clf_deviance.fit(X,Y_class,w)

        if (self.label == "sw_mc"):
            pickle.dump(clf_deviance, gzip.open("/mnt/t3nfs01/data01/shome/krgedia/Workspace/lxplus/qreg_pt_norm/class/sw_mc"+"_"+var+".pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

        if (self.label == "sw_data"):
            pickle.dump(clf_deviance, gzip.open("/mnt/t3nfs01/data01/shome/krgedia/Workspace/lxplus/qreg_pt_norm/class/sw_data"+"_"+var+".pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

        if (self.label == "mc"):
            pickle.dump(clf_deviance, gzip.open("/mnt/t3nfs01/data01/shome/krgedia/Workspace/lxplus/qreg_pt_norm/class/mc"+"_"+var+".pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

        var_list = []

        if (var == "Jet_leptonDeltaR" or var == "Jet_vtxMass"):
            if  (var == "Jet_leptonDeltaR"): 
                var_list = ["Jet_leptonPt","Jet_leptonPtRel","Jet_leptonPtRelInv"]
            if (var == "Jet_vtxMass"): 
                var_list = ["Jet_vtxPt","Jet_vtx3DSig","Jet_vtx3DVal"]
            if (self.label == "sw_mc"):
                for v in var_list:
                    pickle.dump(clf_deviance, gzip.open("/mnt/t3nfs01/data01/shome/krgedia/Workspace/lxplus/qreg_pt_norm/class/sw_mc"+"_"+v+".pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

            if (self.label == "sw_data"):
                for v in var_list:
                    pickle.dump(clf_deviance, gzip.open("/mnt/t3nfs01/data01/shome/krgedia/Workspace/lxplus/qreg_pt_norm/class/sw_data"+"_"+v+".pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

            if (self.label == "mc"):
                for v in var_list:
                    pickle.dump(clf_deviance, gzip.open("/mnt/t3nfs01/data01/shome/krgedia/Workspace/lxplus/qreg_pt_norm/class/mc"+"_"+v+".pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
         
    def trainQuantile(self, var, alpha, maxDepth = 3, minLeaf = 9):

        print self.hdf_train.loc[0:5,var]

        Y = []
        if (var[:14] == "Jet_energyRing" or var == "Jet_chEmEF" or var == "Jet_muEF" or var == "Jet_neHEF" or var[:7] == "Jet_vtx" or var[:10] == "Jet_lepton"):
            if (var[:14] == "Jet_energyRing" or var == "Jet_chEmEF" or var == "Jet_muEF" or var == "Jet_neHEF" or var == "Jet_vtxMass" or var == "Jet_leptonDeltaR"):
                if (alpha == 0.05): self.trainDeviance(var) 
            #else:              
            #    if (alpha == 0.05 and var == "Jet_leptonDeltaR"):
            #        self.trainDeviance(var)
            #        print "I am here"
            hdf_train_cut = self.hdf_train.query(var+">0")
            hdf_train_cut.reset_index(inplace = True)
            self.hdf_train = hdf_train_cut
            #print "#events in train tree during class clf"+ str(self.tree_name)+ " :", len(self.hdf_train.index)

        #Y = self.hdf_train[var]
        #Y_raw = self.hdf_train[var] * self.hdf_train["Jet_rawEnergy"]
        
        if (var[:14] == "Jet_energyRing"): Y = self.hdf_train[var]*self.hdf_train['Jet_rawEnergy']
        else: Y = self.hdf_train[var]

        #elif (var == "Jet_leptonPt" or var == "Jet_leptonPtRel" or var == "Jet_leptonDeltaR" or var == "Jet_leptonPtRelInv"):
        #    hdf_train_cut = self.hdf_train.query(var+">0")
        #    hdf_train_cut.reset_index(inplace = True)
        #    self.hdf_train = hdf_train_cut
            #X = self.hdf_train.loc[:,['Jet_pt', 'Jet_eta', 'Jet_phi', 'rho_copy']]
        #    Y = self.hdf_train[var]

#        else:
            #X = self.hdf_train.loc[:,['Jet_pt', 'Jet_eta', 'Jet_phi', 'rho_copy']] 
         #   Y = self.hdf_train[var]

        #print self.hdf_train.head()
        print "#events in train tree during quantile clf "+ str(self.tree_name)+ " :", len(self.hdf_train.index)

        var_list = []
        X = []
        X1 = []

        if (var[:7] == "Jet_vtx" or var[:10] == "Jet_lepton"):
            if (var[:7] == "Jet_vtx"):
                var_list = ["Jet_vtxMass","Jet_vtxPt","Jet_vtx3DSig","Jet_vtx3DVal"]
            if (var[:10] == "Jet_lepton"):
                var_list = ["Jet_leptonDeltaR","Jet_leptonPt","Jet_leptonPtRel","Jet_leptonPtRelInv"]
            var_list.remove(var)
            X1 =  self.hdf_train.loc[:,['Jet_pt', 'Jet_eta', 'Jet_phi', 'rho_copy',var_list[0],var_list[1],var_list[2]]]

        X = self.hdf_train.loc[:,['Jet_pt', 'Jet_eta', 'Jet_phi', 'rho_copy']]    

        #print X[:2]
        #X = self.hdf_train.loc[:,['Jet_pt', 'Jet_eta', 'Jet_phi', 'rho_copy']]
        # target
        #Y = self.hdf_train[var]
        #event weight

        w = []

        #X = self.hdf_train.loc[:,['Jet_pt', 'Jet_eta', 'Jet_phi', 'rho_copy']]
        # target
        #Y = self.hdf_train[var]
        #event weight
     
        if self.tree_name == "t_dataset_sig":
            w = self.hdf_train['yield_b_sw']*self.hdf_train['p_sw_mc_norm']
           
        if self.tree_name == "t_r_dataset_sig":
            w = self.hdf_train['r_yield_b_sw']*self.hdf_train['p_sw_data_norm']

        if self.tree_name == "mc_sig":
            w = self.hdf_train['rho_weight']

        clf = GradientBoostingRegressor(loss='quantile', alpha=alpha,n_estimators=250, max_depth=maxDepth,learning_rate=.1, min_samples_leaf=minLeaf,min_samples_split=minLeaf)         
        clf.fit(X, Y, w)
  
        y_upper = clf.predict(X)
        self.hdf_train = self.hdf_train.assign(y_upper=y_upper)    

        print var + ' ' + str(alpha)
        print Y[0:5]
        print self.hdf_train.loc[0:5,'y_upper']

        if (self.label == "sw_mc"):
            pickle.dump(clf, gzip.open("/mnt/t3nfs01/data01/shome/krgedia/Workspace/lxplus/qreg_pt_norm/weights/sw_mc"+"_"+var+"_"+str(alpha)+".pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

        if (self.label == "sw_data"):
            pickle.dump(clf, gzip.open("/mnt/t3nfs01/data01/shome/krgedia/Workspace/lxplus/qreg_pt_norm/weights/sw_data"+"_"+var+"_"+str(alpha)+".pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
         
        if (self.label == "mc"):
            pickle.dump(clf, gzip.open("/mnt/t3nfs01/data01/shome/krgedia/Workspace/lxplus/qreg_pt_norm/weights/mc"+"_"+var+"_"+str(alpha)+".pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

        if (self.label == "sw_mc"):
            if (var[:7] == "Jet_vtx" or var[:10] == "Jet_lepton"):
                nclf = GradientBoostingRegressor(loss='quantile', alpha=alpha,n_estimators=250, max_depth=maxDepth,learning_rate=.1, min_samples_leaf=minLeaf,min_samples_split=minLeaf)
                nclf.fit(X1, Y, w)
                pickle.dump(nclf, gzip.open("/mnt/t3nfs01/data01/shome/krgedia/Workspace/lxplus/qreg_pt_norm/weights/sw_mc_shift"+"_"+var+"_"+str(alpha)+".pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


    def loadWeights(self, var, quantiles = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99]):

        for q in quantiles:
            smcWeights = "/mnt/t3nfs01/data01/shome/krgedia/Workspace/lxplus/qreg_pt_norm/weights/sw_mc" + "_" + var + "_" + str(q) + ".pkl"
            self.smcclf.append(pickle.load(gzip.open(smcWeights)))
             
            sdataWeights = "/mnt/t3nfs01/data01/shome/krgedia/Workspace/lxplus/qreg_pt_norm/weights/sw_data" + "_" + var + "_" + str(q) + ".pkl"
            self.sdataclf.append(pickle.load(gzip.open(sdataWeights)))
               
            mcWeights = "/mnt/t3nfs01/data01/shome/krgedia/Workspace/lxplus/qreg_pt_norm/weights/mc" + "_" + var + "_" + str(q) + ".pkl"
            self.mcclf.append(pickle.load(gzip.open(mcWeights)))

            if (var[:7] == "Jet_vtx" or var[:10] == "Jet_lepton"):
                snmcWeights = "/mnt/t3nfs01/data01/shome/krgedia/Workspace/lxplus/qreg_pt_norm/weights/sw_mc_shift" + "_" + var + "_" + str(q) + ".pkl"
                self.snmcclf.append(pickle.load(gzip.open(snmcWeights)))

        print len(self.smcclf)
        print len(self.sdataclf)
        print len(self.mcclf)

    def loadClass(self, var):

        if (var[:14] == "Jet_energyRing" or var == "Jet_chEmEF" or var == "Jet_muEF" or var == "Jet_neHEF" or var[:7] == "Jet_vtx" or var[:10] == "Jet_lepton"):
            self.smcClass = pickle.load(gzip.open("/mnt/t3nfs01/data01/shome/krgedia/Workspace/lxplus/qreg_pt_norm/class/sw_mc" + "_" + var+ ".pkl"))
            self.sdataClass = pickle.load(gzip.open("/mnt/t3nfs01/data01/shome/krgedia/Workspace/lxplus/qreg_pt_norm/class/sw_data" + "_" + var + ".pkl"))
            self.mcClass = pickle.load(gzip.open("/mnt/t3nfs01/data01/shome/krgedia/Workspace/lxplus/qreg_pt_norm/class/mc" + "_" + var + ".pkl"))
             
 
    def correctY(self, y, quantiles = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99]):
        var = y
        Y = []
        if (var[:14] == "Jet_energyRing" or var == "Jet_chEmEF" or var == "Jet_muEF" or var == "Jet_neHEF" or var[:7] == "Jet_vtx" or var[:10] == "Jet_lepton"):
            print self.hdf_test.loc[0:5,var]
            self.shiftY(y)
            Y = self.hdf_test[y+'_co'] 
            if (var[:14] == "Jet_energyRing"): nY = self.hdf_test[y]*self.hdf_test['Jet_rawEnergy']
            else: nY = self.hdf_test[y]
            print self.hdf_test.loc[0:5,var+'_co']
            Y_raw = self.hdf_test['Jet_rawEnergy']
        else:
            Y = self.hdf_test[y]


        print "Get corrections for ", y, " with quantiles ", quantiles

        y_tmp = []
        y_add = []
        y_check = []

        # quantile regressions features
        #X    = self.hdf_test.loc[:,['Jet_pt', 'Jet_eta', 'Jet_phi', 'rho_copy']]
        #var_list = []
        #X = []

        #if (var[:7] == "Jet_vtx" or var[:10] == "Jet_lepton"):
        #    if (var[:7] == "Jet_vtx"):
        #        var_list = ["Jet_vtxMass","Jet_vtxPt","Jet_vtx3DSig","Jet_vtx3DVal"]
        #    if (var[:10] == "Jet_lepton"):
        #        var_list = ["Jet_leptonDeltaR","Jet_leptonPt","Jet_leptonPtRel","Jet_leptonPtRelInv"]
        #    var_list.remove(var)
        #    X =  self.hdf_test.loc[:,['Jet_pt', 'Jet_eta', 'Jet_phi', 'rho_copy',var_list[0],var_list[1],var_list[2]]]
        #else: 
        X = self.hdf_test.loc[:,['Jet_pt', 'Jet_eta', 'Jet_phi', 'rho_copy']]


        # target e.g. y = "R9"
        #Y    = self.hdf_test[y]

        y_mc   = [] # list storing the n=q predictions on mc   for each event
        y_data = [] # list storing the n=q predictions on data for each event
        y_mc_truth = []
   
        for q in range(0, len(quantiles)):
            y_mc  .append(self.smcclf[q]  .predict(X))
            y_data.append(self.sdataclf[q].predict(X))
            y_mc_truth.append(self.mcclf[q].predict(X))

        for ievt in range(0, len(Y)):

            qmc_low  = 0
            qmc_high = 0
            q = 0
            while q < len(quantiles): 
                if y_mc[q][ievt] < Y[ievt]:
                    q+=1
                else:
                    break
            if q == 0:
                qmc_low  = 0                               # all shower shapes have a lower bound at 0
                qmc_high = y_mc[0][ievt]
            elif q < len(quantiles):
                qmc_low  = y_mc[q-1][ievt]
                qmc_high = y_mc[q ][ievt]
            else:
                qmc_low  = y_mc[q-1][ievt]
                qmc_high = quantiles[len(quantiles)-1]#*1.2 # some variables (e.g. sigmaRR) have values above 1
                                                       # to set the value for the highest quantile 20% higher
            qtmc_low  = 0
            qtmc_high = 0
            if q == 0:
                qtmc_low  = 0                              # all shower shapes have a lower bound at 0
                qtmc_high = y_mc_truth[0][ievt]
            elif q < len(quantiles):
                qtmc_low  = y_mc_truth[q-1][ievt]
                qtmc_high = y_mc_truth[q][ievt]
            else:
                qtmc_low  = y_mc_truth[q-1][ievt]
                qtmc_high = quantiles[len(quantiles)-1]


            qdata_low  = 0
            qdata_high = 0
            if q == 0:
                qdata_low  = 0                              # all shower shapes have a lower bound at 0
                qdata_high = y_data[0][ievt]
            elif q < len(quantiles):
                qdata_low  = y_data[q-1][ievt]
                qdata_high = y_data[q ][ievt]
            else:
                qdata_low  = y_data[q-1][ievt]
                qdata_high = quantiles[len(quantiles)-1]#*1.2 # see comment above for mc            


            if (var[:14] == "Jet_energyRing" or var == "Jet_chEmEF" or var == "Jet_muEF" or var == "Jet_neHEF" or var[:7] == "Jet_vtx" or var[:10] == "Jet_lepton"):

                nqmc_low  = 0
                nqmc_high = 0
                nq = 0
                while nq < len(quantiles):
                    if y_mc[nq][ievt] < nY[ievt]:
                        nq+=1
                    else:
                        break
                if nq == 0:
                    nqmc_low  = 0                               # all shower shapes have a lower bound at 0
                    nqmc_high = y_mc[0][ievt]
                elif nq < len(quantiles):
                    nqmc_low  = y_mc[nq-1][ievt]
                    nqmc_high = y_mc[nq ][ievt]
                else:
                    nqmc_low  = y_mc[nq-1][ievt]
                    nqmc_high = quantiles[len(quantiles)-1]#*1.2 # some variables (e.g. sigmaRR) have values above 1
                                                       # to set the value for the highest quantile 20% higher
                nqtmc_low  = 0
                nqtmc_high = 0
                if nq == 0:
                    nqtmc_low  = 0                              # all shower shapes have a lower bound at 0
                    nqtmc_high = y_mc_truth[0][ievt]
                elif nq < len(quantiles):
                    nqtmc_low  = y_mc_truth[nq-1][ievt]
                    nqtmc_high = y_mc_truth[nq][ievt]
                else:
                    nqtmc_low  = y_mc_truth[nq-1][ievt]
                    nqtmc_high = quantiles[len(quantiles)-1]

         # interplopate the correction
            y_corr = (qdata_high-qdata_low)/(qmc_high-qmc_low) * (Y[ievt] - qmc_low) + qdata_low

            if (var[:14] == "Jet_energyRing" or var == "Jet_chEmEF" or var == "Jet_muEF" or var == "Jet_neHEF" or var[:7] == "Jet_vtx" or var[:10] == "Jet_lepton"):
                y_tcorr = (nqtmc_high-nqtmc_low)/(nqmc_high-nqmc_low) * (nY[ievt] - nqmc_low) + nqtmc_low
                y_error = y_tcorr - nY[ievt]
            else:
                y_tcorr = (qtmc_high-qtmc_low)/(qmc_high-qmc_low) * (Y[ievt] - qmc_low) + qtmc_low
                y_error = y_tcorr - Y[ievt]

            #if (var[:14] == "Jet_energyRing"): y_corr = y_corr/Y_raw[ievt]
            y_add = [y_corr,y_tcorr,y_error]

            if (var[:10] == "Jet_lepton"):
                if (Y[ievt] == -99):
                    y_add = [-99,-99,0]

            if (var[:14] == "Jet_energyRing" or var == "Jet_chEmEF" or var == "Jet_muEF" or var == "Jet_neHEF" or var[:7] == "Jet_vtx"):
                if (Y[ievt] == 0):
                    y_add = [0,y_tcorr,y_error]
                    if (nY[ievt] == 0):
                        y_add = [0,0,0]
                
            y_tmp.append(y_add)
   
            y_check.append(y_corr)

        ycorr = y+"_corr"
        ytcorr = y+"_tcorr"
        yerror = y+"_error"
        self.hdf_test[ycorr] = y_check
        
        hdf_corr = pd.DataFrame(y_tmp,columns=[ycorr,ytcorr,yerror])  

        if (var[:14] == "Jet_energyRing" or var == "Jet_chEmEF" or var == "Jet_muEF" or var == "Jet_neHEF" or var[:7] == "Jet_vtx" or var == "Jet_leptonPt" or var == "Jet_leptonPtRel" or var == "Jet_leptonDeltaR" or var == "Jet_leptonPtRelInv"):
            hdf_corr = pd.concat([hdf_corr,self.hdf_corr_add], axis=1)

        pd.set_option('display.max_columns', None)
        print hdf_corr.head(5)

        #return hdf_corr
        hdf_var_corr = pd.HDFStore('/mnt/t3nfs01/data01/shome/krgedia/Workspace/lxplus/qreg_pt_norm/h5_corr_files/'+self.tree_name +'_'+ycorr+ '.h5')
        hdf_var_corr.put('hdf_corr',hdf_corr)
        hdf_var_corr.close()


    def shiftY(self,var):

        cls = []
        cl = 0

        #print len(self.hdf_test)

        #for i in range(0, len(self.hdf_test)):
        #    if (self.hdf_test.at[i,var] == 0):cl = 0
        #    else:cl = 1
        #    cls.append(cl)

        for i in range(0, len(self.hdf_test)):
            if (var[:10] == "Jet_lepton"):
                if (self.hdf_test.at[i,var] == -99):cl = 0
                else:cl = 1
            else:
                if (self.hdf_test.at[i,var] == 0):cl = 0
                else:cl = 1
            cls.append(cl)

        #print len(self.hdf_test)
        #print len(cls)
        #self.hdf_corr = pd.DataFrame()
        self.hdf_corr_add[var+'_class'] = cls
        self.hdf_test[var+'_class'] = cls
   
        X    = self.hdf_test.loc[:,['Jet_pt', 'Jet_eta', 'Jet_eta', 'rho_copy']]
        
        #print self.smcClass
        y_mc = self.smcClass.predict_proba(X)
        y_data = self.sdataClass.predict_proba(X)

        self.hdf_corr_add.loc[:,var+'_mc_proba_0'] = y_mc[:,0]
        self.hdf_corr_add.loc[:,var+'_mc_proba_1'] = y_mc[:,1]
        self.hdf_corr_add.loc[:,var+'_data_proba_0'] = y_data[:,0]
        self.hdf_corr_add.loc[:,var+'_data_proba_1'] = y_data[:,1]

        df_cut = self.hdf_test.query(var+'>0')
        df_cut.reset_index(inplace = True)
        Y_cut = df_cut[var]
        y_max = np.percentile(Y_cut,100.0)

        vr_co = 0
        var_co = []

        mc_algo_cls = 0
        mc_algo_class = []

        weight = []
        wgt = 0

        #print len(self.hdf_corr_add)

        var_list = []

        if (var[:7] == "Jet_vtx" or var[:10] == "Jet_lepton"):
            if (var[:7] == "Jet_vtx"):
                var_list = ["Jet_vtxMass","Jet_vtxPt","Jet_vtx3DSig","Jet_vtx3DVal"]
            if (var[:10] == "Jet_lepton"):
                var_list = ["Jet_leptonDeltaR","Jet_leptonPt","Jet_leptonPtRel","Jet_leptonPtRelInv"]
            var_list.remove(var)    

        print var_list 
        
        for i in range(0,len(self.hdf_test)):
    #print mc.loc[i]['Jet_energyRing_dR4_em'] 
            if (var[:10] == "Jet_lepton"): r = self.hdf_test.loc[i]['random_1']
            elif (var[:7] == "Jet_vtx"): r = self.hdf_test.loc[i]['random_2']  
            else: r = np.random.uniform(0,1)
            if (self.hdf_test.loc[i][var+'_class'] == 0):
                wgt = y_data[i][0]/ y_mc[i][0]
                if ((y_data[i][1]>y_mc[i][1]) and (r < ((y_data[i][1]-y_mc[i][1])/y_mc[i][0]))): 
                    mc_algo_cls = 1
            #rp = r*100        
            #p = np.percentile(Y_cut,rp)
            #print "yes" 
            #r1 = np.random.uniform(0,1)
            #print r1
            #print "yes"

                    list_func = []
                    if (var[:7] == "Jet_vtx" or var[:10] == "Jet_lepton"):
                        list_func = self.p2t(self.hdf_test.loc[i]['Jet_pt'],self.hdf_test.loc[i]['Jet_eta'],self.hdf_test.loc[i]['Jet_phi'],self.hdf_test.loc[i]['rho_copy'],y_max,self.hdf_test.loc[i][var_list[0]],self.hdf_test.loc[i][var_list[1]],self.hdf_test.loc[i][var_list[2]])
                    else:
                        list_func = self.p2t(self.hdf_test.loc[i]['Jet_pt'],self.hdf_test.loc[i]['Jet_eta'],self.hdf_test.loc[i]['Jet_phi'],self.hdf_test.loc[i]['rho_copy'],y_max)
                    vr_co = list_func[0]
            #print "yes"
            #print jet_energyRing_dR4_em_co
                else:
                    mc_algo_cls = 0
                    if (var[:14] == "Jet_energyRing"): vr_co = self.hdf_test.loc[i][var]*self.hdf_test.loc[i]['Jet_rawEnergy']
                    else: vr_co = self.hdf_test.loc[i][var]  
            if (self.hdf_test.loc[i][var+'_class'] == 1):
                wgt = y_data[i][1]/ y_mc[i][1]
                if ((y_data[i][0]>y_mc[i][0]) and  (r < ((y_data[i][0]-y_mc[i][0])/y_mc[i][1]))):
                    mc_algo_cls = 0
                    if (var[:10] == "Jet_lepton"): vr_co = -99
                    else: vr_co = 0
                    
                else:
                    mc_algo_cls = 1
                    if (var[:14] == "Jet_energyRing"): vr_co = self.hdf_test.loc[i][var]*self.hdf_test.loc[i]['Jet_rawEnergy']
                    else: vr_co = self.hdf_test.loc[i][var]  
   
 
            weight.append(wgt)
            mc_algo_class.append(mc_algo_cls)
            var_co.append(vr_co)
    
        self.hdf_corr_add.loc[:,var+'_co'] = var_co
        self.hdf_test.loc[:,var+'_co'] = var_co
        self.hdf_corr_add.loc[:,var+'_mc_algo_class'] = mc_algo_class
        self.hdf_corr_add.loc[:,var+'_class_weight'] = weight

       

    def p2t(self,pt,eta,phi,rho,y_max,var0 = 'zero',var1 = 'one',var2 = 'two',quantiles = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99]):
        
        X = []

        if (var0 == 'zero'):
            X = [pt,eta,phi,rho]
        else:
            X = [pt,eta,phi,rho,var0,var1,var2]
        X = np.array(X).reshape((1, -1))
        X.reshape(1, -1)
        y_mc = []
        #if (var[:10] == "Jet_lepton"):r1 = ran_2
        r1 = np.random.uniform(0,1)
                
        if (var0 == 'zero'):
            for q in range(0, len(quantiles)):
                y_mc.append(self.smcclf[q].predict(X))
        else:
            for q in range(0, len(quantiles)):
                y_mc.append(self.snmcclf[q].predict(X))
    
        q=0
        while q < len(quantiles):
            
            if (r1>quantiles[q]):
                q = q + 1
            else:
                break
   
        if (q<len(quantiles)):
            y_tail = (y_mc[q]-y_mc[q-1])*(r1-quantiles[q-1])/(quantiles[q]-quantiles[q-1]) + y_mc[q-1]    
        else:
            y_tail = (y_max-y_mc[q-1])*(r1-quantiles[q-1])/(1.00 - quantiles[q-1]) + y_mc[q-1]
        
        return y_tail


        #Following method should be used only after "def loadDFh5(self,rndm = 12345)" method is called. Also, if you want y_corrected branch as well, then use it after calling "trainQuantile" method       
    def getDF_test(self):
        #print self.hdf.head(5)
        return self.hdf_test

    def loadDFh5_corr(self,var):
        hdf_var_corr =  pd.read_hdf('/mnt/t3nfs01/data01/shome/krgedia/Workspace/lxplus/qreg_pt_norm/h5_corr_files/'+self.tree_name +'_'+var+'_corr'+ '.h5', 'hdf_corr')
        print "number of events:", len(hdf_var_corr.index)
        #self.hdf = hdf
        return hdf_var_corr

#    def getDF_corr(self):
 #       #print self.hdf_corr.tail(5)
#        return self.hdf_corr


    '''
    def renewDFh5(self):
        hdf_new = pd.HDFStore(self.tree_name + '.h5')
        hdf_new.put('hdf', self.hdf)
        self.hdf_new = hdf_new
#        print "number of events:", len(self.hdf.index)
        hdf_new.close()
    '''

#    def corrHDF(self):
        







