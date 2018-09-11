#!/bin/bash

k=0.1

#file_dir = "/mnt/t3nfs01/data01/shome/krgedia/Workspace/lxplus/qreg/qreg_file.root"
#tree_sw_mc = "t_dataset_sig"
#tree_sw_data = "t_r_dataset_sig"

#for var in "Jet_rawPt"; #"Jet_chHEF" "Jet_neEmEF" "Jet_leadTrackPt" "Jet_ptd"; #"Jet_rawEnergy" "Jet_mass" "Jet_rawEnergy" "Jet_rawPt" "Jet_chHEF" "Jet_neEmEF" "Jet_leadTrackPt" "Jet_ptd";

#for var in "Jet_chHEF" "Jet_leadTrackPt" "Jet_ptd" "Jet_mass" "Jet_rawEnergy" "Jet_rawPt" "Jet_chHEF" "Jet_neEmEF";
#for var in "Jet_vtxMass" "Jet_vtxPt" "Jet_vtx3DSig" "Jet_vtx3DVal" "Jet_ptd" "Jet_neHEF" "Jet_muEF" "Jet_chEmEF" "Jet_energyRing_dR0_neut" "Jet_energyRing_dR1_neut" "Jet_energyRing_dR2_neut" "Jet_energyRing_dR3_neut" "Jet_energyRing_dR4_neut" "Jet_energyRing_dR0_ch" "Jet_energyRing_dR1_ch" "Jet_energyRing_dR2_ch" "Jet_energyRing_dR3_ch" "Jet_energyRing_dR4_ch" "Jet_energyRing_dR0_em" "Jet_energyRing_dR1_em" "Jet_energyRing_dR2_em" "Jet_energyRing_dR3_em" "Jet_energyRing_dR4_em" "Jet_energyRing_dR0_mu" "Jet_energyRing_dR1_mu" "Jet_energyRing_dR2_mu" "Jet_energyRing_dR3_mu" "Jet_energyRing_dR4_mu" "Jet_mass" "Jet_rawEnergy" "Jet_rawPt" "Jet_chHEF" "Jet_neEmEF" "Jet_leadTrackPt" "Jet_leptonDeltaR" "Jet_leptonPt" "Jet_leptonPtRel" "Jet_leptonPtRelInv";
#for var in "Jet_leptonDeltaR" "Jet_leptonPt" "Jet_leptonPtRel" "Jet_leptonPtRelInv" "Jet_vtxMass" "Jet_vtxPt" "Jet_vtx3DSig" "Jet_vtx3DVal";
for var in "Jet_vtx3DVal";
do
    echo Jobs for $var
    for q in 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 0.99;
    do
	#q=`echo "$k*$i"|bc`

        #qsub jobba.sh "sw_mc" "/mnt/t3nfs01/data01/shome/krgedia/Workspace/lxplus/qreg/qreg_file.root" "t_dataset_sig" "Jet_chHEF" 0.1 3 9 -q short.q	
        qsub -l h_vmem=5g jobba.sh "sw_mc" "/mnt/t3nfs01/data01/shome/krgedia/Workspace/lxplus/qreg_pt_norm/qreg_pt_norm.root" "t_dataset_sig" $var $q 3 9 -q long.q
        qsub -l h_vmem=5g jobba.sh "sw_data" "/mnt/t3nfs01/data01/shome/krgedia/Workspace/lxplus/qreg_pt_norm/qreg_pt_norm.root" "t_r_dataset_sig" $var $q 3 9 -q long.q
        qsub -l h_vmem=5g jobba.sh "mc" "/mnt/t3nfs01/data01/shome/krgedia/Workspace/lxplus/qreg_pt_norm/qreg_pt_norm.root" "mc_sig" $var $q 3 9 -q long.q
        #qsub jobba.sh "sw_mc" "/mnt/t3nfs01/data01/shome/krgedia/Workspace/lxplus/qreg_pt_norm/qreg_pt_norm.root" "t_dataset_sig" $var $q 3 9 -q long.q
        #qsub jobba.sh "sw_data" "/mnt/t3nfs01/data01/shome/krgedia/Workspace/lxplus/qreg_pt_norm/qreg_pt_norm.root" "t_r_dataset_sig" $var $q 3 9 -q long.q
        #qsub jobba.sh "mc" "/mnt/t3nfs01/data01/shome/krgedia/Workspace/lxplus/qreg_pt_norm/qreg_pt_norm.root" "mc_sig" $var $q 3 9 -q long.q


	#qsub jobba.sh "sw_mc" $file_dir $tree_sw_mc $var $q 3 9 -q short.q
        #qsub jobba.sh "sw_data" $file_dir $tree_sw_data $var $q 3 9 -q short.q
	#qsub -q all.q -l h_vmem=6G jobba.sh "mc"   $var $q 0 -1 3 9 "EE"
      	#qsub -q all.q -l h_vmem=6G jobba.sh "data" $var $q 0 -1 3 9 "EB"
	#qsub -q all.q -l h_vmem=6G jobba.sh "data" $var $q 0 -1 3 9 "EE"
	#qsub -q all.q -l h_vmem=6G jobba.sh "data" $var $q 0 -1 3 9 "EBEE"
	#qsub -q all.q -l h_vmem=6G jobba.sh "mc"   $var $q 0 -1 3 9 "EBEE"
	#qsub -q all.q -l h_vmem=6G jobba_woRF.sh "mc"   $var $q 0 -1 3 9 "EB" 
	#qsub -q all.q -l h_vmem=6G jobba_woRF.sh "mc"   $var $q 0 -1 3 9 "EE"
      	#qsub -q all.q -l h_vmem=6G jobba_woRF.sh "data" $var $q 0 -1 3 9 "EB"
	#qsub -q all.q -l h_vmem=6G jobba_woRF.sh "data" $var $q 0 -1 3 9 "EE"
	#qsub -q all.q -l h_vmem=6G jobba_woRF.sh "data" $var $q 0 -1 3 9 "EBEE"
	#qsub -q all.q -l h_vmem=6G jobba_woRF.sh "mc"   $var $q 0 -1 3 9 "EBEE"
	#qsub -q all.q -l h_vmem=6G jobba_RunC.sh "mc"   $var $q 0 -1 3 9 "EB" 
	#qsub -q all.q -l h_vmem=6G jobba_RunC.sh "mc"   $var $q 0 -1 3 9 "EE"
      	#qsub -q all.q -l h_vmem=6G jobba_RunC.sh "data" $var $q 0 -1 3 9 "EB"
	#qsub -q all.q -l h_vmem=6G jobba_RunC.sh "data" $var $q 0 -1 3 9 "EE"
	#qsub -q all.q -l h_vmem=6G jobba_RunC.sh "data" $var $q 0 -1 3 9 "EBEE"
	#qsub -q all.q -l h_vmem=6G jobba_RunC.sh "mc"   $var $q 0 -1 3 9 "EBEE"
	done
done
