#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 10:30:23 2022

@author: adelino
"""
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

plt.close('all')

EXPERIMENT_FOLDER = './Calculos_S2NR/'
CSV_EXPLORATORY_DATA_FILE = 'Prev_S2NR_Calibration_data.csv'

df = pd.read_csv(os.path.join(EXPERIMENT_FOLDER,CSV_EXPLORATORY_DATA_FILE))

SNR_target = df.loc[:,'SNR_target']
S2NR_m = df.loc[:,'S2NR_m']
H2NR_m = df.loc[:,'H2NR_m']
S2NR_mdif = df.loc[:,'dS2NR_m']
H2NR_mdif = df.loc[:,'dH2NR_m']

# Média S2NR
# Média H2NR
# Média da diferença S2NR
# Média da diferença H2NR
vecNFFT = df.loc[:,'NFFT']
vecRth = df.loc[:,'Rth']
vecSigma = df.loc[:,'sigma']
dataNFFT = vecNFFT.unique()
dataRth = vecRth.unique()
dataSigma = vecSigma.unique()
# Recortes
# NFFT


# Rth
# lstResult = {np.zeros((len(dataNFFT)*len(dataRth)*len(dataSigma),))}
lstResultH2NR = {}
lstResultS2NR = {}
lstResultH2NR_diff = {}
lstResultS2NR_diff = {}
lstTarget = {}
lstLabel = []
fig = plt.figure(figsize =(6, 6))
k = 0
for iNFFT in dataNFFT:
    for iSigma in dataSigma:
        for iRth in dataRth:
            vecResult = [(x and y and z) for x, y, z in zip((np.array(vecNFFT) == iNFFT), (np.array(vecSigma) == iSigma), (np.array(vecRth) == iRth))]
            vecResult = np.array(vecResult)
            idx_sel = vecResult.nonzero()[0]
            lstResultH2NR[k] = np.array(H2NR_m[idx_sel] - SNR_target[idx_sel])
            lstResultS2NR[k] = np.array(S2NR_m[idx_sel] - SNR_target[idx_sel])
            lstResultH2NR_diff[k] = np.array(H2NR_mdif[idx_sel])
            lstResultS2NR_diff[k] = np.array(S2NR_mdif[idx_sel])
            lstTarget[k] = np.array(SNR_target[idx_sel])
            lstLabel.append('N_{:04d}_S_{:5.3f}_R_{:5.3f}'.format(iNFFT,iSigma,iRth))
            k += 1

lstLabel = np.array(lstLabel)
nSel = 6
combSel = []
nsRes = {}
meanRes = {}
absmeanRes = {}
stdmeanRes = {}
#
nsRes['S2NR'] = [len(lstResultS2NR[i]) for i in range(0,100) ]
meanRes['S2NR'] = [np.mean(lstResultS2NR[i]) for i in range(0,100) ]
absmeanRes['S2NR'] = [np.abs(np.mean(lstResultS2NR[i])) for i in range(0,100) ]
stdmeanRes['S2NR'] = [np.std(lstResultS2NR[i]) for i in range(0,100) ]
xSort = np.argsort(absmeanRes['S2NR'])
[combSel.append(x) for x in xSort[:nSel] if x not in combSel]
xSort = np.argsort(stdmeanRes['S2NR'])
[combSel.append(x) for x in xSort[:nSel] if x not in combSel]
#
nsRes['H2NR'] = [len(lstResultH2NR[i]) for i in range(0,100) ]
meanRes['H2NR'] = [np.mean(lstResultH2NR[i]) for i in range(0,100) ]
absmeanRes['H2NR'] = [np.abs(np.mean(lstResultH2NR[i])) for i in range(0,100) ]
stdmeanRes['H2NR'] = [np.std(lstResultH2NR[i]) for i in range(0,100) ]
xSort = np.argsort(absmeanRes['H2NR'])
[combSel.append(x) for x in xSort[:nSel] if x not in combSel]
xSort = np.argsort(stdmeanRes['H2NR'])
[combSel.append(x) for x in xSort[:nSel] if x not in combSel]
#
nsRes['S2NR_diff'] = [len(lstResultS2NR_diff[i]) for i in range(0,100) ]
meanRes['S2NR_diff'] = [np.mean(lstResultS2NR_diff[i]) for i in range(0,100) ]
absmeanRes['S2NR_diff'] = [np.abs(np.mean(lstResultS2NR_diff[i])) for i in range(0,100) ]
stdmeanRes['S2NR_diff'] = [np.std(lstResultS2NR_diff[i]) for i in range(0,100) ]
xSort = np.argsort(absmeanRes['S2NR_diff'])
[combSel.append(x) for x in xSort[:nSel] if x not in combSel]
xSort = np.argsort(stdmeanRes['S2NR_diff'])
[combSel.append(x) for x in xSort[:nSel] if x not in combSel]
#
nsRes['H2NR_diff'] = [len(lstResultH2NR_diff[i]) for i in range(0,100) ]
meanRes['H2NR_diff'] = [np.mean(lstResultH2NR_diff[i]) for i in range(0,100) ]
absmeanRes['H2NR_diff'] = [np.abs(np.mean(lstResultH2NR_diff[i])) for i in range(0,100) ]
stdmeanRes['H2NR_diff'] = [np.std(lstResultH2NR_diff[i]) for i in range(0,100) ]
xSort = np.argsort(absmeanRes['H2NR_diff'])
[combSel.append(x) for x in xSort[:nSel] if x not in combSel]
xSort = np.argsort(stdmeanRes['H2NR_diff'])
[combSel.append(x) for x in xSort[:nSel] if x not in combSel]
#
T = np.array([int(i) for i in combSel])
selParamComb = lstLabel[T]
np.array(absmeanRes['S2NR'])[T]
np.array(stdmeanRes['S2NR'])[T]
np.array(absmeanRes['H2NR'])[T]
np.array(stdmeanRes['H2NR'])[T]
np.array(absmeanRes['S2NR_diff'])[T]
np.array(stdmeanRes['S2NR_diff'])[T]
np.array(absmeanRes['H2NR_diff'])[T]
np.array(stdmeanRes['H2NR_diff'])[T]

# TODO: Analisar as tendências dos mínimos
