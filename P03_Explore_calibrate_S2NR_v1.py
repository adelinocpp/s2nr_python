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
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
import dill
import sys
plt.close('all')

EXPERIMENT_FOLDER = './Calculos_S2NR/'
CSV_EXPLORATORY_DATA_FILE = 'Prev_S2NR_Calibration_data.csv'
TXT_DETERMINATION_COEF = 'determination_coef_v0.txt'

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
vecWinLength = df.loc[:,'winLength']
dataNFFT = vecNFFT.unique()
dataRth = vecRth.unique()
dataSigma = vecSigma.unique()
dataWinLength = vecWinLength.unique()
# Recortes
# NFFT


# Rth
# lstResult = {np.zeros((len(dataNFFT)*len(dataRth)*len(dataSigma),))}
lstResul = {}
lstLabel = []
vecIdxSel = []
vecR2 = []
vecB2 = []
vecR2S = []
vecR2H = []
vecCPS = []
vecCPH = []
newRUN = True
# fig = plt.figure(figsize =(6, 6))
k = 0
determCoef = os.path.join(EXPERIMENT_FOLDER, TXT_DETERMINATION_COEF)
if (not os.path.exists(determCoef)) or newRUN:
    for iWinLen in dataWinLength:
        for iNFFT in dataNFFT:
            for iSigma in dataSigma:
                for iRth in dataRth:
                    vecResult = [(x and y and z and w) for x, y, z, w in zip((np.array(vecNFFT) == iNFFT), (np.array(vecSigma) == iSigma), (np.array(vecRth) == iRth), (np.array(vecWinLength) == iWinLen))]
                    vecResult = np.array(vecResult)
                    idx_sel = vecResult.nonzero()[0]
                    if (len(idx_sel) == 0):
                        print("NFFT: {:}({:}); sigma: {:}({:}); Rth: {:}({:}); WinLength: {:}({:}).".
                              format(iNFFT,len(np.array(vecNFFT == iNFFT).nonzero()[0]),
                                     iSigma,len(np.array(vecSigma == iSigma).nonzero()[0]),
                                     iRth,len(np.array(vecRth == iRth).nonzero()[0]),
                                     iWinLen,len(np.array(vecWinLength == iWinLen).nonzero()[0])))
                        continue
                    # -- duas vairaveis
                    x = np.concatenate((np.array(S2NR_m[idx_sel]).reshape((-1, 1)), \
                                        np.array(H2NR_m[idx_sel]).reshape((-1, 1))),axis=1)
                    y = np.array(SNR_target[idx_sel])
                    model = LinearRegression()
                    model.fit(x, y)
                    RScore = model.score(x, y)
                    vecR2.append(RScore)
                    
                    reg = BayesianRidge(tol=1e-6, fit_intercept=False, compute_score=True)
                    reg.fit(x, y)
                    RScore = reg.score(x, y)
                    vecB2.append(RScore)
                    
                    # --- Uma variável
                    x = np.array(S2NR_m[idx_sel]).reshape((-1, 1))
                    model = LinearRegression()
                    model.fit(x, y)
                    RScore = model.score(x, y)
                    vecR2S.append(RScore)
                    vecCPS.append(np.corrcoef(S2NR_m[idx_sel],SNR_target[idx_sel])[0,1])
                    
                    x = np.array(H2NR_m[idx_sel]).reshape((-1, 1))
                    model = LinearRegression()
                    model.fit(x, y)
                    RScore = model.score(x, y)
                    vecR2H.append(RScore)
                    vecCPH.append(np.corrcoef(H2NR_m[idx_sel],SNR_target[idx_sel])[0,1])
                    
                    
                    vecIdxSel.append(idx_sel)
                    print('k - {:2d} R^2: {:5.3f}'.format(k,model.score(x, y)))
                    lstLabel.append('N_{:04d}_S_{:5.3f}_R_{:5.3f}_L_{:5.3f}'.format(iNFFT,iSigma,iRth,iWinLen))
                    k += 1
    lstResul['R2'] = vecR2
    lstResul['B2'] = vecB2
    lstResul['R2S'] = vecR2S
    lstResul['B2H'] = vecR2H
    lstResul['CPS'] = vecCPS
    lstResul['CPH'] = vecCPH
    lstResul['lstLabel'] = lstLabel
    lstResul['vecIdxSel'] = vecIdxSel
    dill_file = open(determCoef, "wb")
    dill.dump(lstResul,dill_file)
    dill_file.close()
else:
    dill_file = open(determCoef, "rb")
    lstResul = dill.load(dill_file)
    dill_file.close()
    lstLabel = lstResul['lstLabel']
    vecR2 = lstResul['R2']
    vecB2 = lstResul['B2']
    vecR2S = lstResul['R2S']
    vecR2H = lstResul['B2H']
    vecCPS = lstResul['CPS']
    vecCPH = lstResul['CPH']
    vecIdxSel = lstResul['vecIdxSel']
    
# ----------------------------------------------------------------------------
numplot = 5

idxS = np.argsort(vecCPS)[-numplot:]
T = [lstLabel[i] for i in idxS]

fig = plt.figure(figsize =(6, 6))
plt.title("Aferição do valor de SNR")
plt.ylabel("SNR desejado")
plt.xlabel('S2NR medido')

for i in idxS:
    idx_sel = vecIdxSel[i]
    y = np.array(SNR_target[idx_sel])
    x = np.array(S2NR_m[idx_sel])
    plt.plot(x,y,'x', linewidth=1)
plt.legend(T)
plt.grid(color='k',  linewidth=0.5)


idxS = np.argsort(vecCPH)[-numplot:]
T = [lstLabel[i] for i in idxS]

fig = plt.figure(figsize =(6, 6))
plt.title("Aferição do valor de SNR")
plt.ylabel("SNR desejado")
plt.xlabel('H2NR medido')

for i in idxS:
    idx_sel = vecIdxSel[i]
    y = np.array(SNR_target[idx_sel])
    x = np.array(H2NR_m[idx_sel])
    plt.plot(x,y,'x', linewidth=1)
plt.legend(T)
plt.grid(color='k',  linewidth=0.5)