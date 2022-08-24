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
import dill

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
lstResultH2NR = {}
lstResultS2NR = {}
lstResultS2NR_lin = {}
lstResultH2NR_lin = {}
lstResultH2NR_diff = {}
lstResultS2NR_diff = {}
lstResultH2NR_cor = {}
lstResultS2NR_cor = {}
lstResul = {}
lstTarget = {}
lstLabel = []
vecR2S2NR = []
vecR2H2NR = []
# fig = plt.figure(figsize =(6, 6))
xyS2NR = {}
xyH2NR = {}
modelS2NR = {}
modelH2NR = {}
k = 0
determCoef = os.path.join(EXPERIMENT_FOLDER, TXT_DETERMINATION_COEF)
if (not os.path.exists(determCoef)):
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
                    
                    lstResultH2NR[k] = np.array(H2NR_m[idx_sel] - SNR_target[idx_sel])
                    x = np.array(H2NR_m[idx_sel]).reshape((-1, 1))
                    y = np.array(SNR_target[idx_sel])
                    model = LinearRegression()
                    model.fit(x, y)
                    RScore = model.score(x, y)
                    lstResultH2NR_lin[k] = np.array([RScore, model.intercept_, model.coef_[0]])
                    lstResultH2NR_cor[k] = np.array(model.intercept_ + model.coef_[0]*H2NR_m[idx_sel] - SNR_target[idx_sel])
                    vecR2H2NR.append(RScore)
                    xyH2NR[k] = np.array([x.T[0],y])
                
                    lstResultS2NR[k] = np.array(S2NR_m[idx_sel] - SNR_target[idx_sel])
                    x = np.array(S2NR_m[idx_sel]).reshape((-1, 1))
                    y = np.array(SNR_target[idx_sel])
                    model = LinearRegression()
                    model.fit(x, y)
                    RScore = model.score(x, y)
                    lstResultS2NR_lin[k] = np.array([RScore, model.intercept_, model.coef_[0]])
                    lstResultS2NR_cor[k] = np.array(model.intercept_ + model.coef_[0]*S2NR_m[idx_sel] - SNR_target[idx_sel])
                    vecR2S2NR.append(RScore)
                    xyS2NR[k] = np.array([x.T[0],y])
                    
                    print('k - {:2d} R^2: {:5.3f}'.format(k,model.score(x, y)))
                    lstResultH2NR_diff[k] = np.array(H2NR_mdif[idx_sel])
                    lstResultS2NR_diff[k] = np.array(S2NR_mdif[idx_sel])
                    lstTarget[k] = np.array(SNR_target[idx_sel])
                    lstLabel.append('N_{:04d}_S_{:5.3f}_R_{:5.3f}_L_{:5.3f}'.format(iNFFT,iSigma,iRth,iWinLen))
                    k += 1
    lstResul['S2NR'] = lstResultS2NR
    lstResul['H2NR'] = lstResultH2NR
    lstResul['S2NR_diff'] = lstResultS2NR_diff
    lstResul['H2NR_diff'] = lstResultH2NR_diff
    lstResul['S2NR_lin'] = lstResultS2NR_lin
    lstResul['H2NR_lin'] = lstResultH2NR_lin
    lstResul['S2NR_cor'] = lstResultS2NR_cor
    lstResul['H2NR_cor'] = lstResultH2NR_cor
    lstResul['R2S2NR'] = vecR2S2NR
    lstResul['R2H2NR'] = vecR2H2NR
    lstResul['lstLabel'] = lstLabel
    lstResul['xyS2NR'] = xyS2NR
    lstResul['xyH2NR'] = xyH2NR
    lstResul['lstTarget'] = lstTarget
    dill_file = open(determCoef, "wb")
    dill.dump(lstResul,dill_file)
    dill_file.close()
else:
    dill_file = open(determCoef, "rb")
    lstResul = dill.load(dill_file)
    dill_file.close()
    lstLabel = lstResul['lstLabel']
    lstTarget = lstResul['lstTarget']
    lstResultS2NR_lin = lstResul['S2NR_lin']
    lstResultH2NR_lin = lstResul['H2NR_lin']
    vecR2S2NR = lstResul['R2S2NR']
    vecR2H2NR = lstResul['R2H2NR']
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
nResults = len(vecR2H2NR)
nSamples = int(0.1*nResults)
idxS2NR = np.argsort(vecR2S2NR)
idxH2NR = np.argsort(vecR2H2NR)
indexSelS2NR = []
indexSelH2NR = []
# for i in range(nResults-nSamples,nResults):
# ----------------------------------------------------------------------------
for i in range(1,nSamples+1):
    if np.isfinite(vecR2S2NR[idxS2NR[-i]]):
        indexSelS2NR.append(idxS2NR[-i])
    if np.isfinite(vecR2H2NR[idxH2NR[-i]]):
        indexSelS2NR.append(idxH2NR[-i])

# ----------------------------------------------------------------------------
lstLabel = np.array(lstLabel)
nSel = 1
combSel = []
nsRes = {}
meanRes = {}
absmeanRes = {}
stdmeanRes = {}
resTags = ['S2NR', 'H2NR','S2NR_diff', 'H2NR_diff','S2NR_cor','H2NR_cor']
for tag in resTags:
    nsRes[tag] = [len(lstResul[tag][i]) for i in range(0,100) ]
    meanRes[tag] = [np.mean(lstResul[tag][i]) for i in range(0,100) ]
    absmeanRes[tag] = [np.abs(np.mean(lstResul[tag][i])) for i in range(0,100) ]
    stdmeanRes[tag] = [np.std(lstResul[tag][i]) for i in range(0,100) ]
    xSort = np.argsort(absmeanRes[tag])
    [combSel.append(x) for x in xSort[:nSel] if x not in combSel]
    xSort = np.argsort(stdmeanRes[tag])
    [combSel.append(x) for x in xSort[:nSel] if x not in combSel]

T = np.array([int(i) for i in combSel])
selParamComb = lstLabel[T]
tabRes = np.empty([len(T),2*len(resTags)])
for idx, tag in enumerate(resTags[:4]):
    iA = 2*idx
    iB = 2*idx + 1
    tabRes[:,iA] = np.array(absmeanRes[tag])[T]
    tabRes[:,iB] = np.array(stdmeanRes[tag])[T]

for idx, tag in enumerate(resTags):
    fig = plt.figure(figsize =(6, 6))
    plt.title("Aferição do valor de SNR com {:}".format(tag))
    plt.ylabel("SNR desejado")
    plt.xlabel('{:}'.format(tag))
    for idxT in T:
        y = np.unique(lstTarget[idxT])
        x = np.zeros((len(y),))
        X = lstResul[tag][idxT]
        for i, yv in enumerate(y):
            idx = (y==yv).nonzero()[0]
            x[i] = np.mean(X[idx])
        plt.plot(x,y,'o',linestyle='-.', linewidth=1)
    plt.legend(T)
    plt.grid(color='k', linestyle='-.', linewidth=0.5)

nLines = np.min([len(indexSelS2NR),15])

xyS2NR = lstResul['xyS2NR']
xyH2NR = lstResul['xyH2NR']
fig = plt.figure(figsize =(6, 6))
plt.title("Aferição do valor de SNR")
plt.xlabel("SNR medido")
plt.ylabel('S2NR esperado')
for i in range(0,nLines):
    idxS = indexSelS2NR[i]
    X0 = xyH2NR[idxS][0]
    X1 = xyS2NR[idxS][0]
    Y = xyS2NR[idxS][1]
    X = np.array([X0,X1])
    model = LinearRegression()
    model.fit(X.T,Y)
    plt.scatter(X0,Y)
    x = np.linspace(X0.min(), X0.max(), num=10)
    y = model.intercept_ + model.coef_[0]*x
    Z = model.predict(X.T)
    plt.plot(x,y,linestyle='-.', linewidth=1)
    print("idx {:}, N {:}, STD: {:}".format(i,len(Y),np.std(Z-Y)))
        
        
    
    
# for idx, tag in enumerate(resTags):
#     fig = plt.figure(figsize =(6, 6))
#     plt.title("Aferição do valor de SNR com {:}".format(tag))
#     plt.xlabel("SNR desejado")
#     plt.ylabel('{:}'.format(tag))
#     for idxT in T:
#         Y = lstResul[tag][idxT]
        
#         X = lstTarget[idxT]
#         plt.scatter(X,Y)
#     plt.legend(T)
# # plt.grid(color='k', linestyle='-.', linewidth=0.5)
# # plt.show()


