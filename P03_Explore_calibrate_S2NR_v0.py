#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 10:30:23 2022

@author: adelino
"""
from IPython import get_ipython

get_ipython().magic('reset -sf')

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
EXPERIMENT_FOLDER = './Calculos_S2NR/'
CSV_EXPLORATORY_DATA_FILE = 'Prev_S2NR_Calibration_data.csv'

df = pd.read_csv(os.path.join(EXPERIMENT_FOLDER,CSV_EXPLORATORY_DATA_FILE))

SNR_target = df.loc[:,'SNR_target']
S2NR_m = df.loc[:,'S2NR_m']
H2NR_m = df.loc[:,'H2NR_m']
S2NR_mdif = df.loc[:,'dS2NR_m']
H2NR_mdif = df.loc[:,'dH2NR_m']

# Média S2NR
X = np.array(SNR_target)
Y = np.array(S2NR_m)
fig = plt.figure(figsize =(6, 6))
plt.scatter(X,Y)
plt.title("Aferição do valor de S2NR")
plt.xlabel("SNR desejado")
plt.ylabel("S2NR medido")
plt.grid(color='k', linestyle='-', linewidth=0.5)
plt.show()
# Média H2NR
X = np.array(SNR_target)
Y = np.array(H2NR_m)
fig = plt.figure(figsize =(6, 6))
plt.scatter(X,Y)
plt.title("Aferição do valor de S2NR")
plt.xlabel("HNR desejado")
plt.ylabel("H2NR medido")
plt.grid(color='k', linestyle='-.', linewidth=0.5)
plt.show()

# Média da diferença S2NR
X = np.array(SNR_target)
Y = np.array(S2NR_mdif)
fig = plt.figure(figsize =(6, 6))
plt.scatter(X,Y)
plt.title("Aferição do valor de S2NR")
plt.xlabel("SNR desejado")
plt.ylabel("média da diferenaça  S2NR medido")
plt.grid(color='k', linestyle='-', linewidth=0.5)
plt.show()
# Média da diferença H2NR
X = np.array(SNR_target)
Y = np.array(H2NR_mdif)
fig = plt.figure(figsize =(6, 6))
plt.scatter(X,Y)
plt.title("Aferição do valor de S2NR")
plt.xlabel("HNR desejado")
plt.ylabel("média da diferenaça H2NR medido")
plt.grid(color='k', linestyle='-.', linewidth=0.5)
plt.show()

vecNFFT = df.loc[:,'NFFT']
vecRth = df.loc[:,'Rth']
vecSigma = df.loc[:,'sigma']
dataNFFT = vecNFFT.unique()
dataRth = vecRth.unique()
dataSigma = vecSigma.unique()
# Recortes
# NFFT


# Rth
for iRth in dataRth:
    idx_sel = (vecRth == iRth).nonzero()[0]
    X = np.array(SNR_target[idx_sel])
    Y = np.array(H2NR_mdif[idx_sel])
    fig = plt.figure(figsize =(6, 6))
    plt.scatter(X,Y)
    plt.title("Diferença de H2NR com Rth")
    plt.xlabel("HNR desejado")
    plt.ylabel("média da diferenaça H2NR medido")
    plt.grid(color='k', linestyle='-.', linewidth=0.5)
    plt.show()
plt.legend()
# Sigma

