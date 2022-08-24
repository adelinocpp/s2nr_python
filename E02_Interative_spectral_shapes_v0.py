#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 13:44:12 2022

@author: adelino
"""

from S2NR import S2NR
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pickle
from scipy import signal, fft
from scipy.io import wavfile
from scipy.interpolate import pchip_interpolate
from utils.sigprocess import spec_peaks_slice, build_mel_triang_filters

plt.close('all')

CALIBRATE_FOLDER = './Vozes_Calibracao/'
EXPERIMENT_FOLDER = './Calculos_S2NR/'
CSV_EXPLORATORY_DATA_FILE = 'Prev_S2NR_Calibration_data.csv'
feature_list = glob(os.path.join(CALIBRATE_FOLDER, '*.p'), recursive=True)
feature_list.sort()
nfeature_list = len(feature_list)
nSelFlist = 42
selFeature = feature_list[nSelFlist]

with open(selFeature, 'rb') as f:
    suport_feature = pickle.load(f)
    
wavename = os.path.join(CALIBRATE_FOLDER, suport_feature['filename'])
sr, audio = wavfile.read(wavename)

win_length = 0.025
tIni = 0.250
tFim = 0.275
nFFT = 2048
hFFT = int(0.5*nFFT)
nIni = int(tIni*sr)
nFim = int(tFim*sr)

sAudio = audio[nIni:nFim]

epsTol = 1e-5
iteractive=True
gausWin = signal.windows.gaussian((nFim-nIni), std=(nFim-nIni)/5)
linear_spect = fft.fft(sAudio*gausWin, n=nFFT)
fAudio = np.abs(linear_spect)[:hFFT]
logfAudio = 20*np.log10(fAudio)
t = np.array([i/sr for i in range(nIni,nFim)])
f = np.array([i*0.5*sr/hFFT for i in range(0,hFFT)])
pM, pF, bM, bF, pI, bI = spec_peaks_slice(logfAudio,f)
minVal = np.min(bM)
maxVal = np.max(pM)

maxC = pchip_interpolate(pF, pM,f)
minC = pchip_interpolate(bF, bM,f)

if (iteractive):
    upMaxIdx = ((logfAudio - maxC) > epsTol).nonzero()[0]
    while (len(upMaxIdx) > 0):
        for idx in range(0,len(pI)-1):
            idxR = np.array([x and y for x,y in zip(upMaxIdx > pI[idx],upMaxIdx <= pI[idx+1])]).nonzero()[0]
            if (len(idxR) < 1):
                continue
            idxB = upMaxIdx[idxR]
            idxIns = np.argmax(logfAudio[idxB])
            tpI = np.concatenate((pI[:(idx+1)], np.array([idxB[idxIns]]), pI[(idx+1):]))
            break   
        pI = np.array(tpI,dtype=np.int32)
        pF = f[pI]
        # pM = logfAudio[pI]
        pM = np.concatenate( (pM[:1], logfAudio[pI[1:-1]], pM[-1:] ) )
        maxCX = pchip_interpolate(pF, pM,f)    
        upMaxIdx = ((logfAudio - maxCX) > epsTol).nonzero()[0]
        if (len(upMaxIdx) == 1) and ( (upMaxIdx[0] == 0) or (upMaxIdx[0] == (len(logfAudio) - 1))):
            break
    
    
    dwMinIdx = ((minC - logfAudio) > epsTol).nonzero()[0]
    while (len(dwMinIdx) > 0):
        for idx in range(0,len(bI)-1):
            idxR = np.array([x and y for x,y in zip(dwMinIdx > bI[idx],dwMinIdx <= bI[idx+1])]).nonzero()[0]        
            if (len(idxR) < 1):
                continue
            idxB = dwMinIdx[idxR]
            idxIns = np.argmin(logfAudio[idxB])
            tbI = np.concatenate((bI[:(idx+1)], np.array([idxB[idxIns]]), bI[(idx+1):]))
            break   
        bI = np.array(tbI,dtype=np.int32)
        bF = f[bI]
        # bM = logfAudio[bI]
        bM = np.concatenate( (bM[:1], logfAudio[bI[1:-1]], bM[-1:] ) )
        minCX = pchip_interpolate(bF, bM,f) 
        dwMinIdx = ((minCX - logfAudio) > epsTol).nonzero()[0]
        if (len(dwMinIdx) == 1) and ( (dwMinIdx[0] == 0) or (dwMinIdx[0] == (len(logfAudio) - 1))):
            break


medC = 0.5*(maxC+minC)
difC = (maxC-minC)
medCL = np.power(10,0.5*(maxC+minC)/20)

# maxCX = maxC + difC #maxC + np.std(medC) #- 0.5*(np.min(maxC)- np.max(minC)) # medC + 0.5*(maxVal -minVal) + 0.5*(np.min(maxC)- np.max(minC))
# minCX = minC - difC#minC - np.std(medC) # + 0.5*(np.min(maxC)- np.max(minC)) #medC - 0.5*(maxVal -minVal) - 0.5*(np.min(maxC)- np.max(minC))


wts, vFmid = build_mel_triang_filters(nFFT,sr,nfilts=29)


fig = plt.figure(figsize =(6, 6))
plt.title("Quadro de voz")
plt.ylabel("Magnitude")
plt.xlabel('tempo (s)')
plt.plot(f,logfAudio,'g-', linewidth=1)
plt.plot(f,maxC,'b-', linewidth=2)
plt.plot(f,minC,'r-', linewidth=2)
plt.plot(f,maxCX,'c-', linewidth=2)
plt.plot(f,minCX,'m-', linewidth=2)
plt.grid(color='k',  linewidth=0.5)

fig = plt.figure(figsize =(6, 6))
plt.title("Quadro de voz")
plt.ylabel("Magnitude")
plt.xlabel('tempo (s)')
plt.plot(f,logfAudio,'g-', linewidth=1)
plt.plot(f,maxC,'c-', linewidth=2)
plt.plot(f,minC,'m-', linewidth=2)
plt.grid(color='k',  linewidth=0.5)

nfAudio = np.divide((logfAudio - minCX),(maxCX-minCX)) + minVal


fig = plt.figure(figsize =(6, 6))
plt.title("Quadro de voz")
plt.ylabel("Magnitude")
plt.xlabel('tempo (s)')
plt.plot(f,nfAudio,'g-', linewidth=1)
plt.grid(color='k',  linewidth=0.5)

    
sys.exit("DEPURA")
vFeat = np.zeros((vFmid.shape))
fig = plt.figure(figsize =(6, 6))
plt.title("filtros")
plt.ylabel("Magnitude")
plt.xlabel('tempo (s)')
for i in range(0,wts.shape[0]):
    plt.plot(f,wts[i,:],'-', linewidth=1)
    vFeat[i] = np.log(np.mean(wts[i,:]*medCL))
plt.grid(color='k',  linewidth=0.5)


fig = plt.figure(figsize =(6, 6))
plt.title("Quadro de voz")
plt.ylabel("Magnitude")
plt.xlabel('tempo (s)')
plt.plot(f,logfAudio,'-', linewidth=1)
plt.plot(f,20*np.log10(medC),'b-', linewidth=2)
plt.plot(vFmid,vFeat,'rx', linewidth=2)
plt.grid(color='k',  linewidth=0.5)
    

