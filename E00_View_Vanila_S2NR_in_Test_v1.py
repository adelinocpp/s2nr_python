# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# import librosa
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
from utils.sigprocess import spec_peaks_slice

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
nFFT = 512
hFFT = int(0.5*nFFT)
nIni = int(tIni*sr)
nFim = int(tFim*sr)
sAudio = audio[nIni:nFim]

TRY_PEAKS = False

if (TRY_PEAKS):
    gausWin = signal.windows.gaussian((nFim-nIni), std=(nFim-nIni)/4)
    linear_spect = fft.fft(sAudio*gausWin, n=nFFT)
    fAudio = np.abs(linear_spect)[:hFFT]
    logfAudio = 20*np.log10(fAudio)
    t = np.array([i/sr for i in range(nIni,nFim)])
    f = np.array([i*0.5*sr/hFFT for i in range(0,hFFT)])
    pM, pF, bM, bF = spec_peaks_slice(fAudio,f)
    
    maxVal = np.max(pM)
    minVal = np.min(bM)
    maxC = pchip_interpolate(pF, pM,f)
    minC = pchip_interpolate(bF, bM,f)
    # maxC = np.interp(f,pF, pM)
    # minC = np.interp(f,bF, bM)
    maxC = np.vstack([maxC,fAudio]).max(axis=0)
    minC = np.vstack([minC,fAudio]).min(axis=0)
    medC = 0.5*(maxC+minC)
    fAudio = fAudio
    nfAudio = np.divide((fAudio - minC),(maxC-minC)) + minVal
   
    
    fig = plt.figure(figsize =(6, 6))
    plt.title("Quadro de voz")
    plt.ylabel("Magnitude")
    plt.xlabel('tempo (s)')
    plt.plot(f,nfAudio,'-', linewidth=1)
    plt.grid(color='k',  linewidth=0.5)
    
    fig = plt.figure(figsize =(6, 6))
    plt.title("Quadro de voz")
    plt.ylabel("Magnitude")
    plt.xlabel('tempo (s)')
    plt.plot(f,logfAudio,'-', linewidth=1)
    plt.plot(pF,20*np.log10(pM),'rs', linewidth=1)
    plt.plot(bF,20*np.log10(bM),'gx', linewidth=1)
    plt.plot(f,20*np.log10(maxC),'r-.', linewidth=1)
    plt.plot(f,20*np.log10(minC),'g-.', linewidth=1)
    plt.plot(f,20*np.log10(medC),'b-', linewidth=2)
    plt.grid(color='k',  linewidth=0.5)
    

NFFT = 1024;    # fft window size 
RTH = 0.6;      # reliability threshold
alpha = 0.5;    # fft window overlap 
sigma = 0.1;    # image segmentation threshold

TimeStep = 0.01;
TimeWindow = 0.025;

vecH2NR, vecS2NR, mH2NR, mS2NR = S2NR(audio, sr,TimeWindow, TimeStep, NFFT , RTH, sigma, True)

time = [t for t in range(0,len(vecH2NR))]

fig = plt.figure(figsize=(10,8))
plt.plot(time,vecH2NR)
plt.plot(time,mH2NR*np.ones(len(vecH2NR),))
plt.plot(time,vecS2NR)
plt.plot(time,mS2NR*np.ones(len(vecS2NR),))


