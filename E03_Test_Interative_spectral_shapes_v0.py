#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 13:44:12 2022

@author: adelino
"""
from S2NR2 import S2NR2
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pickle
from scipy import signal, fft
from scipy.io import wavfile
from scipy.interpolate import pchip_interpolate


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

NFFT = 1024;    # fft window size 
RTH = 0.6;      # reliability threshold
alpha = 0.5;    # fft window overlap 
sigma = 0.1;    # image segmentation threshold

TimeStep = 0.01;
TimeWindow = 0.025;

win_length = 0.025
tIni = 1
tFim = 2
nIni = int(tIni*sr)
nFim = int(tFim*sr)

# sAudio = audio[nIni:nFim]
sAudio = audio[nIni:nFim]
vecH2NR, vecS2NR, mH2NR, mS2NR = S2NR2(sAudio, sr,TimeWindow, TimeStep, NFFT , RTH, sigma,mmNorm=True)

time = [t/sr for t in range(0,len(vecH2NR))]

fig = plt.figure(figsize=(10,8))
plt.plot(time,vecH2NR)
plt.plot(time,mH2NR*np.ones(len(vecH2NR),))
plt.plot(time,vecS2NR)
plt.plot(time,mS2NR*np.ones(len(vecS2NR),))


