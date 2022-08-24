# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import librosa
from S2NR import S2NR
import matplotlib.pyplot as plt
import numpy as np

filename = './Locutor_0006F_011.wav';
NFFT = 1024;    # fft window size 
RTH = 0.6;      # reliability threshold
alpha = 0.5;    # fft window overlap 
sigma = 0.1;    # image segmentation threshold

TimeStep = 0.01;
TimeWindow = 0.025;

audio, sr = librosa.load(filename, sr=None, mono=True)

vecH2NR, vecS2NR, mH2NR, mS2NR = S2NR(audio, sr,TimeWindow, TimeStep, NFFT , RTH, sigma)

time = [t for t in range(0,len(vecH2NR))]

fig = plt.figure(figsize=(10,8))
plt.plot(time,vecH2NR)
plt.plot(time,mH2NR*np.ones(len(vecH2NR),))
plt.plot(time,vecS2NR)
plt.plot(time,mS2NR*np.ones(len(vecS2NR),))


