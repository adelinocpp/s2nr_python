#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 08:06:41 2022

@author: adelino
"""
from scipy import signal
import matplotlib.pyplot as plt
plt.close('all')
import numpy as np



vNFFT = [512, 1024, 2048]
vDIV = [5, 7, 9]
gColor = ['g-','b-','r-']

for nFFT in vNFFT:
    t = np.linspace(0,1,nFFT)
    hannWin = signal.windows.hann(nFFT) 
    hammWin = signal.windows.hamming(nFFT)
    fig = plt.figure(figsize =(6, 6))
    plt.title("Janelas espectrais n:{:}".format(nFFT))
    plt.ylabel("Magnitude")
    plt.xlabel('tempo (s)')
    plt.plot(t,hannWin,'c-', linewidth=1)
    plt.plot(t,hammWin,'m-', linewidth=1)
    for idx, nDIV in enumerate(vDIV):
        gausWin = signal.windows.gaussian(nFFT, std=nFFT/nDIV)
        plt.plot(t,gausWin,gColor[idx], linewidth=1)
    plt.grid(color='k',  linewidth=0.5)