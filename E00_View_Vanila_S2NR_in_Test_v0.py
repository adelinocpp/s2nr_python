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
# -----------------------------------------------------------------------------
def levinson_aps(r, p):
    X = np.zeros((p,p))
    r_w = r[0:p]
    r_c = r_w[::-1]
    for idx in range(0,p):
        if (idx == 0):
            X[idx,:] = r_w
        else:
            X[idx,:] = np.roll(r_c,idx+1)
    b = -r[1:p+1]
    a = np.linalg.lstsq(X, b.T,rcond=None)[0]
    G = (r[0] - np.matmul(a.T,r[1:p+1]))
    a = np.concatenate(([1],a))
    return a, G
# -----------------------------------------------------------------------------
def lpc_aps(x,p):
    npts = len(x)
    hpts = int(np.ceil(0.5*npts))
    x_corr = signal.correlate(x,x, mode='same', method='fft')
    a, g = levinson_aps(x_corr[-hpts:], p)
    return a, g
# -----------------------------------------------------------------------------
def lpc_slice(x,p,f,sr):
    aL, gL = lpc_aps(x, p)
    sL = np.zeros((len(f),),dtype=np.complex128)
    k = 0
    omega = np.exp(1j*2*np.pi*f/sr)
    for an in aL:
        sL += an*np.power(omega,-k)
        k += 1
    H = np.abs(gL/sL)
    return H
# -----------------------------------------------------------------------------
def spec_peaks_slice(m,f):
    nM = len(m)
    pM = np.array([])
    pF = np.array([])
    pI = np.array([])
    for i in range(2,nM-2):
        if ( (m[i-2] < m[i]) and (m[i-1] < m[i]) and (m[i-2] < m[i]) and (m[i+1] < m[i]) ):
            vY = np.array([m[i-2], m[i-1], m[i],m[i+1],m[i+2]])
            vX = np.array([[f[i-2]**2, f[i-2], 1],
                           [f[i-1]**2, f[i-1], 1],
                           [f[i]**2,   f[i],   1],
                           [f[i+1]**2, f[i+1], 1],
                           [f[i+2]**2, f[i+2], 1]])
            b = np.matmul(np.matmul(np.linalg.inv(np.matmul(vX.T,vX)),vX.T),vY)
            yE = np.matmul(vX,b)
            Rsq1 = 1 - np.sum((vY - yE)**2)/np.sum((vY - np.mean(vY))**2)
            cX = -0.5*b[1]/b[0]
            if ((f[i-1] < cX) and (cX < f[i+1]) and (b[0] < 1) and (Rsq1 > 0.99)):
                cY = b[0]*(cX**2) + b[1]*cX + b[2]
                pM = np.append(pM,cY)
                pF = np.append(pF,cX)
                pI = np.append(pI,i)
    
    bM = np.array([])
    bF = np.array([])
    for i in range(0,len(pI)+1):
        if ((i == 0) and (pI[i] != 0)):
            iIni = 0
            iFim = int(pI[i])
        if (i > 0) and (i < (len(pI))):
            iIni = int(pI[i-1])
            iFim = int(pI[i])
        if (i == len(pI)) and (pI[i-1] != (len(m)-1)):
            iIni = int(pI[i-1])
            iFim = len(m)-1
        
        idx = np.argmin(m[iIni:iFim])
        bY = np.min(m[iIni:iFim])
        bM = np.append(bM,bY)
        bF = np.append(bF,f[iIni+idx])
    if (pF[0] != f[0]):
        pF = np.append(f[0],pF)
        pM = np.append(pM[0],pM)
    if (pF[-1] != f[-1]):
        pF = np.append(pF,f[-1])
        pM = np.append(pM,pM[-1])
    if (bF[0] != f[0]):
        bF = np.append(f[0],bF)
        bM = np.append(bM[0],bM)
    if (bF[-1] != f[-1]):
        bF = np.append(bF,f[-1])
        bM = np.append(bM,bM[-1])
    
    return pM, pF, bM, bF
# -----------------------------------------------------------------------------

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

TRY_PEAKS = True
TRY_LPC = False

if (TRY_PEAKS):
    gausWin = signal.windows.gaussian((nFim-nIni), std=(nFim-nIni))
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
    # for i in range(0,len(f)):
        
        # if (fAudio[i] > medC[i]):
            
        # else:
        
    
    
    # fig = plt.figure(figsize =(6, 6))
    # plt.title("Quadro de voz")
    # plt.ylabel("Magnitude")
    # plt.xlabel('tempo (s)')
    # plt.plot(f,logfAudio,'-', linewidth=1)
    # plt.plot(pF,pM,'rs', linewidth=1)
    # plt.plot(bF,bM,'gx', linewidth=1)
    # plt.grid(color='k',  linewidth=0.5)
    
    fig = plt.figure(figsize =(6, 6))
    plt.title("Quadro de voz")
    plt.ylabel("Magnitude")
    plt.xlabel('tempo (s)')
    plt.plot(f,nfAudio,'-', linewidth=1)
    # plt.plot(pF,pM,'rs', linewidth=1)
    # plt.plot(bF,bM,'gx', linewidth=1)
    # plt.plot(f,maxC,'r-.', linewidth=1)
    # plt.plot(f,minC,'g-.', linewidth=1)
    # plt.plot(f,medC,'b-', linewidth=2)
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
    
    
if (TRY_LPC):
    lpc_ord = int(np.floor(sr/1000) + 2)    
    hammWin = signal.windows.hamming((nFim-nIni))
    sAudio = signal.lfilter(np.array([1, -0.975]), np.array([1]), sAudio,axis=0)
    linear_spect = fft.fft(sAudio*hammWin, n=nFFT)
    fAudio = np.abs(linear_spect)[:hFFT]
    logfAudio = 20*np.log10(fAudio)
    t = np.array([i/sr for i in range(nIni,nFim)])
    f = np.array([i*0.5*sr/hFFT for i in range(0,hFFT)])
    a_lpc, g_lpc = lpc_aps(sAudio*hammWin, lpc_ord)
    Hm = lpc_slice(sAudio*hammWin, lpc_ord, f,sr)
    
    fig = plt.figure(figsize =(6, 6))
    plt.title("Quadro de voz")
    plt.ylabel("Magnitude")
    plt.xlabel('tempo (s)')
    plt.plot(t,sAudio,'-', linewidth=1)
    plt.grid(color='k',  linewidth=0.5)
    
    fig = plt.figure(figsize =(6, 6))
    plt.title("Quadro de voz")
    plt.ylabel("Magnitude")
    plt.xlabel('tempo (s)')
    plt.plot(f,logfAudio,'-', linewidth=1)
    plt.plot(f,20*np.log10(Hm),'r-', linewidth=1)
    plt.grid(color='k',  linewidth=0.5)
    
    fig = plt.figure(figsize =(6, 6))
    plt.title("Quadro de voz")
    plt.ylabel("Magnitude")
    plt.xlabel('tempo (s)')
    plt.plot(f,logfAudio,'-', linewidth=1)
    plt.plot(f,20*np.log10(Hm),'r-', linewidth=1)
    plt.plot(f,logfAudio - 20*np.log10(Hm),'g-', linewidth=1)
    plt.grid(color='k',  linewidth=0.5)
    
    
    sys.exit("DEPURA")
    
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


