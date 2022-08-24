#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 17:48:00 2022

@author: adelino
"""
import numpy as np

# -----------------------------------------------------------------------------
def hertz2mel(hertz_freq,Slaney=False):
    if (Slaney):
        f_0 = 0 # 133.33333;
        f_sp = 200/3 # 66.66667;
        brkfrq = 1000
        brkpt  = (brkfrq - f_0)/f_sp;  # starting mel value for log region
        logstep = np.exp(np.log(6.4)/27); # the magic 1.0711703 which is the ratio needed to get from 1000 Hz to 6400 Hz in 27 steps, and is *almost* the ratio between 1000 Hz and the preceding linear filter center at 933.33333 Hz (actually 1000/933.33333 = 1.07142857142857 and  exp(log(6.4)/27) = 1.07117028749447)
        if (hertz_freq < brkfrq):
            mb = (hertz_freq - f_0)/f_sp;
        else:
            mb =  brkpt+(np.log(hertz_freq/brkfrq))/np.log(logstep);
    else:
        k=1000/np.log(1+1000/700) # 1127.01048
        af = np.abs(hertz_freq)
        mb = np.sign(hertz_freq)*np.log(1+af/700)*k
    return mb
# -----------------------------------------------------------------------------
def mel2hertz(mel_freq,Slaney=False):
    if (Slaney):
        f_0 = 0 # 133.33333;
        f_sp = 200/3 # 66.66667;
        brkfrq = 1000
        brkpt  = (brkfrq - f_0)/f_sp # starting mel value for log region
        logstep = np.exp(np.log(6.4)/27) # the magic 1.0711703 which is the ratio 
        # needed to get from 1000 Hz to 6400 Hz in 27 steps, and is *almost* the 
        # ratio between 1000 Hz and the preceding linear filter center at 933.33333 Hz 
        # (actually 1000/933.33333 = 1.07142857142857 and  exp(log(6.4)/27) = 1.07117028749447)

        if (mel_freq < brkpt):
            mb = f_0 + f_sp*mel_freq
        else:
            mb = brkfrq*np.exp(np.log(logstep)*(mel_freq-brkpt))
    else:
        k=1000/np.log(1+1000/700) # 1127.01048
        am = np.abs(mel_freq)
        mb = 700*np.sign(mel_freq)*(np.exp(am/k)-1)
    return mb
# -----------------------------------------------------------------------------
def build_mel_triang_filters(nfft,sr,nfilts=0,min_freq=0,max_freq=0,Slaney=False):
    if (max_freq == 0):
        max_freq = 0.5*sr
    min_mel = hertz2mel(min_freq,Slaney)
    nyq_mel = hertz2mel(max_freq,Slaney) - min_mel
    if (nfilts == 0):
        nfilts = np.ceil(4.6*np.log10(sr))
        # nfilts = np.ceil(nyq_mel) + 1
    h_fft = int(0.5*nfft)
    wts = np.zeros((nfilts, h_fft))
    step_mel = nyq_mel/(nfilts+1)
    bin_hertz = np.array([i*sr/nfft for i in range(0,h_fft)])
    limits = np.empty((2,h_fft))
    vFmid = np.zeros((nfilts,))
    for i in range(0,nfilts):
        f_mid = mel2hertz(min_mel + (i+1)*step_mel,Slaney)
        f_ini = mel2hertz(min_mel + i*step_mel,Slaney)
        f_fim = mel2hertz(min_mel + (i+2)*step_mel,Slaney)
        
        vFmid[i] = f_mid
        limits[0,:] = (bin_hertz - f_fim)/(f_mid-f_fim)
        limits[1,:] = (bin_hertz - f_ini)/(f_mid-f_ini)
        if (Slaney):
            kMult = 2/(f_fim - f_ini)
        else:
            kMult = 1
        wts[i,:] = kMult*np.maximum(np.zeros((h_fft,)), np.min(limits,axis=0))
    return wts, vFmid

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
    bM = np.array([])
    bF = np.array([])
    bI = np.array([])
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
            # cX = -0.5*b[1]/b[0]
            cX = f[i]
            if ((f[i-1] < cX) and (cX < f[i+1]) and (b[0] < 1) and (Rsq1 > 0.99)):
                # cY = b[0]*(cX**2) + b[1]*cX + b[2]
                cY = m[i]
                pM = np.append(pM,cY)
                pF = np.append(pF,cX)
                pI = np.append(pI,i)
                
    if (len(pI) < 1):
        idxM = np.argmax(m)
        mM = np.max(m)
        pM = np.append(pM,mM)
        pF = np.append(pF,f[idxM])
        pI = np.append(pI,idxM)
        idxM = np.argmin(m)
        mM = np.min(m)
        bM = np.append(bM,mM)
        bF = np.append(bF,f[idxM])
    else:
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
            
            idx = np.argmin(m[iIni:(iFim+1)])
            bY = np.min(m[iIni:(iFim+1)])
            bM = np.append(bM,bY)
            bF = np.append(bF,f[iIni+idx])
            bI = np.append(bI,iIni+idx)
     
    if (bF[0] != f[0]):
        bF = np.append(f[0],bF)
        bM = np.append(m[0],bM)
        bI = np.append(0,bI)
    if (bF[-1] != f[-1]):
        bF = np.append(bF,f[-1])
        bM = np.append(bM,m[-1])
        bI = np.append(bI,len(f)-1)
        
    if (pF[0] != f[0]):
        pF = np.append(f[0],pF)
        pM = np.append(pM[0],pM)
        pI = np.append(0,pI)

    if (pF[-1] != f[-1]):
        pF = np.append(pF,f[-1])
        pM = np.append(pM,pM[-1])
        pI = np.append(pI,len(f)-1)
        
    
    return pM, pF, bM, bF, pI, bI
# -----------------------------------------------------------------------------