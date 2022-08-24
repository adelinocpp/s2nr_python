# -*- coding: utf-8 -*-
import numpy as np
# ------------------------------------------------------------------------------
def basicSNR(S,N):
    # Compute SNR with RMS values of signal S and noise N
    return (10*np.log10(S)+120)/(10*np.log10(N)+120)
# ------------------------------------------------------------------------------
def signalMixedSNR(S,N):
    # Compute correct SNR with RMS values of (signal and noise) S and (silence only noise) N
    #K = (S/N)**2
    #S0 = (K-1)*(N**2)
    #return 10*np.log10(S0/N**2), S0
    K = (10*np.log10(S)+120)/(10*np.log10(N)+120)
    Mn = 10*np.log10(S)+120
    N0 = 10**(Mn/10)
    SNR = 10*np.log10(N0**(K-1) -1)
    SNR_d = 10**(SNR/10)
    S0 = SNR_d*N
    return SNR, S0
# ------------------------------------------------------------------------------
def rms(x):
    return np.sqrt(np.mean(np.square(x)))
# ------------------------------------------------------------------------------
def frame_rms(audio,sr,hop_length=0.01, win_length=0.025):
    audioLength = len(audio)
    nWinLength = int(np.ceil(sr*win_length))
    nWinStep = int(np.ceil(sr*hop_length))
    # hWinStep = int(np.floor(0.5*sr*hop_length))
    # k = 0;
    mag = np.array([])
    for i in range(0,(audioLength-nWinLength),nWinStep):
        mag = np.append(mag,rms(audio[i:i+nWinLength]))
        # k += 1
    return mag
# ------------------------------------------------------------------------------
def snr_vad_total(audio,vad,sr, win_length=0.025, hop_length=0.01):
    n_win_length = int(np.ceil(sr*win_length))
    n_hop_length = int(np.ceil(sr*hop_length))
    n_frames = len(vad)
    n_points = len(audio)
    sig = np.array([])
    noi = np.array([])
    i_frame = 0
    for i in range(0,n_points-n_win_length,n_hop_length):
        if (vad[i_frame] == 1.0):
            sig = np.append(sig,audio[i:i+n_win_length])
        if (vad[i_frame] == 0.0):
            noi = np.append(noi,audio[i:i+n_win_length])
        if (i_frame == n_frames):
            break
        i_frame += 1
    return signalMixedSNR(rms(sig),rms(noi))
# ------------------------------------------------------------------------------
def snr_mean_noise(rms,vad):
    nFrames = np.min([len(rms),len(vad)])
    snr = np.zeros((nFrames,1))
    lastNoise = 0
    noi = np.array([])
    for i in range(0,nFrames):
        if ((i > 0) and (vad[i] == 1.0) and (vad[i-1] == 0.0)):
            lastNoise = np.mean(noi)
        if ((i > 0) and (vad[i] == 0.0) and (vad[i-1] == 1.0)):
            noi = np.array([])
        if (vad[i] == 1.0):
            snr[i] = (rms[i]**2)/(lastNoise**2)
        if (vad[i] == 0.0):
            noi = np.append(noi,rms[i])
    return snr
# ------------------------------------------------------------------------------

