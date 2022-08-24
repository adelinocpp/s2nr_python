#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 17:19:54 2022

@author: adelino
"""
import os
from glob import glob
import librosa
from S2NR import S2NR
import pickle
import numpy as np
import pandas as pd
from utils.snr import snr_vad_total
from scipy import stats


CALIBRATE_FOLDER = './Vozes_Calibracao/'
EXPERIMENT_FOLDER = './Calculos_S2NR/'
CSV_EXPLORATORY_DATA_FILE = 'Prev_S2NR_Calibration_data.csv'
feature_list = glob(os.path.join(CALIBRATE_FOLDER, '*.p'), recursive=True)
feature_list.sort()

win_length = 0.025
hop_length = 0.01
calibrate_S2NR = True

# SAMSAO & VIEIRA
# NFFT = 1024
# RTH=0.6
# sigma=0.1

# NFFT
# 512, 1024, 2048
# RTH
# 0.03 0.1 0.3 0.6 0.95
# sigma
# 0.008 0.01 0.08 0.1
vecNFFT = [512, 1024, 2048, 4096]
vecRTH = [0.03, 0.1, 0.3, 0.6, 0.95]
vecSIGMA = [0.007, 0.01, 0.07, 0.1, 0.4]
vecWinLength = [.75, 1, 1.25, 1.5]
nRounds = 23

if (calibrate_S2NR):
    print("Calibrando S2NR")
    if (not os.path.exists(EXPERIMENT_FOLDER)):
            os.mkdir(EXPERIMENT_FOLDER)
            
    lstSNRtarget = []
    lstSNRvad = []
    lstNFFT = []
    lstRTH = []
    lstSIGMA = []
    lstWINLength = []
    lstS2NRmean = []
    lstH2NRmean = []
    lstdiffS2NRmean = []
    lstdiffH2NRmean = []
    lstdiffS2NRstd = []
    lstdiffH2NRstd = []
    
    for iRound in range(0,nRounds):
        for idx, filename in enumerate(feature_list):
            with open(filename, 'rb') as f:
                suport_feature = pickle.load(f)
            wavename = os.path.join(CALIBRATE_FOLDER, suport_feature['filename'])
            audio, sr = librosa.load(wavename, sr=None, mono=True)
            
            sLength = vecWinLength[np.random.randint(0,len(vecWinLength))]
            sNFFT = vecNFFT[np.random.randint(0,len(vecNFFT))]
            sRTH = vecRTH[np.random.randint(0,len(vecRTH))]
            sSIGMA = vecSIGMA[np.random.randint(0,len(vecSIGMA))]
            rH2NR, rS2NR, mH2NR, mS2NR = S2NR(audio, sr, sLength*win_length, hop_length, NFFT=sNFFT, RTH=sRTH, sigma=sSIGMA)
            
            vad_audio = suport_feature['vad']
            SNR_target = suport_feature['frame_snr_target']
            snr_vad, _ = snr_vad_total(audio,vad_audio,sr)
            
            frame_SNR = suport_feature['frame_snr']
            # nptRef = len(suport_feature['frame_snr'])
            # nptS2NR = len(rH2NR)
            numFrames = np.min([len(suport_feature['frame_snr']),len(rH2NR), len(vad_audio)])
            
            vad_audio = vad_audio[:numFrames]
            diffS2NR = np.array(rS2NR[:numFrames] - frame_SNR[:numFrames])
            diffH2NR = np.array(rH2NR[:numFrames] - frame_SNR[:numFrames])
            
            idx_vad = vad_audio.nonzero()[0]
            idx_pbm = (idx_vad >= numFrames).nonzero()[0]
            if (len(idx_pbm) > 0):
                idx_vad = np.delete(idx_vad, idx_pbm)
                
            rH2NR = rH2NR[idx_vad]
            rS2NR = rS2NR[idx_vad]
            diffH2NR = diffH2NR[idx_vad]
            diffS2NR = diffS2NR[idx_vad]
            lstSNRtarget.append(SNR_target)
            lstSNRvad.append(snr_vad)
            lstNFFT.append(sNFFT)
            lstRTH.append(sRTH)
            lstSIGMA.append(sSIGMA)
            lstWINLength.append(sLength)
            lstS2NRmean.append(stats.trim_mean(rS2NR,0.0015))
            lstH2NRmean.append(stats.trim_mean(rH2NR,0.0015))
            lstdiffS2NRmean.append(stats.trim_mean(diffS2NR,0.0015))
            lstdiffH2NRmean.append(stats.trim_mean(diffH2NR,0.0015))
            lstdiffS2NRstd.append(stats.mstats.trimmed_std(diffS2NR, limits=(0.0015, 0.0015)))
            lstdiffH2NRstd.append(stats.mstats.trimmed_std(diffH2NR, limits=(0.0015, 0.0015)))
            print('\tFinalizado arquivo {:4d} de {:4d}, rodada {:2d} de {:2d}'.format(idx, len(feature_list)-1, iRound,nRounds-1))
        
    dict_data = {'SNR_target': lstSNRtarget,
                 'SNR_vad': lstSNRvad,
                 'NFFT': lstNFFT,
                 'Rth': lstRTH,
                 'sigma': lstSIGMA,
                 'winLength': lstWINLength,
                 'S2NR_m': lstS2NRmean,
                 'H2NR_m': lstH2NRmean,
                 'dS2NR_m': lstdiffS2NRmean,
                 'dH2NR_m': lstdiffH2NRmean,
                 'dS2NR_v': lstdiffS2NRstd,
                 'dH2NR_v': lstdiffH2NRstd}

    dataS2NR = pd.DataFrame(dict_data)
    dataS2NR.to_csv(os.path.join(EXPERIMENT_FOLDER,CSV_EXPLORATORY_DATA_FILE))