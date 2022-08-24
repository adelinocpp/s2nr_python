#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 17:19:54 2022

@author: adelino
"""
import os
from glob import glob
import librosa
from S2NR2 import S2NR2
import pickle
import numpy as np
import pandas as pd
from utils.snr import snr_vad_total
from scipy import stats
import multiprocessing
import sys

CALIBRATE_FOLDER = './Vozes_Calibracao/'
EXPERIMENT_FOLDER = './Calculos_S2NR/'
if (applySpectrunSlice):
    CSV_EXPLORATORY_DATA_FILE = 'Prev_Slice_S2NR_Calibration_data.csv'
else:
    CSV_EXPLORATORY_DATA_FILE = 'Prev_Vanila_S2NR_Calibration_data.csv'

feature_list = glob(os.path.join(CALIBRATE_FOLDER, '*.p'), recursive=True)
feature_list.sort()

win_length = 0.025
hop_length = 0.01
calibrate_S2NR = True

dataNFFT = [512, 1024, 2048, 4096]
dataRth = [0.03, 0.1, 0.3, 0.6, 0.95]
dataSigma = [0.007, 0.01, 0.07, 0.1, 0.4]
dataWinLength = [.75, 1, 1.25, 1.75]

applySpectrunSlice = False

mtxComb = np.empty(shape=(0,4))
for iWinLen in range(0,len(dataWinLength)):
    for iNFFT in range(0,len(dataNFFT)):
        for iSigma in range(0,len(dataSigma)):
            for iRth in range(0,len(dataRth)):
                selComb = np.array([iNFFT,iSigma,iRth,iWinLen])
                selComb.shape = (1,4)
                mtxComb = np.append(mtxComb, selComb, axis=0)

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

nRounds = len(mtxComb)
# -----------------------------------------------------------------------------
class returnObject:
    error = False
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
# -----------------------------------------------------------------------------
def run(filename,idx,iRound):
    with open(filename, 'rb') as f:
        suport_feature = pickle.load(f)

    wavename = os.path.join(CALIBRATE_FOLDER, suport_feature['filename'])
    audio, sr = librosa.load(wavename, sr=None, mono=True)
    
    # dataNFFT = [512, 1024, 2048, 4096]
    # dataRth = [0.03, 0.1, 0.3, 0.6, 0.95]
    # dataSigma = [0.007, 0.01, 0.07, 0.1, 0.4]
    # dataWinLength = [.75, 1, 1.25, 1.5]
    
    # iA = int(np.random.randint(0,len(dataNFFT)))
    # iB = int(np.random.randint(0,len(dataRth)))
    # iC = int(np.random.randint(0,len(dataSigma)))
    # iD = int(np.random.randint(0,len(dataWinLength)))
    # sNFFT = vecNFFT[iA]
    # sRTH = vecRTH[iB]
    # sSIGMA = vecSIGMA[iC]
    # sLength = vecWinLength[iD]
    
    # iE = int(np.random.randint(0,len(mtxComb)))
    sLength = dataWinLength[int(mtxComb[iRound,0])]
    sNFFT = dataNFFT[int(mtxComb[iRound,1])]
    sSIGMA = dataSigma[int(mtxComb[iRound,2])]
    sRTH = dataRth[int(mtxComb[iRound,3])]
    
    retVAls = returnObject()
    retVAls.error = False
    try:
        if (applySpectrunSlice):
            rH2NR, rS2NR, mH2NR, mS2NR = S2NR2(audio, sr, sLength*win_length, hop_length, NFFT=sNFFT, RTH=sRTH, sigma=sSIGMA, mmNorm=True)
        else:
            rH2NR, rS2NR, mH2NR, mS2NR = S2NR(audio, sr, sLength*win_length, hop_length, NFFT=sNFFT, RTH=sRTH, sigma=sSIGMA)
    except ValueError:
        print("Erro no calculo do S2NR")
        retVAls.error = True
        return retVAls
    
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
    
    retVAls = returnObject()
    retVAls.lstSNRtarget = SNR_target
    retVAls.lstSNRvad = snr_vad
    retVAls.lstNFFT = sNFFT
    retVAls.lstRTH = sRTH
    retVAls.lstSIGMA = sSIGMA
    retVAls.lstWINLength = sLength
    retVAls.lstS2NRmean = stats.trim_mean(rS2NR,0.0015)
    retVAls.lstH2NRmean = stats.trim_mean(rH2NR,0.0015)
    retVAls.lstdiffS2NRmean = stats.trim_mean(diffS2NR,0.0015)
    retVAls.lstdiffH2NRmean = stats.trim_mean(diffH2NR,0.0015)
    retVAls.lstdiffS2NRstd = stats.mstats.trimmed_std(diffS2NR, limits=(0.0015, 0.0015))
    retVAls.lstdiffH2NRstd = stats.mstats.trimmed_std(diffH2NR, limits=(0.0015, 0.0015))
    print('\tFinalizado arquivo {:4d} de {:4d}, rodada {:2d} de {:2d}'.format(idx, len(feature_list)-1, iRound,nRounds-1))
    return retVAls
# -----------------------------------------------------------------------------

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
    
    pool = multiprocessing.Pool(os.cpu_count())
    jobs = []
    for iRound in range(0,nRounds):
        for idx, filename in enumerate(feature_list):
            jobs.append(pool.apply_async(run, (filename,idx,iRound )))
            
    futureReturn = [job.get() for job in jobs]
    for idx, futureRetObj in enumerate(futureReturn):
        if (futureRetObj.error):
            print("Erro encontrado no resultado indice {:}.".format(idx))
            continue
        print("Indice {:} sem registro de erro.".format(idx))
        lstSNRtarget.append(futureRetObj.lstSNRtarget)
        lstSNRvad.append(futureRetObj.lstSNRvad)
        lstNFFT.append(futureRetObj.lstNFFT)
        lstRTH.append(futureRetObj.lstRTH)
        lstSIGMA.append(futureRetObj.lstSIGMA)
        lstWINLength.append(futureRetObj.lstWINLength)
        lstS2NRmean.append(futureRetObj.lstS2NRmean)
        lstH2NRmean.append(futureRetObj.lstH2NRmean)
        lstdiffS2NRmean.append(futureRetObj.lstdiffS2NRmean)
        lstdiffH2NRmean.append(futureRetObj.lstdiffH2NRmean)
        lstdiffS2NRstd.append(futureRetObj.lstdiffS2NRstd)
        lstdiffH2NRstd.append(futureRetObj.lstdiffH2NRstd)
        
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