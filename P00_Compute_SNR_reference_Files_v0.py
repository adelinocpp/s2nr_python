#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 11:24:38 2022

@author: adelino
"""
from glob import glob
import os
import pickle
import librosa
import numpy as np
from utils.snr import frame_rms, rms, snr_vad_total
from utils import vad

NOISE_FOLDER = './RUIDOS/'
VOICE_FOLDER = './Vozes_Sintetizadas_Sem_ruido/'
NOISE_FEATURES_FILE = 'noise_raw_feature.p'
VOICE_FEATURES_FILE = 'voice_raw_feature.p'
MAKE_NEW_COMPUTE = True

voice_list = glob(os.path.join(VOICE_FOLDER, '*.wav'), recursive=True)
noise_list = glob(os.path.join(NOISE_FOLDER, '*.wav'), recursive=True)
voice_list.sort()
noise_list.sort()
voiceff_exist = os.path.exists(os.path.join(VOICE_FOLDER, VOICE_FEATURES_FILE))
noiseff_exist = os.path.exists(os.path.join(NOISE_FOLDER, NOISE_FEATURES_FILE))

win_length = 0.025
hop_length = 0.01

if ((not MAKE_NEW_COMPUTE) or voiceff_exist):
    voice_feature_list = []
    for idx, filename in enumerate(voice_list):
        audio, sr = librosa.load(filename, sr=None, mono=True)
        audio += np.random.randn(len(audio),)/(2**16-1)
        n_win_length = np.ceil(sr*win_length)
        n_FFT = int(2 ** np.ceil(np.log2(n_win_length)))
        mag = frame_rms(audio, sr)
        rms_audio = rms(audio)
        vad_sohn = vad.VAD(audio, sr, nFFT=n_FFT, win_length=win_length, \
                           hop_length=hop_length, theshold=0.7)
        sigSNRMN = snr_vad_total(audio,vad_sohn,sr)
        suport_feature = {}
        suport_feature['filename'] = filename
        suport_feature['vad'] = vad_sohn
        suport_feature['frame_rms'] = mag
        suport_feature['rms'] = rms_audio
        suport_feature['snr_vad'] = sigSNRMN
        suport_feature['snr_quant'] = 2*16+4.77-20*np.log10(np.max(np.abs(audio))/np.std(audio))
        voice_feature_list.append(suport_feature)
        print('Finalizados arquivo de vozes {:} de {:}'.format(idx,len(voice_list)-1))
    with open(os.path.join(VOICE_FOLDER, VOICE_FEATURES_FILE), 'wb') as f:
        pickle.dump(voice_feature_list,f)
    print('Finalizados arquivos de vozes')
    
if ((not MAKE_NEW_COMPUTE) or noiseff_exist):
    noise_feature_list = []
    for idx, filename in enumerate(noise_list):
        audio, sr = librosa.load(filename, sr=None, mono=True)
        n_win_length = np.ceil(sr*win_length)
        n_FFT = int(2 ** np.ceil(np.log2(n_win_length)))
        mag = frame_rms(audio, sr)
        rms_audio = rms(audio)
        suport_feature = {}
        suport_feature['filename'] = filename
        suport_feature['frame_rms'] = mag
        suport_feature['rms'] = rms_audio
        noise_feature_list.append(suport_feature)
        print('Finalizados arquivo de ruido {:} de {:}'.format(idx,len(noise_list)-1))
    with open(os.path.join(NOISE_FOLDER, NOISE_FEATURES_FILE), 'wb') as f:
        pickle.dump(noise_feature_list,f)
    print('Finalizados arquivos de ruido')
    