#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 13:53:12 2022

@author: adelino
"""
from glob import glob
import os
import pickle
import librosa
import numpy as np
from utils.snr import frame_rms, rms, snr_vad_total
import pathlib
from scipy.io.wavfile import write
import random

NOISE_FOLDER = './RUIDOS/'
VOICE_FOLDER = './Vozes_Sintetizadas_Sem_ruido/'
CALIBRATE_FOLDER = './Vozes_Calibracao/'
NOISE_FEATURES_FILE = 'noise_raw_feature.p'
VOICE_FEATURES_FILE = 'voice_raw_feature.p'
MAKE_NEW_COMPUTE = True

voice_list = glob(os.path.join(VOICE_FOLDER, '*.wav'), recursive=True)
noise_list = glob(os.path.join(NOISE_FOLDER, '*.wav'), recursive=True)
voice_list.sort()
noise_list.sort()
voiceff_exist = os.path.exists(os.path.join(VOICE_FOLDER, VOICE_FEATURES_FILE))
noiseff_exist = os.path.exists(os.path.join(NOISE_FOLDER, NOISE_FEATURES_FILE))

target_Noise = [25, 22, 19, 16, 13, 10]

if ((not voiceff_exist) or (not noiseff_exist)):
    print('Arquivo(s) de caracteristicas não encontrado, execute a rotina P00 primeiro')
    if (voiceff_exist):
        print('\t{:}'.format(os.path.join(VOICE_FOLDER, VOICE_FEATURES_FILE)))
    if (noiseff_exist):
        print('\t{:}'.format(os.path.join(NOISE_FOLDER, NOISE_FEATURES_FILE)))
    
if (voiceff_exist and noiseff_exist):
    with open(os.path.join(NOISE_FOLDER, NOISE_FEATURES_FILE), 'rb') as f:
        noise_feature_list = pickle.load(f)
    with open(os.path.join(VOICE_FOLDER, VOICE_FEATURES_FILE), 'rb') as f:
        voice_feature_list = pickle.load(f)

    if (not os.path.exists(CALIBRATE_FOLDER)):
            os.mkdir(CALIBRATE_FOLDER)

    for idx_v, voicefile in enumerate(voice_list):
        audio_v, sr_v = librosa.load(voicefile, sr=None, mono=True)
        npt_v = len(audio_v)
        base_SNR = voice_feature_list[idx_v]['snr_vad'][0]
        base_RMS = voice_feature_list[idx_v]['rms']
        base_vad = voice_feature_list[idx_v]['vad']
        base_frame_rms = voice_feature_list[idx_v]['frame_rms']
        mean_RMS = base_frame_rms.mean()
        for idx_n, noisefile in enumerate(noise_list):
            audio_n, sr = librosa.load(noisefile, sr=None, mono=True)
            npt_n = len(audio_n)
            prod = int(np.ceil(npt_v/npt_n))
            if (prod > 1):
                audio_nr = np.array([])
                for i in range(1,prod+1):
                    audio_nr = np.append(audio_nr,audio_n)
                audio_n = audio_nr[:npt_v]
            else:
                audio_n = audio_n[:npt_v]
            
            noise_rms_ref = rms(audio_n)
            frame_noise_rms = frame_rms(audio_n,sr)
            for SNRtg in target_Noise:
                useSNR = random.gauss(SNRtg, 1)
                if (useSNR > base_SNR):
                    continue
                
                
                SNR_target = (10**(useSNR/20))
                SNR_Atual = noise_rms_ref/base_RMS   
                Kmult = SNR_target*noise_rms_ref/base_frame_rms.mean()
                audio_c = Kmult*audio_v + audio_n
                
                
                print('Check ruido: SNR_base: {:3.1f} SNR_atual: {:3.1f}, SNR_tg: {:}, K: {:4.2f}'\
                      .format(base_SNR, 10*np.log10(SNR_Atual),useSNR,Kmult))
                
                audio_c = np.divide(audio_c,np.max(np.abs(audio_c)))
                
                SNR_vec = 20*np.log10(np.divide(np.multiply(base_frame_rms,Kmult),frame_noise_rms))
                sigSNRMN = snr_vad_total(audio_c,base_vad,sr_v)
                
                suport_feature = {}
                basename = pathlib.Path(voicefile).stem
                wavename = '{:}_N_{:02}_SNR_{:02}.wav'.format(basename,idx_n,SNRtg)
                featurename = '{:}_N_{:02}_SNR_{:02}.p'.format(basename,idx_n,SNRtg)
                suport_feature['filename'] = wavename
                suport_feature['vad'] = base_vad
                
                suport_feature['frame_snr'] = SNR_vec 
                suport_feature['frame_rms_audio'] = np.multiply(base_frame_rms,Kmult) 
                suport_feature['frame_rms_noise'] = frame_noise_rms 
                suport_feature['frame_snr_target'] = useSNR
                suport_feature['frame_snr_nominal'] = SNRtg
                suport_feature['snr_vad'] = sigSNRMN
                suport_feature['snr_quant'] = 2*16+4.77-20*np.log10(np.max(np.abs(audio_c))/np.std(audio_c))
                with open(os.path.join(CALIBRATE_FOLDER, featurename), 'wb') as f:
                    pickle.dump(suport_feature,f)
                write(os.path.join(CALIBRATE_FOLDER, wavename), sr_v, audio_c)
    
    
