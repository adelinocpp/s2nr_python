#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 12:10:11 2021

@author: adelino
"""
import numpy as np
import librosa
from scipy import signal, ndimage, stats
# ------------------------------------------------------------------------------
class S2NRparam:
  def __init__(self, fs, Ns, OFFSET = 10, NFFT = 1024,RTH = 0.6, alpha = 0.5, sigma = 0.1):
    self.eps = 1e-16;
    self.blksze1 = 5; 
    self.thresh = sigma; 
    self.blksze2 = 16;
    self.gradientsigma = 1;
    self.blocksigma = 5;	
    self.orientsmoothsigma = 5;
    self.windsze = 5; 
    self.minWaveLength = 1;
    self.maxWaveLength = 15;
    self.kx = 0.15;
    self.ky = 0.15;    
    self.rthresh = RTH;
    self.OFFSET = OFFSET;
    self.NFFT = NFFT;
    self.FS = fs;
    self.TotalTime = (Ns-1)/fs;

# ------------------------------------------------------------------------------
def S2NR(y_data, fs, frame_length, frame_shift, NFFT, RTH, sigma):
    nTimeWindow = int(np.ceil(fs*frame_length))
    nTimeStep = int(np.ceil(fs*frame_shift))
    nSamples = len(y_data)

    param = S2NRparam(fs,nSamples, NFFT=NFFT,RTH=RTH, sigma=sigma)

    y_data = y_data/np.max(np.abs(y_data))
    
    [AmpliNorm,AmpliNormLog, Taxis] = GenerateSpectrum( y_data, fs, nTimeWindow, nTimeStep, NFFT)
    NumFrames = AmpliNorm.shape[1]
    sizeAmpli = AmpliNorm.shape
		
    # Pre-processing : zero-padding 	
    zeroMatrix = np.zeros((param.OFFSET,sizeAmpli[1]))
    SpectrumInput = np.concatenate((zeroMatrix, AmpliNorm), axis=0)
    SpectrumInputLog = np.concatenate((zeroMatrix, AmpliNormLog), axis=0)

    H2NR_signal, S2NR_signal = s2nr_measurement(SpectrumInput, SpectrumInputLog, param)

    
    H2NR_norm  = 10 ** (H2NR_signal/10)
    S2NR_norm  = 10 ** (S2NR_signal/10)
    #S2NR_std = std(S2NR_norm)
    #S2NR_mean = mean(S2NR_norm)
    # LimSup = S2NR_mean + 3*S2NR_std
    # LimInf = S2NR_mean - 3*S2NR_std
    # S2NR_mean = 10 * log10 (mean(S2NR_norm( (S2NR_norm < LimSup) & (S2NR_norm > LimInf)) ))

    return H2NR_signal, S2NR_signal, 10*np.log10(stats.trim_mean(H2NR_norm,0.0015)), 10*np.log10(stats.trim_mean(S2NR_norm,0.0015))
# ------------------------------------------------------------------------------
def NormMinMax(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
# ------------------------------------------------------------------------------
def NormMeanStd(data):
    return (data - np.mean(data)) /np.std(data)
# ------------------------------------------------------------------------------    
def fgaussian(size, sigma):
    m,n = size
    h, k = m//2, n//2
    x, y = np.mgrid[-h:h+1, -k:k+1]
    return np.exp(-(x**2 + y**2)/(2*sigma**2))
# ------------------------------------------------------------------------------    
def GenerateSpectrum(audio, fs, n_win_length, n_hop_length, n_FFT):
    Ns = len(audio)
    linear_spect = librosa.stft(audio, win_length=n_win_length, hop_length=n_hop_length, n_fft=n_FFT, window='hamming')
    mag = np.abs(linear_spect)  # magnitude
    AmpliNorm = NormMinMax(mag)
    AmpliLog = 20*np.log10(mag+1e-9);
    AmpliNormLog = (NormMinMax(AmpliLog));    
    Taxis = np.linspace(0,(Ns-1)/fs,AmpliLog.shape[1])    
    return AmpliNorm, AmpliNormLog,Taxis
# ------------------------------------------------------------------------------
def ridgesegment(im, param):
    im = NormMeanStd(im)  # normalise to have zero mean, unit std dev
    stddevim = np.zeros(im.shape)
    nrow, ncol = im.shape
    for i in range(0,nrow,param.blksze1):
        rIni = i
        rFim = np.min([i + param.blksze1-1,nrow])
        for j in range(0,ncol,param.blksze1):
            cIni = j
            cFim = np.min([j + param.blksze1-1,ncol])
            stddevim[rIni:rFim,cIni:cFim] = np.std(im[rIni:rFim,cIni:cFim])
    mask = np.array(stddevim > param.thresh,dtype=int)
    (ml,mc) = mask.nonzero()
    # Renormalise image so that the *ridge regions* have zero mean, unit
    # standard deviation.
    im = im - np.mean(im[ml,mc])
    normim = im/np.std(im[ml,mc])
    return normim, mask
# ------------------------------------------------------------------------------
def ridgeorient(im, param):
    (rows,cols) = im.shape;
    # Calculate image gradients.
    sze = np.fix(6*param.gradientsigma)  
    if (int(np.mod(sze,2)) == 0): 
        sze = sze+1
        
    f = fgaussian((sze,sze), param.gradientsigma)     # Generate Gaussian filter.
    (fx,fy) = np.gradient(f)                      # Gradient of Gausian.
    
    Gx = signal.convolve2d(im, np.rot90(fx,2), mode='same')
    Gy = signal.convolve2d(im, np.rot90(fy,2), mode='same')
    # Gx = filter2(fx, im); # Gradient of the image in x
    # Gy = filter2(fy, im); # ... and y
    
    # Estimate the local ridge orientation at each point by finding the
    # principal axis of variation in the image gradients.
   
    Gxx = Gx**2       # Covariance data for the image gradients
    Gxy = Gx*Gy
    Gyy = Gy**2
    
    # Now smooth the covariance data to perform a weighted summation of the
    # data.
    sze = np.fix(6*param.blocksigma)
    if (int(np.mod(sze,2)) == 0): 
        sze = sze+1
        
    f = fgaussian((sze,sze), param.blocksigma);
    Gxx = signal.convolve2d(Gxx, np.rot90(f,2), mode='same')
    Gxy = 2*signal.convolve2d(Gxy, np.rot90(f,2), mode='same')
    Gyy = signal.convolve2d(Gyy, np.rot90(f,2), mode='same')
    # Gxx = filter2(f, Gxx); 
    # Gxy = 2*filter2(f, Gxy);
    # Gyy = filter2(f, Gyy);
    
    # Analytic solution of principal direction
    denom = np.sqrt(Gxy**2 + (Gxx - Gyy)**2) + param.eps
    sin2theta = Gxy/denom            # Sine and cosine of doubled angles
    cos2theta = (Gxx-Gyy)/denom
       
    sze = np.fix(6*param.orientsmoothsigma)
    if (int(np.mod(sze,2)) == 0): 
        sze = sze+1
    f = fgaussian((sze,sze), param.orientsmoothsigma);  
    cos2theta = signal.convolve2d(cos2theta, np.rot90(f,2), mode='same') # Smoothed sine and cosine of
    sin2theta = signal.convolve2d(sin2theta, np.rot90(f,2), mode='same') # doubled angles
    # cos2theta = filter2(f, cos2theta); # Smoothed sine and cosine of
    # sin2theta = filter2(f, sin2theta); # doubled angles
    orientim = 0.5*np.pi + np.arctan2(sin2theta,cos2theta)/2    
    # Calculate 'reliability' of orientation data.  Here we calculate the
    # area moment of inertia about the orientation axis found (this will
    # be the minimum inertia) and an axis  perpendicular (which will be
    # the maximum inertia).  The reliability measure is given by
    # 1.0-min_inertia/max_inertia.  The reasoning being that if the ratio
    # of the minimum to maximum inertia is close to one we have little
    # orientation information.     
    Imin = (Gyy+Gxx)/2 - (Gxx-Gyy)*cos2theta/2 - Gxy*sin2theta/2
    Imax = Gyy+Gxx - Imin
    reliability = 1 - Imin/(Imax+.001)
    # Finally mask reliability to exclude regions where the denominator
    # in the orientation calculation above was small.  Here I have set
    # the value to 0.001, adjust this if you feel the need
    reliability = reliability*np.array(denom>.001,dtype=int)
    
    return orientim, reliability
# ------------------------------------------------------------------------------
def freqest(im, orientim, param): 
    [rows,cols] = im.shape
    # Find mean orientation within the block. This is done by averaging the
    # sines and cosines of the doubled angles before reconstructing the
    # angle again.  This avoids wraparound problems at the origin.
    orientim = 2*orientim
    cosorient = np.mean(np.cos(orientim))
    sinorient = np.mean(np.sin(orientim))    
    orient = np.arctan2(sinorient,cosorient)/2;

    # Rotate the image block so that the ridges are vertical
    rotim = ndimage.rotate(im,orient/np.pi*180+90,reshape=False)
    # rotim = imrotate(im,orient/pi*180+90,'nearest', 'crop')
    
    # Now crop the image so that the rotated image does not contain any
    # invalid regions.  This prevents the projection down the columns
    # from being mucked up.
    # cropsze = np.fix(rows/sqrt(2)); 
    # offset = np.fix((rows-cropsze)/2);
    # rotim = rotim(offset:offset+cropsze, offset:offset+cropsze);
    # imagesc(rotim);
    # Sum down the columns to get a projection of the grey values down
    # the ridges.
    proj = rotim.sum(0)
    
    # Find peaks in projected grey values by performing a greyscale
    # dilation and then finding where the dilation equals the original
    # values. 
    proj.shape = (1,proj.shape[0])
    dilation = ndimage.rank_filter(proj, rank=param.windsze-1, footprint=np.ones((1,param.windsze)))
    # dilation = ordfilt2(proj, param.windsze, np.ones((1,param.windsze)));
    # maxpts = ((dilation == proj) and (proj > np.mean(proj)))
    maxpts = np.array([x & y for x, y in zip((dilation == proj), (proj > np.mean(proj)))],dtype=int).squeeze()
    maxind = maxpts.nonzero()
    # Determine the spatial frequency of the ridges by divinding the
    # distance between the 1st and last peaks by the (No of peaks-1). If no
    # peaks are detected, or the wavelength is outside the allowed bounds,
    # the frequency image is set to 0
    if (len(maxind[0]) < 2):
    	freqim = np.zeros(im.shape)
    else:
        NoOfPeaks = len(maxind[0])
        waveLength = (maxind[0][-1]-maxind[0][0])/(NoOfPeaks-1)    
        if ((waveLength > param.minWaveLength) and (waveLength < param.maxWaveLength)):
            freqim = 1/waveLength * np.ones(im.shape)
        else:
            freqim = np.zeros(im.shape)
        
    return freqim
# ------------------------------------------------------------------------------
def ridgefreq(im, mask, orient, param):
    # TODO: depurar ridgefreq
    (rows, cols) = im.shape
    blksze = param.blksze2
    freq = np.zeros(im.shape);    
    for r in range(0,rows-blksze,blksze):
        for c in range(0,cols-blksze,blksze):
            blkim = im[r:r+blksze, c:c+blksze]
            blkor = orient[r:r+blksze, c:c+blksze]
            freq[r:r+blksze,c:c+blksze] = freqest(blkim, blkor, param)

    # Mask out frequencies calculated for non ridge regions
    freq = freq*mask;
    
    # Find median freqency over all the valid regions of the image.
    # medianfreq = median(freq(find(freq>0)))
    (ml,mc) = (freq > 0.0).nonzero()
    medianfreq = np.median(freq[ml,mc])
    return freq, medianfreq
# ------------------------------------------------------------------------------
def ridgefilter(im, orient, freq, param):
    angleInc = 1;  # Fixed angle increment between filter orientations in
                   # degrees. This should divide evenly into 180 orignal 3
    im = np.array(im,dtype=float)
    rows, cols = im.shape;
    newim = np. zeros((rows,cols))
    
    (validr,validc) = (freq > 0.0).nonzero()  # find where there is valid frequency data.
    # ind = sub2ind([rows,cols], validr, validc);

    # Round the array of frequencies to the nearest 0.01 to reduce the
    # number of distinct frequencies we have to deal with.
    freq[(validr,validc)] = np.round(freq[(validr,validc)]*100)/100
    
    # Generate an array of the distinct frequencies present in the array
    # freq 
    unfreq = np.unique(freq[(validr,validc)])
    
    # Generate a table, given the frequency value multiplied by 100 to obtain
    # an integer index, returns the index within the unfreq array that it
    # corresponds to
    freqindex = np.zeros((100,1))
    for k in range(0,len(unfreq)):
        freqindex[int(np.round(unfreq[k]*100))] = k
    
    # Generate filters corresponding to these distinct frequencies and
    # orientations in 'angleInc' increments.
    # filter = np.zeros((len(unfreq),int(180/angleInc),None,None))
    filter = {}
    sze = np.zeros((len(unfreq),1))
    
    for k in range(0,len(unfreq)):
        sigmax = 1/unfreq[k]*param.kx
        sigmay = 1/unfreq[k]*param.ky
        
        sze[k] = round(3*max(sigmax,sigmay));
        x, y = np.mgrid[-sze[k]:sze[k],-sze[k]:sze[k]]
        reffilter = np.exp(-(x**2/sigmax**2 + y**2/sigmay**2)/2)*np.cos(2*np.pi*unfreq[k]*x)

        # Generate rotated versions of the filter.  Note orientation
        # image provides orientation *along* the ridges, hence +90
        # degrees, and imrotate requires angles +ve anticlockwise, hence
        # the minus sign.
        for o in range(0,int(180/angleInc)):
            filter[k,o] = ndimage.rotate(reffilter,-(o*angleInc+90),reshape=False)
    
    # Find indices of matrix points greater than maxsze from the image
    # boundary
    maxsze = sze[0]
    
    boolTemp = np.array((validr>maxsze) & (validr<(rows-maxsze)) & (validc>maxsze) & (validc<(cols-maxsze)),dtype=int)
    finalind = boolTemp.nonzero()
    
    # Convert orientation matrix values from radians to an index value
    # that corresponds to round(degrees/angleInc)
    maxorientindex = np.round(180/angleInc)-1
    orientindex = np.round(orient/np.pi*180/angleInc) -1
    (ml,mc) = (orientindex < 0).nonzero()
    orientindex[(ml,mc)] = orientindex[(ml,mc)]+maxorientindex;
    (ml,mc) = (orientindex > maxorientindex).nonzero()
    orientindex[(ml,mc)] = orientindex[(ml,mc)]-maxorientindex 
    # Finally do the filtering
    for k in range(0,len(finalind[0])):
        rs = int(validr[finalind[0][k]])
        cs = int(validc[finalind[0][k]])
        # find filter corresponding to freq(r,c)
        filterindex = int(freqindex[int(np.round(freq[rs,cs]*100))])
        
        ss = int(sze[filterindex])
        newim[rs,cs] = np.sum(im[rs-ss:rs+ss, cs-ss:cs+ss]*filter[filterindex,int(orientindex[rs,cs])])

    return newim
# ------------------------------------------------------------------------------
def s2nr_measurement(im, imlog,param):
    normim, mask = ridgesegment(imlog, param)
    # Determine ridge orientations 
    orientim, reliability = ridgeorient(normim, param)
    # Determine ridge frequency values across the image
    
    freq, medfreq = ridgefreq(normim, mask, orientim, param)

    # Actually I find the median frequency value used across the whole
    # fingerprint gives a more satisfactory result...
    
    freq = medfreq* mask
    newim = ridgefilter(normim, orientim, freq, param)
    
    # Binarise, ridge/valley threshold is 0    
    binim = np.array(newim > 0,dtype=int)

    # Display binary image for where the mask values are one and where
    # the orientation reliability is greater than rthresh
    
    MascaraR = np.array(reliability>param.rthresh,dtype=int)
    Mascara = binim*mask*MascaraR
    MascaraNew = Mascara
    
    ImM = MascaraNew*im
    ImMNot = (1-MascaraNew)*im
    ImMNot = signal.medfilt2d(ImMNot)

    SImH =   np.sum( ImM**2,axis=0)
    SImM =   np.sum( im**2,axis=0)
    SImMNot = np.sum( ImMNot**2,axis=0)

    Temp = (SImM+param.eps)/(SImMNot+param.eps)
    Temp2 = (SImH+param.eps)/(SImMNot+param.eps)
    Temp_filt = signal.medfilt(Temp, kernel_size=15)
    Temp2_filt = signal.medfilt(Temp2, kernel_size=15)
    HNR = 10 * np.log10(Temp2_filt)
    SNR = 10 * np.log10(Temp_filt)
    # HNR = 0
    # SNR = 0
    return HNR, SNR
# ------------------------------------------------------------------------------
