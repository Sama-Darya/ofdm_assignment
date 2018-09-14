#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 10:09:39 2018

@author: Bernd & Sama
"""

import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt

### DATA RECEPTION

# 1) We start by loading the audio data 
[f, receivedDataRaw]= wavfile.read('ofdmsignl2.wav', mmap=False)

# 2) Data loss: during real-life transmission often the start of the data is lost.
# In order to mimic this effect, a part of data is manually removed form the start.

nLostData=20000
receivedData=receivedDataRaw[nLostData:] #removing the first 20k samples

symbolLength = 409604 # duration of the data (i.e T)
CyclicPrefix=1100 # duration of the cyclic prefix (i.e. C)

# 3) Your Task: is to find the correct start point using guard intervals and polit tones
startSample = 390704 + CyclicPrefix

### Data reconstruction
# Now that the start-point is found the process of data reconstruction can begin
# we take one duration of the symbol starting from the start-point found above
symbolPrime=receivedData[startSample:symbolLength+startSample]

# 4) FFT: Performing a Fast Fourier Transformation on the received data results in the 
# frequency domain with the pilot tones.

paddedPilotTonedDataPrime=np.fft.fft(symbolPrime)

# 5) Strip the DC and Remove the padding
# removing the padding and the DC (zero) frequency that was previously added
zero_padding=2
pilotTonedDataPrime=paddedPilotTonedDataPrime[zero_padding+1: -zero_padding]

# Pilot Tones:
# as a second measure for start-point detection, we should check that the pilot tones 
# are all real-valued with zero phase shift. This confirms that the start-point detected
# by guard-interval method was correct, otherwise fine tunning of pilot tones is needed.

# Figure(11) shows the received spectrum with pilot tones
bins3=np.arange(-3.5,2,0.2)                                            
plt.figure("11")
plt.subplot(2,1,1)
plt.plot(np.real(pilotTonedDataPrime[204775:204825]))
plt.title("pilotTonedData: After FFT")
plt.subplot(2,1,2)
plt.hist(np.real(pilotTonedDataPrime), bins3)
plt.title("Histogram of frequency spectrum with pilot tones")

# 6) Removing the Pilot Tones:
# the original spectrum is found by dismissing every other data (i.e. the pilot tones)

fullComplexDataWithMirrorPrimeReal=np.real(pilotTonedDataPrime[0::2])
fullComplexDataWithMirrorPrimeImag=np.imag(pilotTonedDataPrime[0::2])
fullComplexDataWithMirrorPrime=fullComplexDataWithMirrorPrimeReal + fullComplexDataWithMirrorPrimeImag * 1j

# Figure(12) shows this spectrum
plt.figure("12")
plt.subplot(2,1,1)
plt.plot(np.real(fullComplexDataWithMirrorPrime[102375:102425]))
plt.title("fullComplexDataWithMirror: After FFT")
plt.subplot(2,1,2)
plt.hist(np.real(fullComplexDataWithMirrorPrime), bins3)
plt.title("Histogram of frequency spectrum")


# 7) QAM decoding: 
# every received data contain a real and an imaginary part. These are unpacked into two 
# real-valued data again.
# For example: if the first two data are:    -1+1j and 1+0j 
# these will recover the 4 original data as:  -1 , 1 , 1 , 0

datalLength=len(fullComplexDataWithMirrorPrime)
datalLength2=int(len(fullComplexDataWithMirrorPrime)/2)
dataTimeDomainReal=np.real(fullComplexDataWithMirrorPrime[0:datalLength2])
dataTimeDomainImag=np.imag(fullComplexDataWithMirrorPrime[0:datalLength2])

img1DbitsRandomPrime=np.ones(datalLength)
img1DbitsRandomPrime[0::2]=(dataTimeDomainReal + 1 ) / 2
img1DbitsRandomPrime[1::2]=(dataTimeDomainImag + 1 ) / 2

# We have now recovered the original scrambled data
# Figure(13) shows this data
bins2=np.arange(-1.5, 1.5, 0.2)
plt.figure("13")
plt.subplot(2,1,1)
plt.plot(np.real(img1DbitsRandomPrime[200:250]))
plt.title("img1DbitsRandom: After FFT")
plt.subplot(2,1,2)
plt.hist(np.real(img1DbitsRandomPrime) , bins2)
plt.title("Histogram of randomised data")

# 8) Unscramble the data: this is done by dividing the data by the same random array that was
# used to randomise the data originally, and hence the original binary data is recovered
np.random.seed(0)
scrambler = np.random.randint(0,2,img1DbitsRandomPrime.size) * 2.0 - 1.0
img1DbitsPrime = img1DbitsRandomPrime / scrambler

# The next few lines ensure that we only have 0s and 1s left in the data
for i in range (len(img1DbitsPrime)): #look at histogram and change the threshholds
    if (img1DbitsPrime[i] < 0.2):
        img1DbitsPrime[i] = 0
    else:
        img1DbitsPrime[i] = 1
        
# We then turn the data into integers
img1DbitsPrime=np.int_(img1DbitsPrime)

# We are now left with data that is a stream of bits
# Figure(14) shows this data

bins1=np.arange(-0.5,1.5,0.1)
plt.figure("14")
plt.subplot(2,1,1)
plt.plot(img1DbitsPrime[0:60])
plt.title("image1Dbits: After FFT")
plt.subplot(2,1,2)
plt.hist(img1DbitsPrime, bins1)
plt.title("Histogram of binary data")

# 9) packing bits into 1D array
# we pack every 8 bits of data into 1 entry of a 1 dimensional array. These entries now
# correspond to grey-scale value of pixels of the original image
img1DPrime=np.packbits(img1DbitsPrime)

# Figure(15) shows this data
plt.figure("15")
plt.subplot(2,1,1)
plt.plot(img1DPrime)
plt.title("image1D: After FFT")
plt.subplot(2,1,2)
plt.hist(img1DPrime)
plt.title("Histogram of pixles' greyScale")

# 10) Reshaping the array: in this final part the 1D array is reshaped into its original
# 100 by 256 array to form the original image

imgPrime=np.reshape(img1DPrime,(100,256))

# Figure(16) must show the original image correctly
plt.figure("16")
plt.subplot(2,1,1)
plt.imshow(imgPrime, cmap ='gray')
plt.title("Image: After reception (After FFT)")
plt.subplot(2,1,2)
plt.hist(imgPrime)
plt.title("Histogram of pixles' greyScale")

