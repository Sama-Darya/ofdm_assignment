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

symbolLength = 4100 # duration of the symbol (i.e T)
cyclicPrefix = 1100 # duration of the cyclic prefix (i.e. C)
onePeriod = symbolLength + cyclicPrefix

def decodeSymbol(data):
    # This function gets a symbol with its prefix in and returns the decoded symbol (i.e. line of iamge)
    symbolPrime = data[cyclicPrefix::] # Removing the cyclic prefix
    # 5.1) FFT: Performing a Fast Fourier Transformation on the received data results in the 
    # frequency domain with the pilot tones.
    paddedPilotTonedDataPrime=np.fft.fft(symbolPrime)
    
    # 5.2) Strip the DC and Remove the padding
    # removing the padding and the DC (zero) frequency that was previously added
    zero_padding=2
    pilotTonedDataPrime=paddedPilotTonedDataPrime[zero_padding+1: -zero_padding]
    
    # Pilot Tones:
    # as a second measure for start-point detection, we should check that the pilot tones 
    # are all real-valued with zero phase shift. This confirms that the start-point detected
    # by guard-interval method was correct, otherwise fine tunning of pilot tones is needed.
    
    # 5.3) Removing the Pilot Tones:
    # the original spectrum is found by dismissing every other data (i.e. the pilot tones)
    
    fullComplexDataWithMirrorPrime=pilotTonedDataPrime[0::2]
        
    # 5.4) QAM decoding: 
    # every received data contain a real and an imaginary part. These are unpacked into two 
    # real-valued data again.
    # For example: if the first two data are:    -1+1j and 1+0j 
    # these will recover the 4 original data as:  -1 , 1 , 1 , 0
    
    datalLength=len(fullComplexDataWithMirrorPrime)
    datalLength2=int(len(fullComplexDataWithMirrorPrime)/2)
    dataTimeDomainReal=np.real(fullComplexDataWithMirrorPrime[0:datalLength2])
    dataTimeDomainImag=np.imag(fullComplexDataWithMirrorPrime[0:datalLength2])
    
    img1DbitsRandomPrime=np.ones(datalLength)
    img1DbitsRandomPrime[0::2]=dataTimeDomainReal
    img1DbitsRandomPrime[1::2]=dataTimeDomainImag
    # We have now recovered the original scrambled data
    
    # 5.5) Unscramble the data: this is done by dividing the data by the same random array that
    # was used to randomise the data originally, and hence the original binary data is recovered
    np.random.seed(0)
    scrambler = np.random.randint(0,2,img1DbitsRandomPrime.size) * 2.0 - 1.0
    img1DbitsPrime = img1DbitsRandomPrime * scrambler
    
    # The next few lines ensure that we only have 0s and 1s left in the data
    for i in range (len(img1DbitsPrime)): #look at histogram and change the threshholds
        if (img1DbitsPrime[i] < 0):
            img1DbitsPrime[i] = 0
        else:
            img1DbitsPrime[i] = 1
            
    # We then turn the data into integers
    img1DbitsPrime=np.int_(img1DbitsPrime)
    # We are now left with data that is a stream of bits
    
    # 5.6) packing bits into 1D array
    # we pack every 8 bits of data into 1 entry of a 1 dimensional array. These entries now
    # correspond to grey-scale value of pixels of the original image
    img1DPrime=np.packbits(img1DbitsPrime)
    return img1DPrime
    
### DATA RECEPTION    

# 1) We start by loading the audio data 
[f, receivedDataRaw]= wavfile.read('ofdmSignal.wav', mmap=False)

receivedDataRaw=receivedDataRaw/30000

# 2) Data loss: during real-life transmission often the start of the data is lost.
# In order to mimic this effect, a part of data is manually removed form the start.

nLostData=1300
receivedData=receivedDataRaw[nLostData::] #removing the first 20k samples

# 3) Finding the start point:
# For theorethical demonstration, the start-point is given here, which is found using:
#        start-point = one-period - number-of-lost-data = 5200 - 1300 = 3900 
startSample = 3900 

# However, in practice the number-of-lost-data is not known.
# YOUR TASK: is to find the correct start point using "guard intervals" and "polit tones"

# 4) Data reconstruction
# Now that the start-point is found the process of data reconstruction can begin
# We first cut the data from where we think the first cyclic prefix starts

numberOfSymbols=95 #This is the numbers of symbols we think we have detected, given the lost data.
receivedData = receivedData[startSample: startSample + numberOfSymbols * onePeriod] 
# the "forced" indexing is to ensure we don't pick up white nose from the start or end of data

# We then turn this into an array with each row being one symbol with its cyclic prefix
symbolWithCyclicPrefixPrime=np.reshape(receivedData,(numberOfSymbols,onePeriod))

imgPrime = [[] for x in range(numberOfSymbols)] # define an empty array to fill in the grey-Scales

for i in range(numberOfSymbols):
    #5) decodeing the symbols from each period in the received data
    imgPrime[i]=decodeSymbol(symbolWithCyclicPrefixPrime[i])

# Figure(16) must show the original image correctly
plt.figure("t11")
plt.subplot(2,1,1)
plt.imshow(imgPrime, cmap ='gray')
plt.title("Image: After reception (After FFT)")
plt.subplot(2,1,2)
plt.hist(imgPrime)
plt.title("Histogram of pixles' greyScale")
plt.show()
