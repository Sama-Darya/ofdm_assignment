#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Bernd & Sama
"""

import scipy.ndimage as ndimage
import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt

### DATA TRANSMISSION

cyclicPrefix=1100 # Cyclic Prefix is just another name for Guard Interval
zero_padding=2 # pick an even number!
nPilots=20 # number of pilots

def createSymbol(data):
    # 2.1) turning the data into a stream of bits -1,+1
    img1Dbits = np.unpackbits(data) * 2.0 - 1.0

    # The previously grey-scale values have now become 8bit binary values.
    # For example:
    # The first data was previously 239 in the grey-scale; img1D[0,0]=239
    # This data is now represented as: 11101111 in 8 bit binary; img1Dbits[0:8]
    # This means we need 8 times as many entries as before to represent the data.
    # That means expanding from 1 by 25600 array to 1 by 204800 array.

    # 2.2) Scramble the data: in this section we aim to randomise the data.
    # This is to ensure that the data is not constant; if the data is constant it'll be
    # hard to detect the start point of data after transmission using "Guard Intervals"

    # The code below generates a random array that only consists of 1 or -1.
    np.random.seed(0)
    scrambler = np.random.randint(0,2,img1Dbits.size) * 2.0 - 1.0

    # This array is then multiplied by the previous binary data.
    # this will make some 1s to change sign and become -1 
    img1DbitsRandom = img1Dbits * scrambler
    
    # The data now only consists of 1s, 0s and -1s.

    # 2.3) Creating complex frequency coefficients
    # In this section we aim at creating complex numbers from the bits.
    # starting from the first entry, we treat one bit as a real-valued number
    # and treat the next entry as its complex part.
    # For example if the first 4 data are: img1DbitsRandom[0:4] = -1 , 1 , 1 , 0
    # these will make 2 complex data of: -1+1j and 1+0j
    # The next 3 lines of code does just that

    img1DReal = (img1DbitsRandom[0::2])
    img1DImag = (img1DbitsRandom[1::2])
    complexImage = img1DReal + img1DImag * 1j
    #print(complexImage)

    # Notice how the data was scaled by the *2 multiplication and -1 addition.
    # This will map the previous data from 1, 0, -1 to 1, -1, -3 respectively.


    # 2.4) Creating a frequency Spectrum from the data:
    # In order to this we need to appreciate of the main properties of a frequency spectrum:
    #   "A frequency spectrum is always symmetrical (for real-valued time-domain data)"
    # The only exception to this symmetry is the DC (zero) frequency. (see below ***)
    # The next 2 lines of code do exactly that: adds a DC of zero and the mirror of data

    complexMirror = np.conj(complexImage[::-1])
    fullComplexDataWithMirror = np.concatenate((complexImage,
                                                complexMirror))
                                            
                                            
    # 2.5) Pilot Tones:
    # In this section pilot tones are added to the frequency domain. The purpose of this
    # is to help locate the exact Start Point of the data in the receiver side.
    # More specifically, we add a frequency of 1+0j in between the already existing
    # frequencies.
    # On the receiver side, if the start point is detected correctly, we will be able to
    # reconstruct these pilot frequencies as 1+0j , However, if the start point is
    # wrongly located, these frequencies will experience a phase shift and a change in
    # amplitude. For example: you may detect 0.996+0.008j instead of the original 1+0j
                                            
    # The next few lines of code add these pilot frequencies to the spectrum

    pilotTonedDataReal=np.zeros(2 * len(fullComplexDataWithMirror)-1) # -1 for symmetry
    pilotTonedDataImag=np.zeros(2 * len(fullComplexDataWithMirror)-1) # -1 for symmetry
    pilotTonedDataReal[0::2]=np.real(fullComplexDataWithMirror)
    pilotTonedDataImag[0::2]=np.imag(fullComplexDataWithMirror)
    pilotTonedDataReal[1::2]=0
    pilotTonedDataImag[1::2]=0
    pilotTonedDataReal[1::int(len(pilotTonedDataReal)/nPilots)] = 1 # a few pilots but not too many

    pilotTonedData=pilotTonedDataReal + pilotTonedDataImag * 1j

    # 2.6) In this section the frequency spectrum is padded with some zeros in order to
    # meet the requirements for sampling rate and minimum detectable frequency of the
    # receiver device.

    paddedPilotTonedData = np.concatenate((np.zeros(zero_padding+1),
                                           pilotTonedData,
                                           np.zeros(zero_padding)))

    # Note that there is exactly one more zero added to the start of spectrum than the end.
    # *** this is the DC frequency, which is the only exception to the symmetry of spectrum

    # 2.7) IFFT: performing an Inverse Fast Fourier Transformation on the frequency
    # spectrum created above results in a time-domain data; which is then transmitted
    symbol = np.real(np.fft.ifft(paddedPilotTonedData))
    return symbol
    

### DATA TRANSMISSION

plt.close("all")

# 1) loading the data: in this case the data is an image.
img = ndimage.imread("greytee.png")

# At this point the data is a 100 by 256 array.
# you can check this by typing in the command line "img.shape"
# Entries of this array are the grey-scale values of their corresponding pixels
# Figure(00) shows this image and its grey-scale histogram
# grey-scale values are from 0 to 255, corresponding to black and white respectively. 

plt.subplot(2,1,1)
plt.imshow(img, cmap ='gray')
plt.title("Image: Before transmission (before IFFT)")
plt.subplot(2,1,2)
plt.hist(img)
plt.title("Histogram of pixles' greyScale")

transmitData = []

for i in range(len(img)):
    # 2) creating the symbols
    symbol = createSymbol(img[i])

    # 3) Cyclic Prefix:
    # The symbol is in time-domain, a small part of data from its end is taken and is
    # added to the start of data; this repeated section is called "guard interval"
    # or cyclic prefix.
    # For example: given this data: [a,1,b,2,c,3,d,4,e,5], and final 3 entries as the
    # guard interval, we would have: [4,e,5, a,1,b,2,c,3,d,4,e,5]

    symbolWithCyclicPrefix=np.concatenate((symbol[-cyclicPrefix:],
                                           symbol))

    # We add the symbol to our datastream
    transmitData=np.concatenate((transmitData, symbolWithCyclicPrefix))

# 4) Data transmission: finally the data is saved as an audio file to be transmitted and 
# received using a loud speaker and microphone respectively.

transmitDataNorm=transmitData/(max(transmitData)-min(transmitData))
transmitDataScaled=transmitDataNorm * 30000
transmitDataInt=transmitDataScaled.astype(np.int16)
wavfile.write('ofdmSignal.wav',44100,transmitDataInt) 
