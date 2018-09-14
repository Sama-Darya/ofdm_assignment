#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 10:09:39 2018

@author: bp1
"""

import scipy.ndimage as ndimage
import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt

### DATA TRANSMISSION

plt.close("all")

# 1) loading the data: in this case the data is an image.
img = ndimage.imread("greytee.png")

# At this point the data is a 100 by 256 array.
# you can check this by typing in the command line "img.shape"
# Entries of this array are the grey-scale values of their corresponding pixels
# Figure(00) shows this image and its grey-scale histogram
# grey-scale values are from 0 to 255, corresponding to black and white respectively. 

plt.figure("00")
plt.subplot(2,1,1)
plt.imshow(img, cmap ='gray')
plt.title("Image: Before transmission (before IFFT)")
plt.subplot(2,1,2)
plt.hist(img)
plt.title("Histogram of pixles' greyScale")


# 2) Turning the data into a 1D array:
img1D = np.reshape(img,(1,-1))
# The previously 100 by 256 array is now one dimensional with 1 row and 25600 columns
# This is like turning this 2 by 3 array: 
#                              [a, b, c]
#                              [d, e, f] into this 1 by 6 array:
#                                                         [a, b, c, d, e, f]                              
# Firgure(01) shows the data in this form

plt.figure("01")
plt.subplot(2,1,1)
plt.plot(img1D[0])
plt.title("image1D: Before IFFT")
plt.subplot(2,1,2)
plt.hist(img1D[0])
plt.title("Histogram of pixles' greyScale")


# 3) turning the data into a stream of bits:
img1Dbits = np.unpackbits(img1D)

# The previously grey-scale values have now become 8bit binary values.
# For example:
# The first data was previously 239 in the grey-scale; img1D[0,0]=239
# This data is now represented as: 11101111 in 8 bit binary; img1Dbits[0:8]
# This means we need 8 times as many entries as before to represent the data.
# That means expanding from 1 by 25600 array to 1 by 204800 array.
# figure(02) shows a small part of data,and its histogram confirms that
#           it only consists of 0s and 1s.

bins1=np.arange(-0.5,1.5,0.1)
plt.figure("02")
plt.subplot(2,1,1)
plt.plot(img1Dbits[0:60])
plt.title("image1Dbits: Before IFFT")
plt.subplot(2,1,2)
plt.hist(img1Dbits, bins1)
plt.title("Histogram of binary data")

# 4) Scramble the data: in this section we aim to randomise the data.
# This is to ensure that the data is not constant; if the data is constant it'll be
# hard to detect the start point of data after transmission using "Guard Intervals"

# The code below generates a random array that only consists of 1 or -1.
np.random.seed(0)
scrambler = np.random.randint(0,2,img1Dbits.size) * 2.0 - 1.0

# This array is then multiplied by the previous binary data.
# this will make some 1s to change sign and become -1 
img1DbitsRandom = img1Dbits * scrambler
# The data now only consists of 1s, 0s and -1s.
# Figure(03) shows a histogram that confirms this distribution 

bins2=np.arange(-1.5, 1.5, 0.2)
plt.figure("03")
plt.subplot(2,1,1)
plt.plot(np.real(img1DbitsRandom[200:250]))
plt.title("img1DbitsRandom: Before IFFT")
plt.subplot(2,1,2)
plt.hist(np.real(img1DbitsRandom) , bins2)
plt.title("Histogram of randomised data")

# 5) QAM coding:
# In this section we aim at creating complex numbers from the bits.
# starting from the first entry, we treat one bit as a real-valued number
# and treat the next entry as its complex part.
# For example if the first 4 data are: img1DbitsRandom[0:4] = -1 , 1 , 1 , 0
# these will make 2 complex data of: -1+1j and 1+0j
# The next 3 lines of code does just that

img1DReal = (img1DbitsRandom[0::2]) * 2.0 - 1.0
img1DImag = (img1DbitsRandom[1::2]) * 2.0 - 1.0
complexImage = img1DReal + img1DImag * 1j

# Notice how the data was scaled by the *2 multiplication and -1 addition.
# This will map the previous data from 1, 0, -1 to 1, -1, -3 respectively.


# 6) Creating a frequency Spectrum from the data:
# In order to this we need to appreciate of the main properties of a frequency spectrum:
#   "A frequency spectrum is always symmetrical (for real-valued time-domain data)"
# The only exception to this symmetry is the DC (zero) frequency. (see below ***)
# The next 2 lines of code do exactly that: adds a DC of zero and the mirror of data

complexMirror = np.conj(complexImage[::-1])
fullComplexDataWithMirror = np.concatenate((complexImage,
                                            complexMirror))
                                            
# We can now treat the data as a frequency-domain spectrum
# Figure(04) shows the center part of this frequency spectrum which confirms the
#            symmetry of it and its histogram confirming the distribution of [-3,-1,1]                        
                                            
bins3=np.arange(-3.5,2,0.2)                                            
plt.figure("04")
plt.subplot(2,1,1)
plt.plot(np.real(fullComplexDataWithMirror[102375:102425]))
plt.title("fullComplexDataWithMirror: Before IFFT")
plt.subplot(2,1,2)
plt.hist(np.real(fullComplexDataWithMirror), bins3)
plt.title("Histogram of frequency spectrum")

                                            
# 7) Pilot Tones:
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
pilotTonedDataReal[1::2]=1 #amplitude of 1
pilotTonedDataImag[1::2]=0 #Phase of Zero

pilotTonedData=pilotTonedDataReal + pilotTonedDataImag * 1j

# Figure(05) shows the spectrum with the pilot tones added, the histogram shows an 
# increase in the numbers of 1s, which corresponds to the pilot tones added
plt.figure("05")
plt.subplot(2,1,1)
plt.plot(np.real(pilotTonedData[204775:204825]))
plt.title("pilotTonedData: Before IFFT")
plt.subplot(2,1,2)
plt.hist(np.real(pilotTonedData), bins3)
plt.title("Histogram of frequency spectrum with pilot tones")

# 8) In this section the frequency spectrum is padded with some zeros in order to
# meet the requirements for sampling rate and minimum detectable frequency of the
# receiver device.

zero_padding=2 # pick an even number!
paddedPilotTonedData = np.concatenate((np.zeros(zero_padding+1),
                                       pilotTonedData,
                                       np.zeros(zero_padding)))

# Note that there is exactly one more zero added to the start of spectrum than the end.
# *** this is the DC frequency, which is the only exception to the symmetry of spectrum                                     

# 9) IFFT: performing an Inverse Fast Fourier Transformation on the frequency
# spectrum created above results in a time-domain data; which is then transmitted
symbol = np.real(np.fft.ifft(paddedPilotTonedData))

# 10) Guard Intervals: in this section another measure is put in place that will make
# possible to detect the start point of the data on the receiver side. 
# When the data is in time-domain, a small part of data from its end is taken and is
# added to the start of data; this repeated section is called "guard interval"
# For example: given this data: [a,1,b,2,c,3,d,4,e,5], and final 3 entries as the
# guard interval, we would have: [4,e,5, a,1,b,2,c,3,d,4,e,5]

CyclicPrefix=1100 #Cyclic Prefix is just another name for Guard Interval
symbolWithCyclicPrefix=np.concatenate((symbol[-CyclicPrefix:],
                                             symbol))

# 11) Making the data repetitive: when transmitting data it becomes repetitive, in this
# part the data is made repetitive to mimic the real-life transmission scenario
transmitData=np.concatenate((symbolWithCyclicPrefix,symbolWithCyclicPrefix,
                             symbolWithCyclicPrefix))

# 12) Data transmission: finally the data is saved as an audio file to be transmitted and 
# received using a loud speaker and microphone respectively.
wavfile.write('ofdmsignl2.wav',48000,transmitData) 
