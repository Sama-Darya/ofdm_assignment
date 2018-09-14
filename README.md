# OFDM / FFT assignment

This is an illustration for data transmission using FFT and IFFT in the context of OFDM and
a challenge to turn it into a practical full OFDM transmitter / receiver.

In this repository, you find the Python code, and an image used as example data.

## Transmission (tx)

  - Turning the data into a stream of bits
  - Scramble the bits with a known random sequence
  - Creating complex numbers from the bits
  - Creating a mirrored spectrum form it
  - Doing an inverse FFT
  - And finally transmitting the time series (in this case only saving it as an audio file)

## Reception (rx)

All above steps are decoded (reversed) 
and the original Image is reconstructed.

## Your task

However, in real-life applications, the start point of the data is lost in the transmission. 
We have added additional info which makes it possible to detect the start such as a "guard interval"
and "Pilot Tones". Use these on the receiver side to correctly find the start point!

## Differences to a full OFDM encoding

This assignment focusses on the FFT / IFFT. If you look in the code there are two bits per frequency
sample but of course one can encode more than two bits per frequency by introducing more levels. Here,
it's essentially QAM per frequency coefficient.

Usually the frequency coefficients are not mirrored which results in a complex time series. This
series is then turned into a real one with the help of a QAM encoder running at two times the
sampling rate of the complex series. Overall this results in exactly the same number of samples
transmitted but an IFFT with twice the length is by far more expensive than a quadrature modulator.

# Credits

Sama & Bernd
