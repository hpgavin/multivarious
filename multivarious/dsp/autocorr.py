import numpy as np
import matplotlib.pyplot as plt
from multivarious.dsp import psd
from numpy.fft import ifft

def autocorr(x,Fs):
    x = np.asarray(x)
    x = x - np.mean(x)
    n = len(x)
    
    # Power spectrum (element-wise multiplication by complex conjugate)
    #PF = psd(x, Fs) 
    #Pxx = PF[0]
    #f = PF[1]
    # Inverse FFT to get autocorrelation
    # Rxx = ifft(Pxx).real # / (n - np.arange(n))

    # Next power of two for zero-padding (for speed)
    nfft = 2**(int(np.ceil(np.log2(2*n - 1))))
    
    # FFT of the signal with zero-padding
    X = np.fft.fft(x, n=nfft)
    
    # Power spectrum (element-wise multiplication by complex conjugate)
    Pxx = X * np.conj(X)
    
    # Inverse FFT to get autocorrelation
    Rxx = np.fft.ifft(Pxx).real

    Rxx = Rxx[:n] / (n - np.arange(n))

    return Rxx, Pxx

# Example usage and test
if __name__ == "__main__":

    # Example data: noisy sine wave
    N = 500
    T = 10
    t = np.linspace(0, T, N)
    Fs = N/T
    data = np.sin(2*np.pi * t) + 0.50*np.random.randn(len(t))
    
    Rxx, Pxx = autocorr(data,Fs)

    plt.figure(1)
    plt.plot(t,data)
    plt.title('Time domain data')
    plt.xlabel('time')
    plt.ylabel('data')

    plt.figure(2)
    plt.plot(Rxx)
    plt.title('Autocorrelation computed via FFT')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')

    plt.figure(3)
    plt.semilogy(Pxx[:N])
    plt.title('Power Spectrum via FFT')
    plt.xlabel('frequancy')
    plt.ylabel('Power Spectum')

    plt.show()
