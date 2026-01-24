import numpy as np
import matplotlib.pyplot as plt
from multivarious.dsp import psd
from numpy.fft import ifft
from scipy.signal import detrend

def autocorr(x,Fs):
    x = np.asarray(x)
    x = detrend(x, type='linear')
    n = len(x)

    padding_2 = False
    
    if padding_2: # Power of two for zero-padding (for speed)
        n = nfft = 2**(int(np.ceil(np.log2(2*n - 1))))
    else: 
        nfft = n  # no zero-padding

    tau = np.block ( [ np.arange(n/2-1) - (n/2-1) , np.arange(n/2) ] ) / Fs
    
    # FFT of the signal with zero-padding
    X = np.fft.fft(x, n=nfft)
    
    # Inverse |FFT|^2 to get autocorrelation
    Rxx = np.fft.ifft(X * np.conj(X)).real

    #print(f'n = {n}   len Rxx = {len(Rxx)}' ) 
    
    Rxx = np.block( [ Rxx[int(n/2)+1:] , Rxx[:int(n/2)] ] ) / n # / (n - np.arange(int(n/2)))
    X   = np.block( [ X[int(n/2)+1:] , X[:int(n/2)] ] )  

    return Rxx, tau, X

# Example usage and test
if __name__ == "__main__":

    interactive_plots = True

    # Example data: noisy sinunsoidal wave
    N = 500
    T = 10
    t = np.linspace(0, T, N)
    Fs = N/T
    df = 1/T
    print(f' sampleing frequency = {Fs:5.3f}, df = {1/T:5.3f}')
    data = np.cos(2*np.pi * t) + 0.10*np.random.randn(len(t))*np.sqrt(Fs)

    lag_1_corr = np.corrcoef(data[1:N],data[0:N-1])[1,0] 

    Rxx, tau, X = autocorr(data,Fs)

    Rxx = Rxx / Rxx[(int(len(Rxx)/2))] # normalize to lag-1 correlation

    print(f' lag-1 correlation = {lag_1_corr}, Rxx[1] = {Rxx[int(len(Rxx)/2)+1]}')

    if interactive_plots: 
        plt.ion() # interactive plots: on
    
    plt.rcParams.update({'font.size':12})

    plt.figure(1)
    plt.plot(t,data)
    plt.title('Time domain data')
    plt.xlabel('time')
    plt.ylabel('data')

    fig = plt.figure(2)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # you can adjust these values
    min_data = np.min(data)
    max_data = np.max(data)
    plt.plot(data[1:N],data[0:N-1], 'o')
    plt.plot([min_data, max_data], [min_data, max_data], '-k', linewidth = 1.5, alpha=1)
    plt.axis('equal')
    plt.title('lag-1 correlation')
    plt.text(0.9*min_data, 0.9*max_data, rf"$\rho_{{ k,k \pm 1}}$ = {lag_1_corr:6.3f}")
    plt.xlabel('data[k]')
    plt.ylabel('data[k-1]')
    ax.set_box_aspect(1)  # 1 means square box (height = width)

    plt.figure(3)
    n = len(X)
    f = np.block ( [ np.arange(n/2-1) - (n/2-1) , np.arange(n/2) ] ) * df
    plt.plot(f, X.real)
    plt.plot(f, X.imag)
    plt.title('FFT')
    plt.xlabel(rf'frequency, $f$')
    plt.ylabel(rf'FFT $X$ real and imaginary')

    plt.figure(4)
    plt.plot(tau,Rxx)
    plt.title('Autocorrelation computed via FFT')
    plt.xlabel('lag time')
    plt.ylabel(rf'autocorrelation, $R(\tau)$')

    if not interactive_plots: 
        plt.show()
    else:
        input(' ... enter to exit ... ')
        plt.close('all') 
 
