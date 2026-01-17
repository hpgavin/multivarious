import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import chi2

def mimo_tfe(u, y, Fs, nfft, figNo=0):
    """
    Frequency response function of MIMO system from discrete-time I/O data.
    
    Uses Welch's averaged periodogram method, in which the matrices u and y are
    segmented, detrended, and windowed. The magnitude squared of the FFTs of the
    segments of U and Y are averaged to form Puu and Pyy. The products of the
    FFTs of the segments of U and Y are averaged to form Pyu. Hv is the
    geometric mean of the H1 and H2 estimates where H1=Pyu/Puu and H2=Pyy/Pyu.
    The coherence is the ratio H1/H2, and indicates the degree to which 
    the output, y, is linearly related to the input, u.
    
    Parameters:
        u     : matrix of sampled input to the system (r x points)
        y     : matrix of sampled output of the system (m x points)
        Fs    : sampling frequency = 1 / delta_t (scalar)
        nfft  : number of points used in FFT (will be rounded to 2^n)
        figNo : figure number for plotting (default: 0 = no plot)
    
    Returns:
        Hv  : complex-valued Hv freq. response fctn estimate (nfft/2 x m x r)
        f   : frequency values corresponding to H (nfft/2,)
        Puu : Hermitian matrix of power spectra of u (nfft/2 x r x r)
        Pyy : Hermitian matrix of power spectra of y (nfft/2 x m x m)
        Pyu : complex-valued cross-power spectra from u to y (nfft/2 x m x r)
        coh : coherence function (nfft/2 x m x r)
        Sv  : standard deviation of freq. resp. fctn estimate (nfft/2 x m x r)
    
    Author: H.P. Gavin 2017-12-03
    """
    
    u = np.asarray(u)
    y = np.asarray(y)
    
    r, nu = u.shape  # number of inputs and data points
    m, ny = y.shape  # number of outputs and data points
    
    if nu != ny:
        raise ValueError('U and Y must have the same number of columns.')
    else:
        n = nu
    
    nfft = 2**int(np.round(np.log2(nfft)))  # round to closest power of 2
    
    # Defaults
    make_Hv_causal = False  # make causal frequency response function
    nWindow = nfft          # number of points in a segment of data
    dflag = 'linear'        # straight-line detrending of data
    nOverlap = nWindow // 2 # number of overlapping points
    index_vals = np.arange(nWindow)
    Pw = nfft // 20         # number of points in rectangular taper
    
    # Create tapered rectangular window
    taper_left = 0.5 * (1 - np.cos(np.pi * np.arange(Pw + 1) / Pw))
    taper_right = 0.5 * (1 + np.cos(np.pi * np.arange(Pw + 1) / Pw))
    window = np.concatenate([
        taper_left,
        np.ones(nWindow - 2*Pw - 2),
        taper_right
    ])
    
    K = int(np.fix((n - nOverlap) / (nWindow - nOverlap)))  # number of windows
    
    # Normalizing scale factor ==> asymptotically unbiased
    KNW2 = K * np.linalg.norm(window)**2
    
    # Compute PSD matrices from FFTs and store in 3D arrays
    UU = np.zeros((nfft, r, r), dtype=complex)
    Puu = np.zeros((nfft, r, r), dtype=complex)
    YY = np.zeros((nfft, m, m), dtype=complex)
    Pyy = np.zeros((nfft, m, m), dtype=complex)
    YU = np.zeros((nfft, m, r), dtype=complex)
    Pyu = np.zeros((nfft, m, r), dtype=complex)
    
    # Transpose for compatibility with detrend
    u = u.T
    y = y.T
    
    index = np.arange(nWindow)
    
    for k in range(K):
        
        if dflag == 'linear':
            uw = (window * signal.detrend(u[index, :], axis=0, type='linear').T)
            yw = (window * signal.detrend(y[index, :], axis=0, type='linear').T)
        elif dflag == 'mean':
            uw = (window * signal.detrend(u[index, :], axis=0, type='constant').T)
            yw = (window * signal.detrend(y[index, :], axis=0, type='constant').T)
        else:
            uw = window * u[index, :].T
            yw = window * y[index, :].T
        
        U = np.fft.fft(uw, nfft, axis=1)
        Y = np.fft.fft(yw, nfft, axis=1)
        
        for ii in range(r):
            for jj in range(r):
                UU[:, ii, jj] = U[ii, :] * np.conj(U[jj, :])
        
        for ii in range(m):
            for jj in range(m):
                YY[:, ii, jj] = Y[ii, :] * np.conj(Y[jj, :])
        
        for ii in range(m):
            for jj in range(r):
                YU[:, ii, jj] = Y[ii, :] * np.conj(U[jj, :])
        
        Puu += UU
        Pyy += YY
        Pyu += YU
        
        index = index + (nWindow - nOverlap)  # advance to next segment
    
    # Compute H1, H2, and Hv frequency response function estimates
    H1 = np.zeros((nfft, m, r), dtype=complex)
    H2 = np.zeros((nfft, m, r), dtype=complex)
    
    for kk in range(nfft):
        puu = Puu[kk, :, :].reshape(r, r)
        pyy = Pyy[kk, :, :].reshape(m, m)
        pyu = Pyu[kk, :, :].reshape(m, r)
        
        H1[kk, :, :] = (pyu @ puu.conj().T) @ np.linalg.inv(puu @ puu.conj().T)
        H2[kk, :, :] = np.linalg.inv(pyu.conj().T @ pyy) @ (pyy @ pyy.conj().T)
    
    Hv = np.sqrt(H1) * np.sqrt(H2)  # Hv frequency response function estimate
    coh = np.real(H1 / H2)          # coherence
    
    if make_Hv_causal:
        hv = np.fft.ifft(Hv, axis=0)  # impulse response function
        hv_causal = np.real(np.vstack([hv[:nfft//2+1, :, :], 
                                       np.zeros((nfft//2-1, m, r))]))
        Hv = np.fft.fft(hv_causal, axis=0)  # frequency response for causal h
    
    # Normalize double-sided spectral density to agree with Parseval's theorem
    Puu = Puu / (KNW2 * Fs)
    Pyy = Pyy / (KNW2 * Fs)
    Pyu = Pyu / (KNW2 * Fs)
    
    # For real-valued signals, retain only non-negative frequencies
    if not np.any(np.imag(np.concatenate([u.flatten(), y.flatten()])) != 0):
        # Return first half: non-negative frequencies
        if nfft % 2:  # nfft is odd
            nn_idx = np.arange((nfft + 1) // 2)
        else:  # nfft is even, include DC and Nyquist
            nn_idx = np.arange(nfft // 2 + 1)
        
        Puu = Puu[nn_idx, :, :]
        Pyy = Pyy[nn_idx, :, :]
        Pyu = Pyu[nn_idx, :, :]
        Hv = Hv[nn_idx, :, :]
        coh = coh[nn_idx, :, :]
        f = nn_idx * Fs / nfft
    else:
        # Return positive and negative frequencies
        f_pos = np.arange(nfft // 2 + 1)
        f_neg = np.arange(-nfft // 2 + 1, 0)
        f = np.concatenate([f_pos, f_neg]) * Fs / nfft
    
    # Confidence Intervals
    p = 0.864  # probability level for confidence intervals
    alpha = 1 - p
    CI = (K - 1) / np.array([chi2.ppf(alpha/2, K-1), chi2.ppf(1-alpha/2, K-1)])
    Sv = np.abs(Hv) * np.sqrt(0.5 * (CI[0] - CI[1])) / K
    
    # PLOTS
    if figNo > 0:
        Hp = Hv  # which frequency response estimate to plot
        nf = len(f)
        idx = np.arange(1, nfft // 8)
        ff = [f[idx[0]], f[idx[-1]]]
        
        # Unwrap phase for plotting
        pha = np.full((nf, m, r), np.nan)
        pha2 = -np.arctan2(np.imag(Hp[1, :, :]), np.real(Hp[1, :, :]))
        pha[0, :, :] = pha2
        pha[1, :, :] = pha2
        pha2_rep = np.tile(pha2, (nf-2, 1, 1))
        d_pha = np.angle(Hp[2:nf, :, :] / Hp[1:nf-1, :, :])
        dp = np.pi
        d_pha[d_pha > dp] -= 2*np.pi
        d_pha[d_pha < -dp] += 2*np.pi
        pha[2:nf, :, :] = pha2_rep - np.cumsum(d_pha, axis=0)
        
        pha_min = np.floor(np.nanmin(pha[idx, :, :] * 180/np.pi) / 90) * 90
        pha_max = np.ceil(np.nanmax(pha[idx, :, :] * 180/np.pi) / 90) * 90
        
        # INPUT auto power spectra and cross spectra
        plt.figure(figNo)
        plt.clf()
        for ii in range(r):
            for jj in range(r):
                plt.subplot(r, r, ii*r + jj + 1)
                if ii == jj:
                    plt.semilogx(f[idx], np.real(Puu[idx, ii, jj]))
                else:
                    plt.semilogx(f[idx], np.real(Puu[idx, ii, jj]), 
                               f[idx], np.imag(Puu[idx, ii, jj]))
                plt.axis('tight')
                if ii == 0:
                    plt.title(f'u_{jj+1}')
                if jj == 0:
                    plt.ylabel(f'u_{ii+1}')
                if ii == r-1:
                    plt.xlabel('freq (Hz)')
        plt.tight_layout()
        
        # OUTPUT auto power spectra and cross spectra
        plt.figure(figNo + 1)
        plt.clf()
        for ii in range(m):
            for jj in range(m):
                plt.subplot(m, m, ii*m + jj + 1)
                if ii == jj:
                    plt.semilogx(f[idx], np.real(Pyy[idx, ii, jj]))
                else:
                    plt.semilogx(f[idx], np.real(Pyy[idx, ii, jj]),
                               f[idx], np.imag(Pyy[idx, ii, jj]))
                plt.axis('tight')
                if ii == 0:
                    plt.title(f'y_{jj+1}')
                if jj == 0:
                    plt.ylabel(f'y_{ii+1}')
                if ii == m-1:
                    plt.xlabel('freq (Hz)')
        plt.tight_layout()
        
        # Frequency Response Functions
        plt.figure(figNo + 2)
        plt.clf()
        for k in range(r):
            plt.subplot(3, r, k + 1)
            plt.semilogy(f[idx], np.abs(Hp[idx, :, k]))
            plt.axis('tight')
            if k == 0:
                plt.ylabel('magnitude')
            
            plt.subplot(3, r, k + r + 1)
            plt.plot(f[idx], 180/np.pi * pha[idx, :, k])
            plt.yticks(np.arange(pha_min, pha_max + 1, 90))
            plt.axis([ff[0], ff[1], pha_min, pha_max])
            if k == 0:
                plt.ylabel('phase (deg)')
            
            plt.subplot(3, r, k + 2*r + 1)
            plt.plot(f[idx], coh[idx, :, k])
            plt.axis([np.min(f[idx]), np.max(f[idx]), 0, 1])
            if k == 0:
                plt.ylabel('coherence')
            plt.xlabel('frequency (Hertz)')
            plt.grid(True)
        plt.tight_layout()
    
    return Hv, f, Puu, Pyy, Pyu, coh, Sv


# ------------------------------------------------------------------- mimo_tfe
# H.P. Gavin 2017-12-03

# Example usage
if __name__ == "__main__":
    print("Testing mimo_tfe: MIMO Frequency Response Function Estimation\n")
    
    # Generate test data: simple 2-input, 2-output system
    Fs = 1000  # sampling frequency
    T = 10     # duration
    t = np.arange(0, T, 1/Fs)
    
    # Random inputs
    u1 = np.random.randn(len(t))
    u2 = np.random.randn(len(t))
    u = np.vstack([u1, u2])
    
    # Simple transfer functions (filtered outputs)
    from scipy.signal import butter, lfilter
    
    b1, a1 = butter(4, 50 / (Fs/2))
    b2, a2 = butter(4, 100 / (Fs/2))
    
    y1 = lfilter(b1, a1, u1) + 0.5 * lfilter(b2, a2, u2)
    y2 = 0.3 * lfilter(b2, a2, u1) + lfilter(b1, a1, u2)
    
    # Add noise
    y1 += 0.1 * np.random.randn(len(t))
    y2 += 0.1 * np.random.randn(len(t))
    
    y = np.vstack([y1, y2])
    
    # Estimate frequency response function
    nfft = 512
    Hv, f, Puu, Pyy, Pyu, coh, Sv = mimo_tfe(u, y, Fs, nfft, figNo=1)
    
    print(f"Hv shape: {Hv.shape}")
    print(f"Frequency vector shape: {f.shape}")
    print(f"Number of windows averaged: {int((len(t) - nfft//2) / (nfft//2))}")
    print(f"Frequency resolution: {f[1] - f[0]:.2f} Hz")
    
    plt.show()
