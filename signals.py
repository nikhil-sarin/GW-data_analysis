'''
The functions in this script are for gravitational-wave signal processing. 
Throughout, we work with "normalised FFT". We adopt the convention of 
lalinference and pycbc so that a 'normalised FFT' is
  nfft(xt) = numpy.fft(xt) / fs
where fs is the sampling frequency.
'''
import numpy as np
from scipy.interpolate import interp1d
import pdb
from scipy import signal as scisig

def nfft(ht, fs):
    '''
    performs NORMALISED FFT while keeping track of the frequency bins
    assumes input time series is real (positive frequencies only)
    input:
      ht = time series
      fs = sampling frequency
    output:
      hf = single-sided FFT of ht in units of seconds = np.fft(ht)/fs
      f = frequencies associated with hf
    '''
    # add one zero padding if time series does not have even number of
    # sampling times
    if np.mod(len(ht), 2) == 1:
        ht = np.append(ht, 0)
    L = len(ht)
    # frequency range
    f = fs / 2 * np.linspace(0, 1, L/2+1)

    # calculate FFT: rfft computes the fft for real inputs
    hf = np.fft.rfft(ht)

    # normalise to units of seconds
    hf = hf/fs

    return hf, f

def nifft(hf, fs):
    '''
    performs NORMALISED inverse FFT for use in conjunction with nfft
    input:
      hf = normalised, single-side FFT calculated by fft_eht
      fs = sampling frequency
    output:
      ht = time series
    '''
    # undo normalisation
    hf = hf*fs

    # use irfft to work with positive frequencies only
    ht = np.fft.irfft(hf)

    return ht

def inner_product(aa, bb, freq, PSD):
    '''
    calculate the noise-weighted inner product defined for use in matched 
    filtering: (aa, bb)
    input:
      aa = normalised single-side FFT
      bb = normalised single-side FFT
      freq = associated frequency bins
      PSD = noise power spectral density with two columns: f, Pf
    output:
      inner product = (aa, bb) with units = strain^2 / Hz
    '''
    # interpolate the PSD to the freq grid
    PSD_interp_func = interp1d(PSD[:, 0], PSD[:, 1], bounds_error=False, fill_value=np.inf)
    PSD_interp = PSD_interp_func(freq)

    # Note that extrapolated frequencies will receive PSD values of inf.
    # This is a good thing because this means that the noise is treated as 
    # infinite outside the range specified by the PSD array.

    # caluclate the inner product
    integrand = np.conj(aa) * bb / PSD_interp

    df = freq[1] - freq[0]
    integral = np.sum(integrand) * df
    product = 4. * np.real(integral)

    return product

def snr_exp(aa, freq, PSD):
    '''
    calculates the expectation value for the matched filter SNR of template aa
    this is also known as optimal SNR
    input:
      aa = normalised single-side FFT
      freq = associated frequency bins
      PSD = noise power spectral density with two columns: f, Pf
    output:
      optimal snr = expectation value of snr = (aa, aa)**0.5
    '''
    return np.sqrt(inner_product(aa, aa, freq, PSD))

def cal_snr(hf, muf, freq, PSD):
    '''
    calculate matched filter SNR for template muf given data hf
    input:
      hf = normalised single-side FFT for data
      muf = normalised single-side FFT for template
      freq = associated frequency bins
      PSD = noise power spectral density with two columns: f, Pf
    output:
      matched filter snr = (hf, muf) / (muf, muf)**0.5
    '''
    snr = inner_product(hf, muf, freq, PSD) / np.sqrt(inner_product(muf, muf, freq, PSD))
    return snr

def nextpow2(i):
    """
    find 2^n that is equal to or greater than i
    for use in gaussian_noise
    input
      i
    output
      n = next power of 2
    """
    n = 1
    while n < i: n *= 2
    return n

def gaussian_noise(PSD, fs, duration, T=1):
    '''
    generate Gaussian noise from a power spectral density
    adapted from gaussian_noise.m in matapps
    input:
      PSD = noise power spectral density with two columns: f, Pf
      fs = sampling frequency
      duration = length of data
      T = the number of trials (this option is not currently supported)
    output:
      hf = normalised single-side FFT of Gaussian noise
      f = associated frequency bins
    '''
    
    # calculate N = nuber of samples
    N = duration * fs
    N = int(np.round(N))

    # for debugging only, set random seed
    #seeds = [151226, 150914]
    #seed = seeds[1]
    #np.random.seed(seed)

    # prepare for FFT
    if ( np.mod(N,2)== 0 ):
        numFreqs = N/2 - 1
    else:
        numFreqs = (N-1)/2
    deltaF = 1./duration
    flow = deltaF

    # in python, start from DC
    #f = deltaF*np.linspace(0, numFreqs, numFreqs)
    f = deltaF*np.linspace(1, numFreqs, numFreqs)

    # next power of 2 from length of y  
    NFFT = 2^nextpow2(N)
    amp_values = PSD[:,1]
    f_transfer1 = PSD[:,0]
    Pf1_interp_func = interp1d(PSD[:, 0], PSD[:, 1], bounds_error=False, fill_value=np.inf)
    Pf1 = Pf1_interp_func(f)

    # remove infinities and replace with 0
    # In an older version of the code, the infinities were replaced with 1,
    # the thinking being that the extraploated noise should be something really
    # big compared to typical strain power. However, this can cause serious 
    # spectral leakage in band if the data are ifft'ed to the time domain,
    # windowed, and then fft'ed back to frequency domain. The best thing to do
    # is to set the noise equal to zero at these frequencies, and then set the
    # PSD to infinity when we calculate the inner product.
    if sum(np.isinf(Pf1)) > 0:
        Pf1[np.isinf(Pf1)] = 0
    deltaT = 1./fs
    norm1 = np.sqrt(N/(2*deltaT)) * np.sqrt(Pf1)
    re1 = norm1*np.sqrt(0.5) * np.random.randn(numFreqs)
    im1 = norm1*np.sqrt(0.5) * np.random.randn(numFreqs)
    z1  = re1 + 1j*im1

    # freq domain solution for htilde1, htilde2 in terms of z1, z2
    htilde1 = z1
    # convolve data with instrument transfer function
    otilde1 = htilde1*1.
    # set DC and Nyquist = 0
    # python: we are working entirely with positive frequencies
    if ( np.mod(N,2)==0 ):
        otilde1 = np.concatenate(([0], otilde1, [0]))
        f = np.concatenate(([0], f, [fs/2.]))
    else:
        # no Nyquist frequency when N=odd
        otilde1 = np.concatenate(([0], otilde1))
        f = np.concatenate(([0], f))
    
    # redefine variable following fft_eht.m
    hf = otilde1

    # python: transpose for use with infft
    hf = np.transpose(hf)
    f = np.transpose(f)

    # convert to normalised fft
    hf = hf/fs

    # python: return Fourier transform, not time series
    return hf,f

def psd(xf, fs):
    '''
    calculate power spectral density of normalised FFT
    input:
       xf: single-sided Fourier transform created by fft_eht 
       fs: sampling frequency
    output:
      Pf: power spectral density with proper normalization so that x has
      units of strain, then Pf has units of strain^2/Hz.
      Pf will have the same size as xf with bins corresponding to the same
      frequencies.
    '''
    # calculate the length of the time series used to create 
    length_xf = xf.shape[0]
    if np.mod(length_xf, 2):
        L = (2*(length_xf-1))
    else:
        L = (2*(length_xf-2))

    # undo normalisation
    xf = xf*fs

    # calculate PSD
    Pf = 2*abs(xf)**2 / L / fs

    return Pf

def asd(xf, fs):
    '''
    calculate amplitude spectral density of normalised FFT
    input:
      xf: single-sided Fourier transform created by fft_eht 
      fs: sampling frequency
    output
      af: amplitude spectral density with proper normalization so that x has
      units of strain, then af has units of strain/rHz.
      af will have the same size as xf with bins corresponding to the same
      frequencies.
    '''
    # calculate the length of the time series used to create 
    length_xf = xf.shape[0]
    if np.mod(length_xf, 2):
        L = (2*(length_xf-1))
    else:
        L = (2*(length_xf-2))

    # undo normalisation
    xf = xf*fs

    # calculate PSD
    af = np.sqrt(2*abs(xf)**2 / L / fs)

    return af

def psd_matrix(xf, fs):
    '''
    calculates a power spectral density covariance matrix between frequency f
    and frequency f'. useful for detailed studies of spectral leakage
    input:
      xf: single-sided Fourier transform created by fft_eht 
      fs: sampling frequency
    output
      Pf: power spectral density with proper normalization so that x has
    units of strain, then Pf has units of strain^2/Hz.
    Pf will have the same size as xf with bins corresponding to the same
    frequencies. The "matrix" version returns an outer product to investigate
    correlations between frequency bins
    '''
    # calculate the length of the time series used to create 
    length_xf = xf.shape[0]
    if np.mod(length_xf, 2):
        L = (2*(length_xf-1))
    else:
        L = (2*(length_xf-2))
        
    # undo normalisation
    xf = xf*fs

    # calculate PSD
    Pf = 2*np.outer(np.conj(np.transpose(xf)), xf) / L / fs

    return Pf

def window(ht, alpha=0.25):
    '''
    NOTE: this function is under construction. It currently employs only one 
    kind of a window, a Tukey window.
    ---------------------------------------------------------------------------
    generates a window with the same size as input strain data ht
    input:
      ht: strain data
    output:
      wt: window function
    '''
    L = len(ht)
    # note that you need a relatively up-to-date copy of scipy in order to have
    # this window
    #wt = scisig.hann(L)
    wt = scisig.tukey(L, alpha, sym=True)
    return wt
