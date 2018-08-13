'''
Vectorized versions of common tools used in gravitational-wave data analysis
nikhil.sarin@ligo.org
'''
def nfft(ht, Fs):
    '''
    performs an FFT while keeping track of the frequency bins
    assumes input time series is real (positive frequencies only)

    ht = time series
    Fs = sampling frequency

    returns
    hf = single-sided FFT of ft normalised to units of strain / sqrt(Hz)
    f = frequencies associated with hf
    '''
    # add one zero padding if time series does not have even number of sampling times
    if np.mod(max(np.shape(ht)), 2) == 1:
        ht = np.append(ht, 0)
    LL = max(np.shape(ht))
    # frequency range
    ff = Fs / 2 * np.linspace(0, 1, LL/2+1)

    # calculate FFT
    # rfft computes the fft for real inputs
    hf = np.fft.rfft(ht)

    # normalise to units of strain / sqrt(Hz)
    hf = hf / Fs

    return hf, ff

def infft(hf, Fs):
    '''
    inverse FFT for use in conjunction with nfft
    eric.thrane@ligo.org
    input:
    hf = single-side FFT calculated by fft_eht
    Fs = sampling frequency
    output:
    h = time series
    '''
    # use irfft to work with positive frequencies only
    h = np.fft.irfft(hf)
    # undo LAL
    h = h*Fs

    return h


def inner_product(aa, bb, freq, PSD):
    '''
    Calculate the inner product defined in the matched filter statistic

    arguments:
    aai, bb: single-sided Fourier transform, created, e.g., by the nfft function above
    freq: an array of frequencies associated with aa, bb, also returned by nfft
    PSD: an Nx2 array describing the noise power spectral density

    Returns:
    The matched filter inner product for aa and bb
    '''
    # interpolate the PSD to the freq grid
    PSD_interp_func = interp1d(PSD[:, 0], PSD[:, 1], bounds_error=False, fill_value=np.inf)
    PSD_interp = PSD_interp_func(freq)

    # caluclate the inner product
    integrand = np.conj(aa) * bb / PSD_interp

    df = freq[1] - freq[0]

    if len(aa.shape) == 2:
        integral = np.sum(integrand,axis=1) * df
    else:
        integral = np.sum(integrand) * df

    product = 4. * np.real(integral)

    return product


def snr_exp(aa, freq, PSD):
    '''
    Calculates the expectation value for the optimal matched filter SNR

    arguments:
    aa: single-sided Fourier transform, created, e.g., by the nfft function above
    freq: an array of frequencies associated with aa, also returned by nfft
    PSD: an Nx2 array describing the noise power spectral density

    Returns:
    (The expectation value of) the matched filter SNR for aa
    '''
    return np.sqrt(inner_product(aa, aa, freq, PSD))

def snr_matchedfilter(hf, muf, f, Sh):
    '''
    eric.thrane@ligo.org
    calculate matched filter SNR for template muf given data hf
    '''
    snr = inner_product(hf, muf, f, Sh) / np.sqrt(inner_product(muf, muf, f, Sh))
    return snr
