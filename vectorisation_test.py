#setup waveform and PSD
Sh = np.loadtxt('aLIGO_ZERO_DET_high_P_psd.txt')
fs = 1024.
dur = 10.
N = int(np.round(dur*fs))
t = (1./fs) * np.linspace(0, N, N)
AA = np.arange(0,10,1)/1e22

#create a waveform with same time series but different amplitude based on the amplitude term AA
mu = AA[:,None]*np.sin(2*np.pi*100*t)

#create waveforms in for loop
snr = np.zeros(len(AA))
inner_prod = np.zeros(len(AA))
for x in range(0,len(AA)):
    muf, freq = vec.nfft(mu[x], fs)
    inner_prod[x] = vec.inner_product(muf, muf, freq, Sh)
    snr[x] = vec.snr_exp(muf, freq, Sh)

#create vectorised waveforms
vuf, freq = vec.nfft(mu, fs)
v_inner_prod = vec.inner_product(vuf, vuf, freq, Sh)
v_snr = vec.snr_exp(vuf, freq, Sh)

#test whether vectorised SNR and inner product are the same as the for loop
print(v_snr == snr)
print(v_inner_prod == inner_prod)
