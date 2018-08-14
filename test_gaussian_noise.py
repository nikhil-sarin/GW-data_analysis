#!/bin/python -u
# eric.thrane@ligo.org

# imports
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tools as tools
import pdb

# load the detector noise PSD
Sh = np.loadtxt('./NoiseCurves/aLIGO_ZERO_DET_high_P_psd.txt')

# sample rate in Hz
#fs = 8192.
fs = 1024.
#fs = 2148.

# duration
dur = 10.
deltaF = 1./dur

# generate Gaussian noise
hf,f = tools.gaussian_noise(Sh, fs, dur)

# average Gaussian noise
navg = 1000
for x in range(0, navg):
    hf_tmp,f_tmp = tools.gaussian_noise(Sh, fs, dur)
    if x==0:
        psd = abs(hf_tmp)**2
    else:
        psd = psd + abs(hf_tmp)**2
psd = psd/navg

# convert to time domain
ht = tools.infft(hf, fs)

# time stamps
N = int(np.round(dur*fs))
t = (1./fs) * np.linspace(0, N, N)

# generate some toy-model signal for matched filtering SNR testing
navg = 100
snr = np.zeros(navg)
mu = np.exp(-(t-dur/2.)**2 / (2.*0.1**2)) * np.sin(2*np.pi*100*t)
muf,ff = tools.nfft(mu, fs)
for x in range(0, navg):
    hf_tmp,f_tmp = tools.gaussian_noise(Sh, fs, dur)
    snr[x] = tools.snr_matchedfilter(hf_tmp, muf, f, Sh)
print ("std(snr) = %1.3f" % np.std(snr))
print ("mean(snr) = %1.3f" % np.mean(snr))

# time series plot
plt.close('all')
plt.plot(t, np.transpose(ht))
plt.plot(t, np.transpose(mu) * np.max(np.abs(ht)))
plt.savefig('ht.png')

# Fourier transform AGAIN as a sanity check
hf2,f2 = tools.nfft(ht, fs)

# amplitude spectral density
plt.close('all')
plt.loglog(f, np.abs(np.transpose(hf)))
plt.loglog(f, np.sqrt(np.transpose(psd)), 'c')
plt.loglog(Sh[:,0], np.sqrt(Sh[:,1]/(deltaF*2)), 'r', lw=2)
plt.xlim(10, 2000)
plt.ylim(1e-24, 1e-20)
plt.savefig('hf.png')
