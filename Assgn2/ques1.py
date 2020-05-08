import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

f,d=wavfile.read('Dataset/training/nine/0a7c2a8d_nohash_1.wav')

plt.plot(d)
plt.xticks(np.arange(0,16000,4000),np.arange(0,1,0.00025))
plt.ylabel("Amplitude")
plt.xlabel("Time (second)")
plt.show()

def get_xn(Xs,n):
    '''
    Discrete Fourier Transform (DFT)
    '''
    L  = len(Xs)
    ks = np.arange(0,L,1)
    xn = np.sum(Xs*np.exp((1j*2*np.pi*ks*n)/L))/L
    return(xn)

def get_xns(ts):
    '''
    Compute Fourier coefficients only up to the Nyquest Limit Xn, n=1,...,L/2
    and multiply the absolute value of the Fourier coefficients by 2, 
    to account for the symetry of the Fourier coefficients above the Nyquest Limit. (L/2)
    '''
    mag = []
    L = len(ts)
    for n in range(int(L/2)): 
        mag.append(np.abs(get_xn(ts,n))*2)
    return(mag)

def get_Hz_scale_vec(ks,sample_rate,Npoints):
    freq_Hz = ks*sample_rate/Npoints
    freq_Hz  = [int(i) for i in freq_Hz ] 
    return(freq_Hz )

def spectrogram(ts,w):
  
    starts  = np.arange(0,len(ts),w,dtype=int)
    # remove any window with less than sample size
    starts  = starts[starts + w < len(ts)]
    xns = []
    for start in starts:
        # DFT
        ts_window = get_xns(ts[start:start + w]) 
        xns.append(ts_window)
    specX = np.array(xns).T
    # rescale the absolute value of the spectrogram 
    spec = 10*np.log10(specX)
    assert spec.shape[1] == len(starts) 
    return(starts,spec)



def plot_spectrogram(spec,ts,sample_rate):
    
    x=spec.shape[1]
    plt.figure(figsize=(20,8))
    ks      = np.linspace(0,spec.shape[0],10)
    ksHz    = get_Hz_scale_vec(ks,sample_rate,len(ts))
    
    plt.imshow(spec,origin='lower')
    plt.xticks(np.arange(0,x,x/5),np.arange(0,1,1/5,dtype='float32'))
    plt.yticks(ks,ksHz)
    plt.xlabel('Time(s)')
    plt.ylabel('Frequency')
    plt.colorbar()
    plt.show()
    

w = 150
starts, spec = spectrogram(d,w)

plot_spectrogram(spec,d,f)