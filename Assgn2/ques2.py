
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import dct
from ques1 import spectrogram

f,d=wavfile.read('Dataset/training/nine/0a7c2a8d_nohash_1.wav')

plt.plot(d)
plt.xticks(np.arange(0,16000,4000),np.arange(0,1,0.00025))
plt.ylabel("Amplitude")
plt.xlabel("Time (second)")
plt.show()



def mel(spec):
    return(2595*np.log10(1+(spec/700)))

def mfcc(spec):
    
    #melspectrogram from spectrogram via mel conversion from frequency
    melspec=mel(spec)
    #dct filter on mel spectrogram
    mfcc=dct(melspec)
    
    return mfcc
    

w=150
_,spec=spectrogram(d,w)

tt=mfcc(spec)
plt.imshow(tt)
plt.show()