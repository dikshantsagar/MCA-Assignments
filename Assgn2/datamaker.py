import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

from scipy.fftpack import dct

import ques1
import ques2
import random
import glob


trainfiles=glob.glob('Dataset/training/*/*')
validationfiles=glob.glob('Dataset/validation/*/*')

target={
    'zero':0,
    'one':1,
    'two':2,
    'three':3,
    'four':4,
    'five':5,
    'six':6,
    'seven':7,
    'eight':8,
    'nine':9
}

w=150
xtrain=[]
ytrain=[]
xval=[]
yval=[]
for i in trainfiles:
    f,d=wavfile.read(i)
    spec=spectrogram(d.astype('float'),f)
    mf=mfcc(spec)
    xtrain.append(np.array([spec,mf]))
    label=i.split('/')[-2]
    ytrain.append(target[label])

for i in validationfiles:
    f,d=wavfile.read(i)
    spec=spectrogram(d.astype('float'),f)
    mf=mfcc(spec)
    xval.append(np.array([spec,mf]))
    label=i.split('/')[-2]
    yval.append(target[label])

    
    
import pickle
with open('basedata.obj','wb') as f:
    pickle.dump([xtrain,ytrain,xval,yval],f)


noisefiles=glob.glob('Dataset/_background_noise_/*')


xtrainaug=[]
ytrainaug=[]

for i in trainfiles:
    print(trainfiles.index(i))
    f,d=wavfile.read(i)
    noise=noisefiles[random.randint(0,5)]
    fn,dn=wavfile.read(noise)
    lim=len(dn)-len(d)-1
    ind=random.randint(0,lim-1)
    ndn=dn[ind:ind+len(d)]
    sig=ndn+d
    spec=spectrogram(sig.astype('float'),f)
    mf=mfcc(spec)
    xtrainaug.append(np.array([spec,mf]))
    label=i.split('/')[-2]
    ytrainaug.append(target[label])
    


with open('noisedtrain.obj','wb') as f:
    pickle.dump([xtrainaug,ytrainaug],f)