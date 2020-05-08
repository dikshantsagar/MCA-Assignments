import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.metrics import classification_report

print('Loading Base Data...')
with open('basedata.obj','rb') as f:
    basedata=pickle.load(f)

xtrain=basedata[0]
ytrain=basedata[1]
xval=basedata[2]
yval=basedata[3]

xspec=[]
xmfcc=[]
for i in xtrain:
    xspec.append(np.resize(i[0],(128,32)))
    xmfcc.append(np.resize(i[1],(20,32)))


vspec=[]
vmfcc=[]
for i in xval:
    vspec.append(np.resize(i[0],(128,32)))
    vmfcc.append(np.resize(i[1],(20,32)))



xspec=np.array(xspec)
xmfcc=np.array(xmfcc)
vspec=np.array(vspec)
vmfcc=np.array(vmfcc)

xspec=np.nan_to_num(xspec)
xmfcc=np.nan_to_num(xmfcc)
vspec=np.nan_to_num(vspec)
vmfcc=np.nan_to_num(vmfcc)

print('Fitting Model for Spectrogram...')
clf=SVC()
clf.fit(xspec.reshape(10000,4096),ytrain)
specpred=clf.predict(vspec.reshape(2494,4096))
print(classification_report(yval,specpred))

print('Fitting Model For MFCC....')
clf2=SVC()
clf2.fit(xmfcc.reshape(10000,640),ytrain)
mfccpred=clf2.predict(vmfcc.reshape(2494,640))
print(classification_report(yval,mfccpred))

print('Loading Noised Data....')
with open('noisedtrain.obj','rb') as f:
    noiseddata=pickle.load(f)

nxtrain=noiseddata[0]
nytrain=noiseddata[1]

nxspec=[]
nxmfcc=[]
for i in nxtrain:
    nxspec.append(np.resize(i[0],(128,32)))
    nxmfcc.append(np.resize(i[1],(20,32)))


nxspec=np.array(nxspec)
nxmfcc=np.array(nxmfcc)

nxspec=np.nan_to_num(nxspec)
nxmfcc=np.nan_to_num(nxmfcc)


print('Fitting Model for Spectrogram...')
clf3=SVC()
clf3.fit(nxspec.reshape(10000,4096),nytrain)
nspecpred=clf3.predict(vspec.reshape(2494,4096))
print(classification_report(yval,nspecpred))

print('Fitting Model For MFCC....')
clf4=SVC()
clf4.fit(nxmfcc.reshape(10000,640),nytrain)
nmfccpred=clf4.predict(vmfcc.reshape(2494,640))
print(classification_report(yval,nmfccpred))

