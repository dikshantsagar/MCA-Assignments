import nltk
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from IPython import display
import matplotlib.pyplot as plt

from keras.preprocessing.sequence import  skipgrams , make_sampling_table
from keras.layers import Dense , Flatten , Input , Reshape , dot
from keras.models import Sequential , Model
from keras.layers.embeddings import Embedding
from keras.callbacks import LambdaCallback
import pickle

nltk.download('abc')
abc=nltk.corpus.abc.words()
tkn=Tokenizer()
tkn.fit_on_texts(abc)
seq= tkn.texts_to_sequences(abc)
vlen=len(tkn.word_index)


sampling_table = make_sampling_table(vlen+1)
couples, labels = skipgrams(seq, vlen, window_size=5, sampling_table=sampling_table)

xt1=[]
xt2=[]
for i in couples:
  xt1.append(i[0])
  xt2.append(i[1])

from pandas.core.common import flatten
xt1=list(flatten(xt1))
xt2=list(flatten(xt2))
xt1=np.array(xt1)
xt2=np.array(xt2)

wembed_size=128

input_target = Input((1,))
input_context = Input((1,))
embedding = Embedding(vlen+1, wembed_size, input_length=1, name='embedding')

target = embedding(input_target)
target = Reshape((wembed_size, 1))(target)
context = embedding(input_context)
context = Reshape((wembed_size, 1))(context)

dot_product = dot([target, context],axes=1)
dot_product = Reshape((1,))(dot_product)
output = Dense(1, activation='sigmoid')(dot_product)

model = Model(input=[input_target, input_context], output=output)
model.compile(loss='binary_crossentropy', optimizer='rmsprop',metrics=['accuracy'])

def plot(model):
  display.clear_output(wait=True)
  emb=model.layers[2].get_weights()[0]
  words_embeddings = {w:emb[idx] for w, idx in tkn.word_index.items()}
  vis=np.array(list(words_embeddings.values())[:100])
  visem = TSNE(n_components=2).fit_transform(vis)
  x,y=visem[:,0],visem[:,1]
  n=list(words_embeddings.keys())[:100]
  fig, ax = plt.subplots(figsize=(10,7))
  ax.scatter(x,y)
  for i, txt in enumerate(n):
    ax.annotate(txt, (x[i], y[i]))

  plt.show()

plotcall = LambdaCallback(on_epoch_end=lambda epoch,logs: plot(model))

history=model.fit([xt1,xt2],labels,batch_size=6400,epochs=20,callbacks=[plotcall])


plt.title('Accuracy')
plt.plot(history.history['accuracy'],label='Training')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


plt.title('Loss')
plt.plot(history.history['loss'],label='Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

