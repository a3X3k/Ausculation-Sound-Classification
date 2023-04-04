```py
!pip install sounddevice
!sudo apt-get install libportaudio2
!pip install ffmpeg-python

import sounddevice
from scipy.io.wavfile import write
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import numpy as np
import pandas as pd
import os
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow
from tensorflow.keras.models import Model,load_model
from IPython.display import Javascript
from google.colab import output
from base64 import b64decode
```

```py
RECORD = """
const sleep  = time => new Promise(resolve => setTimeout(resolve, time))
const b2text = blob => new Promise(resolve => {
  const reader = new FileReader()
  reader.onloadend = e => resolve(e.srcElement.result)
  reader.readAsDataURL(blob)
})
var record = time => new Promise(async resolve => {
  stream = await navigator.mediaDevices.getUserMedia({ audio: true })
  recorder = new MediaRecorder(stream)
  chunks = []
  recorder.ondataavailable = e => chunks.push(e.data)
  recorder.start()
  await sleep(time)
  recorder.onstop = async ()=>{
    blob = new Blob(chunks)
    text = await b2text(blob)
    resolve(text)
  }
  recorder.stop()
})
"""

def record(sec = 10):
    display(Javascript(RECORD))
    s = output.eval_js('record(%d)' % (sec*1000))
    b = b64decode(s.split(',')[1])

    with open('Input.wav', 'wb') as f:
     f.write(b)
    return 'Input.wav'
    
record()
```

```py
Input = '/content/Input.wav'

Label = ['Bronchiectasis', 'Bronchiolitis', 'COPD', 'Healthy', 'Pneumonia', 'URTI', 'Wheezing']

Model_Loaded = load_model('/content/GRU 1 Weight.h5')

data_in=[]

data_x, sampling_rate = librosa.load(Input,  res_type = 'kaiser_fast')

mfccs = np.mean(librosa.feature.mfcc(y = data_x, sr = sampling_rate, n_mfcc = 40).T, axis = 0)

data_in.append(mfccs)

data_in=np.array(data_in)

data_in = data_in.reshape(data_in.shape[0], 1, data_in.shape[1])

ypred=Model_Loaded.predict(data_in)[0]

ypred = np.argmax(ypred, axis=1)

Label[ypred[0]]
```
