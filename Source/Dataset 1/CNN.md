# Library Imports

```py
import numpy as np
import pandas as pd 
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
import librosa.display
import os
import librosa as lb
import soundfile as sf
from datetime import datetime
from os import listdir
from os.path import isfile, join
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from google.colab import drive

drive.mount('/content/drive')
```

# Text Pre Processing

```py
# We have Patient Ids and Disease info

def getFilenameInfo(file):

    return file.split('_')
    
df = pd.read_csv("/content/drive/MyDrive/FYP/patient_diagnosis.csv", names=['pid', 'disease'])

path = '/content/drive/MyDrive/FYP/Audio/'

files = [s.split('.')[0] for s in os.listdir(path) if '.txt' in s]

files_data = []

for file in files:

    data = pd.read_csv(path + file + '.txt', sep = '\t', names = ['start',  'end',  'Crackles', 'Wheezals'])

    name_data=getFilenameInfo(file)

    data['pid']=name_data[0]

    data['mode']=name_data[-2]

    data['filename']=file

    files_data.append(data)

files_df=pd.concat(files_data)

files_df.reset_index()

files_df.head()
```

```py
# Join both patient_data and files_df

df.pid = df.pid.astype('int32')

files_df.pid = files_df.pid.astype('int32')

data = pd.merge(files_df, df,on='pid')

data.to_csv('Data.csv', index=False)

data.head()
```

# Visualizations

## Count Plot

```py
fig, ax = plt.subplots(figsize=(10, 5))

sns.countplot(df.disease, ax=ax);

plt.xticks(rotation=90)

plt.show()
```

## Scatter Plot

```py
fig, ax = plt.subplots(figsize=(10, 6))

sns.scatterplot(x=(data.end-data.start), y = data.pid, ax=ax, s = 130, color = 'darkblue')

plt.show()
```

## Box Plot

```py
fig, ax = plt.subplots(figsize=(5, 4))

sns.boxplot(y=(data.end-data.start), ax=ax);

plt.show()
```

# Audio Pre Processing

```py
os.makedirs('Processed')

def getPureSample(raw_data,start,end,sr=22050):

    '''
    Takes a numpy array and spilts its using start and end args
    raw_data=numpy array of audio sample
    start=time
    end=time
    sr=sampling_rate
    mode=mono/stereo
    '''
    max_ind = len(raw_data) 
    start_ind = min(int(start * sr), max_ind)
    end_ind = min(int(end * sr), max_ind)
    return raw_data[start_ind: end_ind]

for index, row in data.iterrows():

    print("Index -> ",index)

    print("Data\n",row)

    break
```

```py
i, c = 0, 0

for index, row in data.iterrows():

    maxLen=6

    start=row['start']

    end=row['end']

    filename=row['filename']
       
    if end-start>maxLen:  # If len > maxLen , then change it to maxLen
    
        end=start+maxLen
    
    audio_file_loc = path + filename + '.wav'
    
    if index > 0:
        
        if data.iloc[index-1]['filename']==filename: # Check if more cycles exits for same patient if so then add i to change filename
            
            i += 1

        else:

            i=0

    filename= filename + '_' + str(i) + '.wav'
    
    save_path = 'Processed/' + filename

    c += 1
    
    audioArr,sampleRate=lb.load(audio_file_loc)

    pureSample=getPureSample(audioArr,start,end,sampleRate)
    
    reqLen=6*sampleRate # Pad audio if pureSample len < max_len

    padded_data = lb.util.pad_center(pureSample, reqLen)
    
    sf.write(file=save_path,data=padded_data,samplerate=sampleRate)

print('Total Files Processed : ', c)
```

# Imbalance Data

```py
def extractId(filename):

    return filename.split('_')[0]

path='/content/Processed/'

length=len(os.listdir(path))

index=range(length)

i=0

files_df=pd.DataFrame(index=index,columns=['pid','filename'])

for f in os.listdir(path):

    files_df.iloc[i]['pid']=extractId(f)

    files_df.iloc[i]['filename']=f

    i+=1
    
files_df.pid=files_df.pid.astype('int64')

data=pd.merge(files_df, df, on='pid')
```

## Count Plot

```py
fig, ax = plt.subplots(figsize=(10, 5))

sns.countplot(data.disease, ax=ax);

plt.xticks(rotation=90)

plt.show()
```

# Feature Extraction

```py
path = '/content/drive/MyDrive/FYP/Audio/'

filenames = [f for f in listdir(path) if (isfile(join(path, f)) and f.endswith('.wav'))] 

filepaths = [join(path, f) for f in filenames]

p_id_in_file = []

for name in filenames:
    
    p_id_in_file.append(int(name[:3]))

p_id_in_file = np.array(p_id_in_file) 

max_pad_len = 862 # To make the length of all MFCC equal

def extract_features(file_name):
    """
    This function takes in the path for an audio file as a string, loads it, and returns the MFCC
    of the audio
    """
   
    try:

        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast', duration=20) 

        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

        pad_width = max_pad_len - mfccs.shape[1]

        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
    except Exception as e:

        print("Error encountered while parsing file: ", file_name)
        
        return None 
     
    return mfccs
    
p_diag = pd.read_csv("/content/drive/MyDrive/FYP/patient_diagnosis.csv", header=None)

labels = np.array([p_diag[p_diag[0] == x][1].values[0] for x in p_id_in_file])

features = [] 

for file_name in filepaths: # Iterate through each sound file and extract the features

    data = extract_features(file_name)

    features.append(data)

print('Finished feature extraction from ', len(features), ' files')

features = np.array(features)
```

# MFCC Visualization

```py
plt.figure(figsize=(10, 5))

librosa.display.specshow(features[7], x_axis='time')

plt.colorbar()

plt.title('MFCC')

plt.tight_layout()

plt.show()
```

```py
features = np.array(features)

# Delete the very rare diseases

features1 = np.delete(features, np.where((labels == 'Asthma') | (labels == 'LRTI'))[0], axis=0) 

labels1 = np.delete(labels, np.where((labels == 'Asthma') | (labels == 'LRTI'))[0], axis=0)

unique_elements, counts_elements = np.unique(labels1, return_counts=True)
```

# Bar Plot

```py
y_pos = np.arange(len(unique_elements))

plt.figure(figsize=(10, 4))

plt.bar(unique_elements, counts_elements, align='center', alpha=0.5, color="orange")

plt.xticks(y_pos, unique_elements)

plt.ylabel('Count')

plt.xlabel('Disease')

plt.title('Disease Count in Sound Files (No Asthma or LRTI)')

plt.show()
```

# Train & Test Split

```py
le = LabelEncoder()

i_labels = le.fit_transform(labels1)

oh_labels = to_categorical(i_labels) 

# Add channel dimension for CNN

features1 = np.reshape(features1, (*features1.shape,1)) 

x_train, x_test, y_train, y_test = train_test_split(features1, oh_labels, stratify=oh_labels, test_size=0.2, random_state = 42)
```

# CNN

## Model Construction

```py
num_rows = 40
num_columns = 862 
num_channels = 1

num_labels = oh_labels.shape[1]
filter_size = 2

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=filter_size, input_shape=(num_rows, num_columns, num_channels), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=32, kernel_size=filter_size, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=64, kernel_size=filter_size, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=128, kernel_size=filter_size, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(GlobalAveragePooling2D())

model.add(Dense(num_labels, activation='softmax')) 

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam') 

model.summary()

score = model.evaluate(x_test, y_test, verbose=1)

accuracy = 100*score[1]

print("Pre Training Accuracy : %.4f%%" % accuracy)
```

## Model Training

```py
num_epochs = 250

num_batch_size = 128

callbacks = [
    ModelCheckpoint(filepath='mymodel2_{epoch:02d}.h5',

        # Path where to save the model
        # The two parameters below mean that we will overwrite
        # the current checkpoint if and only if
        # the `val_accuracy` score has improved.

        save_best_only=True,
        monitor='val_accuracy',
        verbose=1)
]

start = datetime.now()

model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), callbacks=callbacks, verbose=1)

duration = datetime.now() - start

print("Training completed in time : ", duration)
```

# Model Evaluation

```py
score = model.evaluate(x_train, y_train, verbose=0)
print("Training Accuracy: ", score[1])

score = model.evaluate(x_test, y_test, verbose=0)
print("Testing Accuracy: ", score[1])
```

```py
preds = model.predict(x_test) # label scores 

classpreds = np.argmax(preds, axis=1) # predicted classes 

y_testclass = np.argmax(y_test, axis=1) # true classes

n_classes=6 

fpr = dict()

tpr = dict()

roc_auc = dict()

for i in range(n_classes):

    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], preds[:, i])

    roc_auc[i] = auc(fpr[i], tpr[i])

c_names = ['Bronchiectasis', 'Bronchiolitis', 'COPD', 'Healthy', 'Pneumonia', 'URTI']

print(classification_report(y_testclass, classpreds, target_names=c_names))

print(confusion_matrix(y_testclass, classpreds))
```

## ROC

```py
fig, ax = plt.subplots(figsize=(13, 7))

ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])

ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve for Each Class')

for i in range(n_classes):

    ax.plot(fpr[i], tpr[i], linewidth=3, label='ROC curve (area = %0.2f) for %s' % (roc_auc[i], c_names[i]))

ax.legend(loc="best", fontsize='x-large')

ax.grid(alpha=.4)

sns.despine()

plt.show()
```
