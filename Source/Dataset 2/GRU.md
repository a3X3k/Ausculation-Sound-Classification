# Importing Libraries

```py
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras.utils import np_utils
from keras.layers import add, Conv2D,Input,BatchNormalization,TimeDistributed,Embedding,LSTM,GRU,Dense,MaxPooling1D,Dropout,LeakyReLU,ReLU,Flatten,concatenate,Bidirectional
from tensorflow.keras.layers import concatenate
from keras.models import Model,load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.utils import class_weight
from keras.regularizers import l1,l2
from tensorflow.keras.optimizers import Adam, SGD, Adamax, Adadelta
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, Normalizer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef,cohen_kappa_score,roc_auc_score,confusion_matrix,classification_report
from keras import backend as K
from seaborn import countplot
from tqdm import tqdm

import numpy as np
import pandas as pd
import os
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow
import plotly.graph_objects as ply_go
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
```

# Import Dataset

```py
from google.colab import drive

drive.mount('/content/drive')
```

```py
data = pd.read_csv("/content/drive/MyDrive/FYP/patient_diagnosis.csv", names=['patient_id', 'disease'])

data.head()
```

```py
data['disease'].value_counts()

df_diff = pd.read_csv('/content/drive/MyDrive/FYP/filename_differences.txt', sep = " ", header=None, names = ['file_names'])

df_diff.head(10)
```

```py
patient_data = pd.read_csv("/content/drive/MyDrive/FYP/demographic_info.txt", sep = " ", header=None, names=['Patient_ID', 'Age', 'Sex', 'BMI', 'Weight', 'Height'])

diagnosis_data = pd.read_csv("/content/drive/MyDrive/FYP/patient_diagnosis.csv", header=None, names=['Patient_ID', 'Diagnosis'])

patient_data = patient_data.merge(diagnosis_data, on='Patient_ID')

print(patient_data.shape)
```

# Pre - Processing

## Imputing Missing Values

```py
print(patient_data.isna().sum() )
```

## Impute Missing Age

```py
patient_data[patient_data['Age'].isnull()]

# Only One entry without age.

# Diagnosis of that Patient is COPD

# Use median of Relevant Population who has COPD

patient_data.loc[patient_data['Age'].isnull(), 'Age'] = patient_data.loc[patient_data['Diagnosis'] == 'COPD',  'Age'].median()
```

## Impute Missing Sex

```py
patient_data[patient_data['Sex'].isnull()]

# Count how many M/F are there with COPD

patient_data.loc[patient_data['Diagnosis'] == 'COPD', ['Sex', 'Patient_ID'] ].groupby(['Sex']).count() 

# Only One entry without Sex 

# Replace with Most Common Outcome - Here it's Male

patient_data.loc[patient_data['Sex'].isnull(), 'Sex'] = 'M' 
```

## Impute Missing BMI

```py
patient_data[patient_data['BMI'].isnull()]

null_bmi = patient_data['BMI'].isnull()

patient_data.loc[null_bmi, 'BMI'] = patient_data.loc[null_bmi, 'Weight'] / (patient_data.loc[null_bmi,'Height']/100)**2
```

## Summary of missing values after Imputation

```py
age_quantiles = patient_data['Age'].quantile([0.2, 0.4, 0.6, 0.8]).values

patient_data['Age_Category'] = 'E'

patient_data.loc[ patient_data['Age'] < age_quantiles[0], 'Age_Category'] = 'A'

patient_data.loc[ patient_data['Age'] < age_quantiles[1], 'Age_Category'] = 'B'

patient_data.loc[ patient_data['Age'] < age_quantiles[2], 'Age_Category'] = 'C'

patient_data.loc[ patient_data['Age'] < age_quantiles[3], 'Age_Category'] = 'D'

tmp_data = (patient_data.loc[~patient_data['BMI'].isnull(), ['Diagnosis', 'Age_Category', 'Sex', 'BMI']]
            .groupby(['Diagnosis', 'Age_Category', 'Sex'])
            .agg('median')
            .reset_index()
            .rename(columns={'BMI' : 'BMI_imputed'}) )
            
patient_data_imputed = patient_data.loc[patient_data['BMI'].isnull(),].merge(tmp_data, on = ['Diagnosis', 'Age_Category', 'Sex'], how='left')

patient_data = patient_data.merge(patient_data_imputed[['Patient_ID', 'BMI_imputed']], on = ['Patient_ID'], how='left')

patient_data.loc[patient_data['BMI'].isnull(), 'BMI'] = patient_data.loc[patient_data['BMI'].isnull(), 'BMI_imputed']

tmp_data = (patient_data.loc[~patient_data['BMI'].isnull(),].
            groupby(['Age_Category']).agg('median').reset_index().rename(columns = {'BMI':'BMI_imputed2'}) )

patient_data_imputed = patient_data.loc[patient_data['BMI'].isnull(),].merge(tmp_data[['Age_Category', 'BMI_imputed2']], 
                                                                             on=['Age_Category'], how='left')

patient_data = patient_data.merge(patient_data_imputed[['Patient_ID', 'BMI_imputed2']], on=['Patient_ID'],how='left')

patient_data.loc[patient_data['BMI_imputed'].isnull(), 'BMI_imputed'] = patient_data.loc[patient_data['BMI_imputed'].isnull(), 'BMI_imputed2']

patient_data.loc[patient_data['BMI'].isnull(), 'BMI'] = patient_data.loc[patient_data['BMI'].isnull(), 'BMI_imputed']

patient_data.drop(['BMI_imputed2'],1,inplace=True)

patient_data['BMI_imputed'] = ~patient_data['BMI_imputed'].isnull()

print(patient_data.isna().sum())
```

## Visualization Of Distributions

```py
my_title_layout = dict({"text" : "my Distribution", 'xanchor' : 'center', 'x' : 0.5, 'y' : 0.9, 'font' : {'size' : 30}})

my_xaxis_layout = dict(title = dict (text="my x axis", font={'size':20}))

my_layout = dict(title = my_title_layout, xaxis = my_xaxis_layout)

bin_size_dict = dict(Age = 2, BMI = 3, Diagnosis = 1, Sex = 1)

xaxis_title_dict = dict(Age = "Age in Years", BMI = "BMI", Diagnosis = "Condition", Sex = "Male/Female")
```

```py
hist_data = ply_go.Histogram(x = patient_data['Age'], name = 'Age', showlegend = True, xbins = {'size' : bin_size_dict['Age']})

fig = ply_go.Figure(data = [hist_data], layout = my_layout)

fig.update_layout(title={'text': "Age " + "Distribution"}, xaxis = {"title": {"text" : xaxis_title_dict['Age']}})

fig.show()
```

```py
hist_data = ply_go.Histogram(x = patient_data['Sex'], name = 'Sex', showlegend = True, xbins = {'size' : bin_size_dict['Sex']})

fig = ply_go.Figure(data = [hist_data], layout = my_layout)

fig.update_layout (title = {'text' : "Sex" + " Distribution"}, xaxis = {"title" : {"text" : xaxis_title_dict['Sex']}})

fig.show()
```

```py
hist_data = ply_go.Histogram(x = patient_data['BMI'], name = 'BMI', showlegend = True, xbins = {'size' : bin_size_dict['BMI']})

fig = ply_go.Figure (data = [hist_data], layout = my_layout)

fig.update_layout (title = {'text' : "BMI" + " Distribution"}, xaxis = {"title" : {"text" : xaxis_title_dict['BMI']}})

fig.show()
```

```py
hist_data = ply_go.Histogram(x = patient_data['Diagnosis'], name = 'Diagnosis', showlegend = True, xbins = {'size' : bin_size_dict['Diagnosis']})

fig = ply_go.Figure(data = [hist_data], layout = my_layout)

fig.update_layout(title = {'text' : "Diagnosis Distribution"}, xaxis = {"title" : {"text" : xaxis_title_dict['Diagnosis']}})

fig.show()
```

```py
fig.update_layout(title = {'text' : "Distribution of Age by type of Diagnosis"}, 
                  xaxis = {"title" : {"text" : None}}, 
                  yaxis = {"title" : {"text" : "Age in Years"}})

fig.show()

fig = ply_go.Figure( layout = my_layout)

for tmp_diag in patient_data['Diagnosis'].unique():

    violin_data = ply_go.Violin(x = patient_data.loc[patient_data['Diagnosis'] == tmp_diag, 'Diagnosis'],
                                y = patient_data.loc[patient_data['Diagnosis'] == tmp_diag, 'BMI'], 
                                name = tmp_diag, 
                                box_visible = True,
                                meanline_visible = True)
    
    fig.add_trace(violin_data)

fig.update_layout(title = {'text' : "Distribution of BMI by type of Diagnosis"}, 
                  xaxis = {"title" : {"text" : None}}, 
                  yaxis = {"title" : {"text" : "BMI"}})

fig.show()
```

```py
patient_data.loc[patient_data['Diagnosis']=='COPD']
```

# Path Setting

```py
path=[]

disease=[]

for soundDir in (os.listdir('/content/drive/MyDrive/Dataset/Kaggle/Audio/')):

        if soundDir[-3:] == 'wav' and soundDir[:3] != '103' and soundDir[:3] != '108' and soundDir[:3] != '115':

            p = list(data[data['patient_id'] == int(soundDir[:3])]['disease'])[0]

            path.append('/content/drive/MyDrive/Dataset/Kaggle/Audio/' + soundDir)

            disease.append(p)

df = pd.DataFrame({'path':path,'label':disease})

df1 = pd.read_csv('/content/drive/MyDrive/Dataset/AIMS/Details.csv')

df2=df1[['SI No', 'Diagnosis']]

df2.rename(columns={'SI No':'path', 'Diagnosis':'label'}, inplace=True)

for i in range(len(df2)):

  k = (df2['path'][i])

  df2['path'][i] = '/content/drive/MyDrive/Dataset/AIMS/Audio/' + str(k) + ".wav"

  df3 = pd.concat([df, df2], axis = 0)

  df3['label'].value_counts()
  
ind = df3[(df3['label'] == 'Cancer') | (df3['label'] == 'Tuberculosis')].index

df3.drop(ind, inplace=True)

df3['label'].value_counts()

df3.to_csv('/content/drive/MyDrive/Dataset/Data.csv', index='false')

df3 = pd.read_csv('/content/drive/MyDrive/Dataset/Data.csv', index_col=0)
```

# Count Plot

```py
from seaborn import countplot

plt.figure(figsize=(13, 5));

ax = countplot(df3['label']);

plt.show();
```

## Data Augmentation Before Augmentation

```py
plt.figure(figsize=(13, 5));

ax = countplot(patient_data['Diagnosis']);

plt.show();
```

```py
def add_noise(data,x):

    noise = np.random.randn(len(data))

    data_noise = data + x * noise

    return data_noise


def shift(data,x):
    
    return np.roll(data, x)
    

def stretch(data, rate):

    data = librosa.effects.time_stretch(data, rate)
    
    return data   
```

```py
X_ = []

y_ = []

count=0

count1 = 0

for i in tqdm(range(len(df3))):

  if df3['label'].iloc[i] == 'COPD' and count < 200:

    count += 1

    data_x, sampling_rate = librosa.load(df3['path'].iloc[i],res_type='kaiser_fast')

    mfccs = np.mean(librosa.feature.mfcc(y = data_x, sr = sampling_rate, n_mfcc = 40).T, axis = 0)

    X_.append(mfccs)

    y_.append(df3['label'].iloc[i])

  elif df3['label'].iloc[i] != 'COPD':

    data_x, sampling_rate = librosa.load(df3['path'].iloc[i], res_type = 'kaiser_fast')

    mfccs = np.mean(librosa.feature.mfcc(y = data_x, sr = sampling_rate, n_mfcc = 40).T, axis = 0) 
    X_.append(mfccs)
    y_.append(df3['label'].iloc[i])
    data_noise = add_noise(data_x, 0.005)
    mfccs_noise = np.mean(librosa.feature.mfcc(y=data_noise, sr=sampling_rate, n_mfcc=40).T,axis=0) 
    X_.append(mfccs_noise)
    y_.append(df3['label'].iloc[i])

    data_shift = shift(data_x, 1600)
    mfccs_shift = np.mean(librosa.feature.mfcc(y=data_shift, sr=sampling_rate, n_mfcc=40).T,axis=0) 
    X_.append(mfccs_shift)
    y_.append(df3['label'].iloc[i])

    data_stretch = stretch(data_x, 1.2)
    mfccs_stretch = np.mean(librosa.feature.mfcc(y=data_stretch, sr=sampling_rate, n_mfcc=40).T,axis=0) 
    X_.append(mfccs_stretch)
    y_.append(df3['label'].iloc[i])
                
    data_stretch_2 = stretch(data_x, 0.8)
    mfccs_stretch_2 = np.mean(librosa.feature.mfcc(y=data_stretch_2, sr=sampling_rate, n_mfcc=40).T,axis=0) 
    X_.append(mfccs_stretch_2)
    y_.append(df3['label'].iloc[i])
    
X=np.array(X_)

Y=np.array(y_)
```

# Count Plot After Augmentation

```py
plt.figure(figsize=(13, 5));

ax = countplot(Y);

plt.show();
```

# Modelling

```py
lb = LabelEncoder()

y_integer = lb.fit_transform(Y)

Y = np_utils.to_categorical(y_integer)

X = X.reshape(X.shape[0],1,X.shape[1])

Y = Y.reshape(Y.shape[0],1,Y.shape[1])

K.clear_session()

batch_size=X.shape[0]

time_steps=X.shape[1]

data_dim=X.shape[2]

def InstantiateModel(in_):

    model_2_1 = GRU(32,return_sequences=True,activation=None,go_backwards=True)(in_)
    model_2 = LeakyReLU()(model_2_1)
    model_2 = GRU(128,return_sequences=True, activation=None,go_backwards=True)(model_2)
    model_2 = LeakyReLU()(model_2)
    
    model_3 = GRU(64,return_sequences=True,activation=None,go_backwards=True)(in_)
    model_3 = LeakyReLU()(model_3)
    model_3 = GRU(128,return_sequences=True, activation=None,go_backwards=True)(model_3)
    model_3 = LeakyReLU()(model_3)
    
    model_add_1 = add([model_3,model_2])
    
    model_5 = GRU(128,return_sequences=True,activation=None,go_backwards=True)(model_add_1)
    model_5 = LeakyReLU()(model_5)
    model_5 = GRU(32,return_sequences=True, activation=None,go_backwards=True)(model_5)
    model_5 = LeakyReLU()(model_5)
    
    model_6 = GRU(64,return_sequences=True,activation=None,go_backwards=True)(model_add_1)
    model_6 = LeakyReLU()(model_6)
    model_6 = GRU(32,return_sequences=True, activation=None,go_backwards=True)(model_6)
    model_6 = LeakyReLU()(model_6)
    
    model_add_2 = add([model_5,model_6,model_2_1])

    model_7 = Dense(64, activation=None)(model_add_2)
    model_7 = LeakyReLU()(model_7)
    model_7 = Dropout(0.2)(model_7)
    model_7 = Dense(16, activation=None)(model_7)
    model_7 = LeakyReLU()(model_7)
    
    model_9 = Dense(32, activation=None)(model_add_2)
    model_9 = LeakyReLU()(model_9)
    model_9 = Dropout(0.2)(model_9)
    model_9 = Dense(16, activation=None)(model_9)
    model_9 = LeakyReLU()(model_9)
    
    model_add_3 = add([model_7,model_9])

    model_10 = Dense(16, activation=None)(model_add_3)
    model_10 = LeakyReLU()(model_10)
    model_10 = Dropout(0.5)(model_10)
    model_10 = Dense(6, activation="softmax")(model_10)

    return model_10
    
Input_Sample = Input(shape=(time_steps,data_dim))

Output_ = InstantiateModel(Input_Sample)

Model_Enhancer = Model(inputs=Input_Sample, outputs=Output_)
```

# Train & Test Splitting

```py
# Split the dataset 

from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state = 42)

Model_Enhancer.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adamax())

Model_Enhancer.summary()
```

```py
ES = EarlyStopping(monitor = 'val_loss', 
                   min_delta = 0.5, 
                   patience = 200, 
                   verbose = 1, 
                   mode = 'auto', 
                   baseline = None, 
                   restore_best_weights = False)

MC = ModelCheckpoint('Model 1.h5', monitor = 'val_accuracy', mode = 'auto', verbose = 0, save_best_only = True)

num_epochs =600

num_batch_size = 20

ModelHistory = Model_Enhancer.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), callbacks = [MC, ES], verbose=1)
```

# Model Evaluation

## Loss Curves

```py
# Loss Curves

plt.figure(figsize=[20, 7]);

plt.plot(ModelHistory.history['loss'], 'r');

plt.plot(ModelHistory.history['val_loss'], 'b');

plt.legend(['Training Loss', 'Validation Loss'], fontsize=20);

plt.xlabel('Epochs', fontsize=20);

plt.ylabel('Loss', fontsize=20);

plt.title('Loss Curves', fontsize=20);
```

## Accuracy Curves

```py
# Accuracy Curves

plt.figure(figsize=[20, 7]);

plt.plot(ModelHistory.history['accuracy'], 'r');

plt.plot(ModelHistory.history['val_accuracy'], 'b');

plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=20);

plt.xlabel('Epochs', fontsize=20);

plt.ylabel('Accuracy', fontsize=20);

plt.title('Accuracy Curves', fontsize=20);
```

## Train & Test Score of Best Model

```py
Model_Loaded = load_model('Model 1.h5')

score = Model_Loaded.evaluate(x_train, y_train, verbose=0)

print("Training Accuracy: ", score[1])

score = Model_Loaded.evaluate(x_test, y_test, verbose=0)

print("Testing Accuracy: ", score[1])
```

## Train & Test Score of Final Model

```py
score = Model_Enhancer.evaluate(x_train, y_train, verbose=0)

print("Training Accuracy: ", score[1])

score = Model_Enhancer.evaluate(x_test, y_test, verbose=0)

print("Testing Accuracy: ", score[1])
```

## Accuracy, Precision, Recall, ROC & AUC, Classification Report & Confusion Matrix

```py
yhat_probs = Model_Loaded.predict(x_test, verbose=1)

yhat_probs = yhat_probs.reshape(yhat_probs.shape[0], yhat_probs.shape[2])

yhat_classes =np.argmax(yhat_probs, axis=1)

testy = y_test.reshape(y_test.shape[0], y_test.shape[2])

testy =np.argmax(testy, axis=1)

accuracy = accuracy_score(testy, yhat_classes)
print('Accuracy: %f' % accuracy)

precision = precision_score(testy, yhat_classes, average='weighted')
print('Precision: %f' % precision)

recall = recall_score(testy, yhat_classes, average='weighted')
print('Recall: %f' % recall)

f1 = f1_score(testy, yhat_classes, average='weighted')
print('F1 score: %f' % f1)

kappa = cohen_kappa_score(testy, yhat_classes)
print('Cohens kappa: %f' % kappa)

MatthewsCorrCoef = matthews_corrcoef(testy, yhat_classes)
print('Matthews correlation coefficient: %f' % MatthewsCorrCoef)

auc = roc_auc_score(testy, yhat_probs, multi_class='ovo')
print('ROC AUC : %f' % auc)

c_names = ['Bronchiectasis', 'Bronchiolitis', 'COPD', 'Healthy', 'Pneumonia', 'URTI']
print(classification_report(testy, yhat_classes, target_names=c_names))

print(confusion_matrix(testy, yhat_classes))
```
