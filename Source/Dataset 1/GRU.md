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

import numpy as np
import pandas as pd
import os
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow
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

```py

```

```py

```

```py

```

```py

```

```py

```

```py

```

```py

```

```py

```

```py

```

















