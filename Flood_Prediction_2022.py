
# coding: utf-8

# # Flood Prediction

# In[ ]:

import pandas as pd
dataframe = pd.read_csv("floodata.csv")
#dataframe[dataframe.Class=='flood']
dataframe


# In[ ]:

#Pearson correlation in pandas
dataframe.corr(method ='pearson')


# # Showing the Distinct Class size of the dataset

# In[ ]:

dataframe.groupby('Class').size()


# # Model Training 

# In[ ]:

import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
import numpy as np
import pandas as pd
get_ipython().magic(u'matplotlib inline')
import csv
import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

# create scaler
scaler = MinMaxScaler(feature_range=(-1,1))
# fix random seed for reproducibility


# fix random seed for reproducibility
seed = 5
numpy.random.seed(seed)
# load dataset
dataframe = pandas.read_csv("flooddata.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:6]
Y = dataset[:,6]

# create scaler
scaler = MinMaxScaler()

# fit scaler on data
scaler.fit(X)

# transform training dataset
trainx = scaler.transform(X)


# hot encoding 
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = np_utils.to_categorical(encoded_Y)

# model definition
model = Sequential()
model.add(Dense(30, input_dim=6, init='normal', activation='relu'))
model.add(Dropout(0.01))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.01))
model.add(Dense(2, activation='softmax'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# model fit
model.fit(trainx, dummy_y, nb_epoch=50, batch_size=5)


# # Model training Visualization

# In[ ]:

import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
import numpy as np
import pandas as pd
get_ipython().magic(u'matplotlib inline')
import csv
import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

# create scaler
scaler = MinMaxScaler(feature_range=(-1,1))
# fix random seed for reproducibility


# fix random seed for reproducibility
seed = 5
numpy.random.seed(seed)
# load dataset
dataframe = pandas.read_csv("flooddata.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:6].astype(float)
Y = dataset[:,6]

# create scaler
scaler = MinMaxScaler()

# fit scaler on data
scaler.fit(X)

# transform training dataset
trainx = scaler.transform(X)

# hot encoding 
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = np_utils.to_categorical(encoded_Y)

# model definition
model = Sequential()
model.add(Dense(45, input_dim=6, init='normal', activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(2, activation='softmax'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

history = model.fit(trainx, dummy_y, epochs=150, batch_size=10, verbose=0)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['loss'])
plt.title('model accuracy/Loss')
#plt.ylabel('accuracy/loss')
plt.ylabel('loss/Accuracy')
plt.xlabel('epoch')
plt.legend(['Accuracy', 'Loss'], loc='upper right')
plt.show()


# # Data Normalization

# In[ ]:

import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

# create scaler
scaler = MinMaxScaler(feature_range=(-1,1))
# fix random seed for reproducibility

seed = 5
numpy.random.seed(seed)
# load dataset
dataframe = pandas.read_csv("flooddata.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:6]
Y = dataset[:,6]

# create scaler
scaler = MinMaxScaler()

# fit scaler on data
scaler.fit(X)

# transform training dataset
trainx = scaler.transform(X)

# transform test

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

print("The normalized X variables ranging between 0 and 1 are: ")
print("")
print(trainx)
print("")
print("The hard-encoded Y or predicted variables are:")
print(dummy_y)


# # Model Tunning and Performance evaluation

# In[ ]:

import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

# create scaler
scaler = MinMaxScaler(feature_range=(-1,1))
# fix random seed for reproducibility

seed = 5
numpy.random.seed(seed)
# load dataset
dataframe = pandas.read_csv("flooddata.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:6]
Y = dataset[:,6]

# create scaler
scaler = MinMaxScaler()

# fit scaler on data
scaler.fit(X)

# transform training dataset
trainx = scaler.transform(X)

# transform test

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(30, input_dim=6, init='normal', activation='relu'))
    model.add(Dropout(0.01))
    model.add(Dense(20, activation='relu'))
    model.add(Dropout(0.01))
    model.add(Dense(2, activation='softmax'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=50, batch_size=5, verbose=0)
kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
results = cross_val_score(estimator, trainx, dummy_y, cv=kfold)
print(results)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# # Model Architecture Summary

# In[ ]:

import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
import numpy as np
import pandas as pd
get_ipython().magic(u'matplotlib inline')
import csv
import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
# fix random seed for reproducibility
seed = 5
numpy.random.seed(seed)
# load dataset
dataframe = pandas.read_csv("flooddata.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:6].astype(float)
Y = dataset[:,6]

# hot encoding 
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = np_utils.to_categorical(encoded_Y)
#Model definition
model = Sequential()
model.add(Dense(30, input_dim = 6, init = 'normal'))
model.add(Dropout(0.01))
model.add(Dense(25, init = 'normal'))
model.add(Activation('relu'))
model.add(Dropout(0.01))
model.add(Dense(2, init='normal'))
model.add(Activation('softmax'))
model.summary()


# # Making Prediction

# In[ ]:

import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
import numpy as np
import pandas as pd
get_ipython().magic(u'matplotlib inline')
import csv
import numpy
from keras import backend as K
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

# create scaler
scaler = MinMaxScaler(feature_range=(-1,1))

# fix random seed for reproducibility
seed = 5
numpy.random.seed(seed)
# load pima indians dataset
dataframe = pandas.read_csv("flooddata.csv", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:6]
Y = dataset[:,6]

# create scaler
scaler = MinMaxScaler()

# fit scaler on data
scaler.fit(X)

# transform training dataset
trainx = scaler.transform(X)


# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
# create model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(30, input_dim=6, init='normal', activation='relu'))
    model.add(Dropout(0.01))
    model.add(Dense(20, activation='relu'))
    model.add(Dropout(0.01))
    model.add(Dense(2, activation='softmax'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
# Model training
estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=50, batch_size=5, verbose=0)
X_train, X_test, Y_train, Y_test = train_test_split(trainx, dummy_y, test_size=0.3, random_state=seed)
estimator.fit(X_train, Y_train)
predictions = estimator.predict(X_test)


print('========== ....................... ======')

print('========== Encoded Prediction Codes ======')
print(predictions)
print('===== Corresponding Meaning of Codes======')
print(encoder.inverse_transform(predictions))
print('                                  ')
print('-------- Summary of Datasets uded ------------')
print('Training dataset used out of 360,6:', X_train.shape)
print('Labels used corresponding to the trained sets',Y_train.shape)
print('Test datasets used out of 360,6:', X_test.shape)
print('Labels used for testing', Y_test.shape)


# In[ ]:

import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
import numpy as np
import pandas as pd
get_ipython().magic(u'matplotlib inline')
import csv
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

# create scaler
scaler = MinMaxScaler(feature_range=(-1,1))
# fix random seed for reproducibility
seed = 5
numpy.random.seed(seed)
# load dataset
dataframe = pandas.read_csv("flooddata.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:6].astype(float)
Y = dataset[:,6]

# create scaler
scaler = MinMaxScaler()

# fit scaler on data
scaler.fit(X)

# transform training dataset
trainx = scaler.transform(X)

# hot encoding 
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = np_utils.to_categorical(encoded_Y)

# model definition
model = Sequential()
model.add(Dense(30, input_dim=6, init='normal', activation='relu'))
model.add(Dropout(0.01))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.01))
model.add(Dense(2, activation='softmax'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# model fit
#model.fit(X, dummy_y, nb_epoch=50, batch_size=5, verbose=0)

history = model.fit(trainx, dummy_y, epochs=50, batch_size=5, verbose=0)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['loss'])
plt.title('model accuracy/Loss')
#plt.ylabel('accuracy/loss')
plt.ylabel('loss/Accuracy')
plt.xlabel('epoch')
plt.legend(['Accuracy', 'Loss'], loc='upper right')
plt.show()


# # Confusion Matrix Evaluation, Precision and Recall

# In[ ]:

# Confusion Matrix Trial and error
from sklearn.metrics import classification_report,confusion_matrix
#from sklearn.metrics import plot_confusion_matrix
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
import numpy as np
import pandas as pd
get_ipython().magic(u'matplotlib inline')
import csv
import numpy
from keras import backend as K
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

# create scaler
scaler = MinMaxScaler(feature_range=(-1,1))

# fix random seed for reproducibility
seed = 5
numpy.random.seed(seed)
# load pima indians dataset
dataframe = pandas.read_csv("flooddata.csv", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:6]
Y = dataset[:,6]

# create scaler
scaler = MinMaxScaler()

# fit scaler on data
scaler.fit(X)

# transform training dataset
trainx = scaler.transform(X)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
# create model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(30, input_dim=6, init='normal', activation='relu'))
    model.add(Dropout(0.01))
    model.add(Dense(20, activation='relu'))
    model.add(Dropout(0.01))
    model.add(Dense(2, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
# Model training
estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=50, batch_size=5, verbose=0)
X_train, X_test, Y_train, Y_test = train_test_split(trainx, dummy_y, test_size=0.30, random_state=seed)
estimator.fit(X_train, Y_train)
predictions = estimator.predict(X_test)



target_names = ['class 0(Flood)', 'class 1(Normal/No_Flood)']
print(classification_report(np.argmax(Y_test,axis=1), predictions,target_names=target_names))
print('Confusion_Matrix Report')
print(confusion_matrix(np.argmax(Y_test,axis=1), predictions))


# # Model Visualization- Training Dataset and Model loss

# In[ ]:

from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
import pandas as pd
import numpy
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

# create scaler
scaler = MinMaxScaler(feature_range=(-1,1))

# fix random seed for reproducibility
seed = 5
numpy.random.seed(seed)
# load pima indians dataset
dataframe = pd.read_csv("flooddata.csv", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:6]
Y = dataset[:,6]

# create scaler
scaler = MinMaxScaler()

# fit scaler on data
scaler.fit(X)

# transform training dataset
trainx = scaler.transform(X)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
# create model
model = Sequential()
model.add(Dense(40, input_dim=6, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(2, kernel_initializer='uniform', activation='softmax'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
# Fit the model
history = model.fit(trainx, dummy_y, validation_split=0.3, epochs=50, batch_size=5, verbose=0)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_acc'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:



