
# coding: utf-8

# In[ ]:

#importing python libraries for reading data


# In[ ]:

import pandas as pd
dataframe = pd.read_csv("Newdatasets.csv")
dataframe[dataframe.TARGET=='Eligible']


# In[ ]:

#Grouping datasets by target


# In[ ]:

dataframe.groupby('TARGET').size()


# In[ ]:

#Visualization of the data using pairplot 


# In[ ]:

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
get_ipython().magic(u'matplotlib notebook')
plt.style.available
#plt.style.use('seaborn-colorblind')
dataframe = pd.read_csv("Dataset.csv")
sns.pairplot(dataframe, hue='TARGET')
#sns.pairplot(dataframe, hue='TARGET', palette="Set2", diag_kind="kde", size=2.5)


# In[ ]:

# Visualization using Andrew curve


# In[ ]:

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
get_ipython().magic(u'matplotlib notebook')
plt.style.available
plt.style.use('seaborn-colorblind')
dataframe = pd.read_csv("Newdatasets.csv")
from pandas.tools.plotting import andrews_curves
andrews_curves(dataframe, 'TARGET')


# In[ ]:

#Data training for the model


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
# fix random seed for reproducibility
seed = 5
numpy.random.seed(seed)
# load dataset
dataframe = pandas.read_csv("Newdataset.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:6].astype(float)
Y = dataset[:,6]

# hot encoding 
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = np_utils.to_categorical(encoded_Y)

# model definition
model = Sequential()
model.add(Dense(35, input_dim=6, init='normal', activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(18, activation='sigmoid'))
#model.add(Dropout(0.1))
model.add(Dense(2, activation='softmax'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model fit
model.fit(X, dummy_y, nb_epoch=50, batch_size=5)


# In[ ]:

#Model accuracy value


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
# fix random seed for reproducibility
seed = 5
numpy.random.seed(seed)
# load dataset
dataframe = pandas.read_csv("Newdataset.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:6].astype(float)
Y = dataset[:,6]

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
    model.add(Dropout(0.1))
    model.add(Dense(20, activation='sigmoid'))
    model.add(Dropout(0.1))
    model.add(Dense(2, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model
estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=50, batch_size=5, verbose=0)
kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print(results)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# In[ ]:

#Model summary


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
dataframe = pandas.read_csv("Datasets.csv", header=None)
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
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(20,  init = 'normal'))
model.add(Activation('sigmoid'))
model.add(Dense(2, init='normal'))
model.add(Activation('softmax'))
model.summary()


# In[ ]:

#Model graphical representation


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
# fix random seed for reproducibility
seed = 5
numpy.random.seed(seed)
# load pima indians dataset
dataframe = pd.read_csv("Newdataset.csv", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:6]
Y = dataset[:,6]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
# create model
model = Sequential()
model.add(Dense(35, input_dim=6, init='normal', activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(18, activation='sigmoid'))
model.add(Dropout(0.1))
model.add(Dense(2, activation='softmax'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
# Fit the model
history = model.fit(X, dummy_y, validation_split=0.19, epochs=50, batch_size=5, verbose=0)
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

# Model Parameters and predictions


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

# fix random seed for reproducibility
seed = 5
numpy.random.seed(seed)
# load pima indians dataset
dataframe = pandas.read_csv("Newdataset.csv", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:6]
Y = dataset[:,6]
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
    model.add(Dense(40, input_dim=6, init='normal', activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(20, activation='sigmoid'))
    model.add(Dropout(0.1))
    model.add(Dense(2, activation='softmax'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
# Model training
estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=50, batch_size=5, verbose=0)
X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.13, random_state=seed)
estimator.fit(X_train, Y_train)
predictions = estimator.predict(X_test)

print('========== Encoded Prediction Codes ======')
print(predictions)
print('===== Corresponding Meaning of Codes======')
print(encoder.inverse_transform(predictions))
print('                                  ')
print('-------- Summary of Datasets uded ------------')
print('Training dataset used out of 173,6:', X_train.shape)
print('Labels used corresponding to the trained sets',Y_train.shape)
print('Test datasets used out of 173,6:', X_test.shape)
print('Labels used for testing', Y_test.shape)


# In[ ]:

#Saving the model


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

# fix random seed for reproducibility
seed = 5
numpy.random.seed(seed)
# load pima indians dataset
dataframe = pandas.read_csv("Newdataset.csv", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:6]
Y = dataset[:,6]
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
    model.add(Dense(40, input_dim=6, init='normal', activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(20, activation='sigmoid'))
    model.add(Dropout(0.1))
    model.add(Dense(2, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
# Model training
estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=50, batch_size=5, verbose=1)
X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.13, random_state=seed)
estimator.fit(X_train, Y_train)
predictions = estimator.predict(X_test)

model.save('my_model2.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model
print("Saved model to disk")


# In[ ]:

from keras.models import load_model

model = load_model('my_model2.h5')
X = [315000,1000000,417272.8,90000,473627.2,62127.64]
X_test = np.expand_dims(X, axis=0)
yhat = model.predict_classes(X_test, verbose =0)
print(yhat)
print(encoder.inverse_transform(yhat))


# In[ ]:



