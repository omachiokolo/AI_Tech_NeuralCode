
# coding: utf-8

# Prediction System Protoptype for Business Viable Software Projects using Artificial Neural Network
#                      ==============Datasets Frame Layout Display==========================

# In[ ]:

import pandas as pd
dataframe = pd.read_csv("newdata.csv")
dataframe[dataframe.Class=='MIV']


# ===========Displaying the Classes of the Datasets=============

# In[ ]:

dataframe.groupby('Class').size()


# ============== Displaying data pattern ==============

# In[ ]:

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
get_ipython().magic(u'matplotlib notebook')
plt.style.available
#plt.style.use('seaborn-colorblind')
dataframe = pd.read_csv("newdata.csv")
#sns.pairplot(dataframe, hue='Class')
sns.pairplot(dataframe, hue='Class', palette="Set2", diag_kind="kde", size=2.5)


# In[ ]:

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
get_ipython().magic(u'matplotlib notebook')
plt.style.available
plt.style.use('seaborn-colorblind')
dataframe = pd.read_csv("newdata.csv")
from pandas.tools.plotting import andrews_curves
andrews_curves(dataframe, 'Class')


# ============Displaying the Structure of the final Model===========

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
dataframe = pandas.read_csv("projdata.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]

# hot encoding 
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = np_utils.to_categorical(encoded_Y)
#Model definition
model = Sequential()
model.add(Dense(27, input_dim = 4, init = 'normal'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(3, init='normal'))
model.add(Activation('softmax'))
model.summary()


# =========== Model Training =====================

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
# fix random seed for reproducibility
seed = 5
numpy.random.seed(seed)
# load dataset
dataframe = pandas.read_csv("projdata.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]

# hot encoding 
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = np_utils.to_categorical(encoded_Y)

# model definition
model = Sequential()
model.add(Dense(27, input_dim=4, init='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model fit
model.fit(X, dummy_y, nb_epoch=50, batch_size=5)


# =============== Visualize the Trainng History ================

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
dataframe = pd.read_csv("projdata.csv", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:4]
Y = dataset[:,4]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
# create model
model = Sequential()
model.add(Dense(27, input_dim=4, kernel_initializer='uniform', activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(3, kernel_initializer='uniform', activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
# Fit the model
history = model.fit(X, dummy_y, validation_split=0.33, epochs=50, batch_size=5, verbose=0)
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

=============== Model Evaluation ===============


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
dataframe = pandas.read_csv("projdata.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]

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
    model.add(Dense(27, input_dim=4, init='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=50, batch_size=5, verbose=0)
kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print(results)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# =============== Making Prediction witth the Model===============

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
dataframe = pandas.read_csv("projdata.csv", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:4]
Y = dataset[:,4]
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
    model.add(Dense(27, input_dim=4, init='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
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
print('Training dataset used out of 75,4:', X_train.shape)
print('Labels used corresponding to the trained sets',Y_train.shape)
print('Test datasets used out of 75,4:', X_test.shape)
print('Labels used for testing', Y_test.shape)


# ============ Saving Model for Transfer Learning ============

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
dataframe = pandas.read_csv("projdata.csv", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:4]
Y = dataset[:,4]
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
    model.add(Dense(27, input_dim=4, init='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))
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


# ============= Transfer Learning/Importing the Saved Model to Make Prediction on Unseen data =============

# In[ ]:

from keras.models import load_model

model = load_model('my_model2.h5')
X = [12,9,8,8]
X_test = np.expand_dims(X, axis=0)
yhat = model.predict_classes(X_test, verbose =0)
print(yhat)
print(encoder.inverse_transform(yhat))


# In[ ]:



