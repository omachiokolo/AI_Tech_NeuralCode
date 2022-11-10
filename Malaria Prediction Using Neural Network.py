
# coding: utf-8

# Malaria Prediction Set-up Using Artificial Neural Network & Decision Tree =============Datasets Frame Layout Display=====================

# In[ ]:

import pandas as pd
dataframe = pd.read_csv("MalariaFieldData1.csv")
#dataframe[dataframe.Class==0]
dataframe


# ========= Data Preprocessing ======================

# In[ ]:

dataframe['Catarrh'].unique


# ===========Displaying the Classes of the Datasets=============

# In[ ]:

dataframe.groupby('Class').size()


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
dataframe = pandas.read_csv("MalariaFieldData11.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:14].astype(float)
Y = dataset[:,14]

# hot encoding 
#encoder = LabelEncoder()
#encoder.fit(Y)
#encoded_Y = encoder.transform(Y)
#dummy_y = np_utils.to_categorical(encoded_Y)

# model definition
model = Sequential()
model.add(Dense(55, input_dim=14, init='normal', activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(65, activation='sigmoid'))
#model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model fit
model.fit(X, Y, nb_epoch=50, batch_size=10)


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
dataframe = pandas.read_csv("MalariaFieldData11.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:14].astype(float)
Y = dataset[:,14]

# encode class values as integers
#encoder = LabelEncoder()
#encoder.fit(Y)
#encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
#dummy_y = np_utils.to_categorical(encoded_Y)
# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(55, input_dim=14, init='normal', activation='relu'))
    model.add(Dropout(0.01))
    model.add(Dense(65, activation='sigmoid'))
    #model.add(Dropout(0.01))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=50, batch_size=5, verbose=0)
kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=kfold)
print(results)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

#from ann_visualizer.visualize import ann_viz
#ann_viz(classifier, view=True, title="test", filename="visualized")


# In[ ]:

# Confusion Matrix Results

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

# fix random seed for reproducibility
seed = 5
numpy.random.seed(seed)
# load pima indians dataset
dataframe = pandas.read_csv("MalariaFieldData11.csv", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:14]
Y = dataset[:,14]
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
    model.add(Dense(55, input_dim=14, init='normal', activation='relu'))
    model.add(Dropout(0.01))
    model.add(Dense(65, activation='sigmoid'))
    #model.add(Dropout(0.01))
    model.add(Dense(2, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
# Model training
estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=50, batch_size=5, verbose=0)
X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.20, random_state=seed)
estimator.fit(X_train, Y_train)
predictions = estimator.predict(X_test)

#print(predictions)
#y_test_class = np.argmax(Y_test,axis=1)
#y_prediction_class = np.argmax(predictions,axis=0)


#print(classification_report(y_test_class,y_prediction_class))
#print(confusion_matrix(y_test_class,y_prediction_class))


#p=model.predict_proba(X_test) # to predict probability

target_names = ['class 0(Negative)', 'class 1(Positive)']
print(classification_report(np.argmax(Y_test,axis=1), predictions,target_names=target_names))
print("===== Confuision Matrix Result====")
print(confusion_matrix(np.argmax(Y_test,axis=1), predictions))
#plot_confusion_matrix(np.argmax(Y_test,axis=1), predictions,target_names=target_names)
#plt.show()


# ============= Displaying the architecture of the final model ============

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
dataframe = pandas.read_csv("MalariaFieldData11.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:14].astype(float)
Y = dataset[:,14]

# hot encoding 
#encoder = LabelEncoder()
#encoder.fit(Y)
#encoded_Y = encoder.transform(Y)
#dummy_y = np_utils.to_categorical(encoded_Y)

# model definition
model = Sequential()
model.add(Dense(55, input_dim=14, init='normal', activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(65, activation='sigmoid'))
#model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))
# model Summary

model.summary()


# In[ ]:

============ Visulaizing Model Performance ================


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
# load dataset
dataframe = pandas.read_csv("MalariaFieldData11.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:14].astype(float)
Y = dataset[:,14]

# hot encoding 
#encoder = LabelEncoder()
#encoder.fit(Y)
#encoded_Y = encoder.transform(Y)
#dummy_y = np_utils.to_categorical(encoded_Y)

# model definition
model = Sequential()
model.add(Dense(55, input_dim=14, init='normal', activation='relu'))
model.add(Dropout(0.01))
model.add(Dense(65, activation='sigmoid'))
#model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
history = model.fit(X, Y, validation_split=0.20, epochs=50, batch_size=5, verbose=0)
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

=============== Making Prediction witth the Model===============


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
#Confusion matrix tries 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# fix random seed for reproducibility
seed = 5
numpy.random.seed(seed)
# load pima indians dataset
dataframe = pandas.read_csv("MalariaDataRework2.csv", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:11]
Y = dataset[:,11]
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
    model.add(Dense(30, input_dim=11, init='normal', activation='relu'))
    model.add(Dropout(0.01))
    model.add(Dense(18, activation='sigmoid'))
    model.add(Dense(2, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
# Model training
estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=50, batch_size=5, verbose=0)
X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.20, random_state=seed)
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

#confusion_matrix(Y_test, y_pred)


# In[ ]:



