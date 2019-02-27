# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
train_file = "../input/train.csv"
test_file = "../input/test.csv"
output_file = "submission.csv"

import numpy as np # linear algebra
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler

train=pd.read_csv(train_file)
print(train.shape)

raw_data = np.loadtxt(train_file, skiprows=1, dtype='int', delimiter=',')
x_train, x_test, y_train, y_test = train_test_split(
    raw_data[:,1:], raw_data[:,0], test_size=0.1)
    
"""    
y=train["label"].values
x=train.drop("label",axis=1).values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1)
#print(train_file.shape)"""

x_train=x_train.reshape(-1,28,28,1)
x_test=x_test.reshape(-1,28,28,1)

x_train = x_train.astype("float32")/255.
x_test = x_test.astype("float32")/255.

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

##Convolutional Network

model=Sequential()
model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu',input_shape = (28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.25))



model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(512,activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(1024, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10,activation="softmax"))


datagen=ImageDataGenerator(zoom_range=0.1,height_shift_range=0.1,width_shift_range=0.1,rotation_range=10)

#model.compile(loss="categorical_crossentrophy",optimizer=Adam(lr=1e-4),metrics=["accuracy"])
model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=1e-4), metrics=["accuracy"])
gradient = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)

cnnModel = model.fit_generator(datagen.flow(x_train, y_train, batch_size=16),
                           steps_per_epoch=500,
                           epochs=300, #Increase this when not on Kaggle kernel
                           verbose=2,  #1 for ETA, 0 for silent
                           validation_data=(x_test[:400,:], y_test[:400,:]), #For speed
                           callbacks=[gradient])
                           
                           
testKaggle=np.loadtxt(test_file,skiprows=1,dtype='int',delimiter=',')
test=testKaggle.astype('float32')
test=test.reshape(-1,28,28,1)/255.0

prediction=model.predict(test,batch_size=64)
predictionFinal=np.argmax(prediction,axis=1)


with open(output_file, 'w') as f :
    f.write('ImageId,Label\n')
    for i in range(len(predictionFinal)) :
        f.write("".join([str(i+1),',',str(predictionFinal[i]),'\n']))
# Any results you write to the current directory are saved as output.