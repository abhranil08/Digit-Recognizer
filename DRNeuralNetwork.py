import numpy
import pandas
from keras.models import Sequential
from keras.layers.core import Dropout
#from keras.models import Dropout
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.utils import to_categorical
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset



data=pd.read_csv("train.csv")
y=data["label"].values
x=data.drop("label",axis=1).values
x=x/255.0

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

"""
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)"""
# define baseline model
def baseline_model():
    # create model
    model = Sequential();
    model.add(Dense(264, input_dim=784, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.3))    
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10, kernel_initializer='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


y_train = to_categorical(y_train) #we are using 1 hot encoding here
model=baseline_model()
neuralModel=model.fit(X_train , y_train,epochs = 100,batch_size = 32 ,verbose=1)
test_loss, test_acc = model.evaluate(X_train, y_train)
print('Test accuracy:', test_acc)


"""estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=100, batch_size=32, verbose=1)
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
estimator.fit(X_train, Y_train)


kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, x, y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))"""



test=pd.read_csv("test.csv").values
predictionsNeural = estimator.predict(test)


outNeural=pd.DataFrame(predictionsNeural)
outNeural.index = np.arange(1, len(outNeural)+1)
outNeural.columns=["Label"]
outNeural.to_csv("predictionNeural.csv",index_label="ImageId")
