# -*- coding: utf-8 -*-

import tensorflow.keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import pickle

from datetime import datetime
startTime = datetime.now()

"""
set variable
"""
headers =  ['easy', 'hard', 'numerical', \
            'visual', 'audio', 'bell', \
            'n_current_viewing',\
            'audio_attachment', 'bell_attachment',\
            'tq_current', 'ta_current', 'eval_current',\
            'tq_prev', 'ta_prev', 'eval_prev',\
            'eval_next' ,'audio_reinforcement'  ]
    
optimizer = 'adam'
loss_function = 'categorical_crossentropy'
num_epochs = 100
num_hidden1_node = 16
num_hidden2_node = 8



"""
read data from the text file and seperate into train set and label set
"""
dataframe = pd.read_csv('input_data_for_training.txt', sep='\t', names=headers)
X = dataframe.drop(columns=['eval_next','audio_reinforcement'])
y_label = np.ravel(dataframe['audio_reinforcement'].values.reshape(X.shape[0], 1))

x_train, x_test, y_train, y_test = train_test_split(X, y_label,test_size=0.2, random_state=2)

"""
choose the columns you want to do normalize 
"""
train_norm = x_train[x_train.columns[0:16]]
test_norm = x_test[x_test.columns[0:16]]

"""
Normalize train and test set by using StandardScaler()
Update the train and test set
"""
std_scale = preprocessing.StandardScaler().fit(train_norm)
x_train_norm = std_scale.transform(train_norm)
x_test_norm = std_scale.transform(test_norm)

training_norm_col = pd.DataFrame(x_train_norm, index=train_norm.index, columns=train_norm.columns) 
x_train.update(training_norm_col)

testing_norm_col = pd.DataFrame(x_test_norm, index=test_norm.index, columns=test_norm.columns) 
x_test.update(testing_norm_col)

"""
Reshape the train set and change the y label to categorical (0/1)
"""
x_train_arr = x_train.values
x_test_arr = x_test.values

x_train_arr = np.reshape(x_train_arr, (x_train_arr.shape[0],1,x_train_arr.shape[1]))
x_test_arr = np.reshape(x_test_arr, (x_test_arr.shape[0],1,x_test_arr.shape[1]))

y_train = to_categorical(y_train)
y_train = np.reshape(y_train, (y_train.shape[0],1,y_train.shape[1]))

y_test = to_categorical(y_test)
y_test = np.reshape(y_test, (y_test.shape[0],1,y_test.shape[1]))



"""
build NN model
"""
model = Sequential()

model.add(Dense(num_hidden1_node,activation = 'sigmoid'))
model.add(Dropout(0.1))
model.add(Dense(num_hidden2_node,activation = 'sigmoid'))
model.add(Dropout(0.1))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer= optimizer, loss=loss_function , metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss',patience=2)

history = model.fit(x_train_arr, y_train,epochs=num_epochs,validation_data=(x_test_arr, y_test),callbacks=[early_stopping])

"""
predict and test the performance of model with test set
"""
preds_train = model.predict(x_train_arr)
preds_test = model.predict(x_test_arr)

preds_test_class = model.predict_classes(x_test_arr)
print('predict result 0: {} , 1: {}'.format(preds_test_class.size-np.count_nonzero(preds_test_class),np.count_nonzero(preds_test_class)))
train_loss, train_accuracy = model.evaluate(x_train_arr,y_train)
print('train loss {} train accuracy {}'.format(train_loss, train_accuracy))
test_loss, test_accuracy = model.evaluate(x_test_arr,y_test)
print('test loss {} test accuracy {}'.format(test_loss, test_accuracy))


"""
save model
"""
model.save('models/RNN_binary_normalize')
pickle.dump(model, open('models/RNN_binary_normalize_test.pkl','wb'))
pickle.dump(std_scale, open('models/RNN_binary_normalize_scaler.pkl','wb'))

"""
summarize history for accuracy and plot graph (accuracy and loss)
"""
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15,5))

ax1.plot(history.history['accuracy'])
ax1.plot(history.history['val_accuracy'])
ax1.set_title('model accuracy')
ax1.set_ylabel('accuracy')
ax1.set_xlabel('epoch')
ax1.legend(['train', 'test'], loc='upper left')

ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('model loss')
ax2.set_ylabel('loss')
ax2.set_xlabel('epoch')
ax2.legend(['train', 'test'], loc='upper left')
plt.show()

