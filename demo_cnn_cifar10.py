import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils

n_train = 40000
n_val = 50000
# Split train, validation, test for datasets
(x_train, y_train),(x_test, y_test) = cifar10.load_data()

X_train, Y_train = x_train[:n_train,:], y_train[:n_train]
X_val, Y_val = x_train[n_train:n_val,:], y_train[n_train:n_val]

# To Categorial from label
encoder_y_train = np_utils.to_categorical(Y_train,10)
encoder_y_val = np_utils.to_categorical(Y_val,10)
encoder_y_test = np_utils.to_categorical(y_test,10)

# Build mode CNN Deep Learning
model = Sequential()

model.add(Conv2D(32,(3,3), activation='sigmoid', input_shape=(32,32,3)))
model.add(Conv2D(32,(3,3), activation='sigmoid'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(10,activation='softmax'))
model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
# Fit model 
H = model.fit(X_train, encoder_y_train, validation_data=(X_val, encoder_y_val), epochs=10)

# Matplotlib for loss, accuracy of training set and validation set
fig = plt.figure()
numOfEpoch = 10
 
plt.plot(np.arange(0, numOfEpoch), H.history['loss'], label = 'training loss')
plt.plot(np.arange(0, numOfEpoch), H.history['accuracy'], label = 'training accuracy')
plt.plot(np.arange(0, numOfEpoch), H.history['val_loss'], label = 'validation loss')
plt.plot(np.arange(0, numOfEpoch), H.history['val_accuracy'], label = 'validation accuracy')
plt.title('Accuracy and Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss|Accuracy')
plt.legend()

# Evaluate Testing set
score = model.evaluate(x_test, encoder_y_test)
print("Score of testing set:",score)

#Predict images
dict_label = {'0':'airplane','1':'automobile','2':'bird','3':'cat','4':'deer',
              '5':'dog','6':'frog','7':'horse','8':'ship','9':'truck'}

image = x_test[0]
plt.imshow(image)
y_predict = model.predict(image.reshape(1,32,32,3))
print('Predict label:', dict_label['{}'.format(np.argmax(y_predict))])