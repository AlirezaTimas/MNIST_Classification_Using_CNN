#yemosht library import mikonim , dataset MNIST , model sequential , laye haye dense dropout flatten , conv2d maxpooling2d
#akharam abzar to_categorical ro import mikonim vase convert vector be binary metric
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical

# MNIST ro load mikonim
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Data Ro reshape mikonim : 
#tozihat : dataset MNIST shamel 60k image greyscale e 28x28 ke har pixel (0,255) e. 
#vase inke beshe az Keras estefade kard niaze ke input 4D tensor be in shekl  bashe (andaze, height, width, channel rangi)
#yani masalan vase MNIST mishe (tedadaksMNIST, 28(height), 28(width), 1(channel rangi))
#code zir hamoon kare reshape ro anjam mide
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# Data ro normalize mikonim
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Class vectora ro be binary metric convert mikonim
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Model ro tarif mikonim
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# model ro compile mikonim
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# model ro train mikonim ba batchsize 128 va 10 epoch
model.fit(X_train, y_train, batch_size=128, epochs=10, verbose=1, validation_data=(X_test, y_test))

# Model ro test mikonim 
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#result test besoorat floating number namayesh dade mishe ke vase mohasebe % accuracy bayad result ro dar 100 zarb konim
#tooye testaye mokhtalef result in model 95-98% boode
#AlirezaTimas