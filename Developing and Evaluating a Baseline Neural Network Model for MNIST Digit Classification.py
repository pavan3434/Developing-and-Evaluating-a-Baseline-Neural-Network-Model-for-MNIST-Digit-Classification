#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# Load (downloaded if needed) the MNIST dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Describe the datasets
print("Training data - X_train shape:", X_train.shape, " Y_train shape:", Y_train.shape)
print("Testing data  - X_test shape:", X_test.shape, " Y_test shape:", Y_test.shape)

# Flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

# Normalize inputs from 0-255 to 0-1
X_train = X_train / 255.
X_test = X_test / 255.

# One hot encode outputs
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)
num_classes = Y_test.shape[1]

# Define baseline model
model = Sequential()
model.add(Dense(num_pixels, input_dim=num_pixels, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Show model summary
model.summary()

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=200, verbose=2)

# Final evaluation of the model
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))


# # 2nd

# In[5]:


import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# Load (downloaded if needed) the MNIST dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

# Normalize inputs from 0-255 to 0-1
X_train = X_train / 255.
X_test = X_test / 255.

# One hot encode outputs
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)
num_classes = Y_test.shape[1]

# Define baseline model
def create_model():
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

# Compile model
model = create_model()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Function to train and evaluate model with varying epochs and batch size
def train_and_evaluate(epochs, batch_size):
    model = create_model()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs, batch_size=batch_size, verbose=0)
    scores = model.evaluate(X_test, Y_test, verbose=0)
    return history, scores

# Varying epochs and batch size
epochs_list = [5, 10, 15, 20, 25]
batch_size_list = [64, 128, 256, 512, 1024]

for epochs in epochs_list:
    for batch_size in batch_size_list:
        print(f"\nTraining with {epochs} epochs and {batch_size} batch size:")
        history, scores = train_and_evaluate(epochs, batch_size)
        print("Test Accuracy: %.2f%%" % (scores[1]*100))


# # 3

# In[6]:


# Define model with additional hidden layers
def create_deep_model():
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, activation='relu'))
    model.add(Dense(512, activation='relu'))  # Additional hidden layer
    model.add(Dense(256, activation='relu'))  # Additional hidden layer
    model.add(Dense(128, activation='relu'))  # Additional hidden layer
    model.add(Dense(num_classes, activation='softmax'))
    return model

# Compile and train deep model
model = create_deep_model()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history_deep = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=200, verbose=2)

# Evaluate deep model
scores_deep = model.evaluate(X_test, Y_test, verbose=0)
print("Deep Model Test Accuracy: %.2f%%" % (scores_deep[1]*100))


# # II
# 

# In[1]:


import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical

from keras.datasets import mnist
import matplotlib.pyplot as plt

from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# load (downloaded if needed) the MNIST dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

img_rows, img_cols = 28, 28

# You probably don’t need this part, (unless your ‘Tensorflow’ uses ‘theano’ as the backend engine)
# but just in case.
if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape) 


# plot 4 images as gray scale
plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()

# Plotting the first 4 images
plt.figure(figsize=(8, 8))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(X_train[i].reshape(28, 28), cmap='gray')
    plt.title("Label: {}".format(Y_train[i]))
plt.show() 

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

num_classes = 10

# one hot encode outputs
Y_train = to_categorical(Y_train, num_classes)
Y_test = to_categorical(Y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
# add batch normalization
model.add(Conv2D(64, (3, 3), activation='relu'))
# add batch normalization
model.add(MaxPooling2D(pool_size=(2, 2)))
# add dropout
model.add(Flatten())
model.add(Dense(128, activation='relu'))
# add dropout
model.add(Dense(num_classes, activation='softmax'))

# Displaying model summary
model.summary() 


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# you may change epochs and batch_size for performance
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=200, verbose=2)

score = model.evaluate(X_test, Y_test, verbose=0)
#print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[12]:


import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical

from keras.datasets import mnist
import matplotlib.pyplot as plt

from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# load (downloaded if needed) the MNIST dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

img_rows, img_cols = 28, 28

# You probably don’t need this part, (unless your ‘Tensorflow’ uses ‘theano’ as the backend engine)
# but just in case.
if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape) 


# plot 4 images as gray scale
plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()

# Plotting the first 4 images
plt.figure(figsize=(8, 8))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(X_train[i].reshape(28, 28), cmap='gray')
    plt.title("Label: {}".format(Y_train[i]))
plt.show() 

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

num_classes = 10

# one hot encode outputs
Y_train = to_categorical(Y_train, num_classes)
Y_test = to_categorical(Y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape))

# add batch normalization
model.add(Conv2D(64, (3, 3), activation='relu'))
# add batch normalization
model.add(MaxPooling2D(pool_size=(2, 2)))
# add dropout
model.add(Flatten())
model.add(Dense(128, activation='relu'))
# add dropout
model.add(Dense(num_classes, activation='softmax'))

# Displaying model summary
model.summary() 


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# you may change epochs and batch_size for performance
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=200, verbose=2)

score = model.evaluate(X_test, Y_test, verbose=0)
#print('Test loss:', score[0])
print('Test accuracy:', score[1])


# (7,7)

# In[13]:


import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical

from keras.datasets import mnist
import matplotlib.pyplot as plt

from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# load (downloaded if needed) the MNIST dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

img_rows, img_cols = 28, 28

# You probably don’t need this part, (unless your ‘Tensorflow’ uses ‘theano’ as the backend engine)
# but just in case.
if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape) 


# plot 4 images as gray scale
plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()

# Plotting the first 4 images
plt.figure(figsize=(8, 8))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(X_train[i].reshape(28, 28), cmap='gray')
    plt.title("Label: {}".format(Y_train[i]))
plt.show() 

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

num_classes = 10

# one hot encode outputs
Y_train = to_categorical(Y_train, num_classes)
Y_test = to_categorical(Y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(7, 7), activation='relu', input_shape=input_shape))

# add batch normalization
model.add(Conv2D(64, (3, 3), activation='relu'))
# add batch normalization
model.add(MaxPooling2D(pool_size=(2, 2)))
# add dropout
model.add(Flatten())
model.add(Dense(128, activation='relu'))
# add dropout
model.add(Dense(num_classes, activation='softmax'))

# Displaying model summary
model.summary() 


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# you may change epochs and batch_size for performance
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=200, verbose=2)

score = model.evaluate(X_test, Y_test, verbose=0)
#print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:





# model.add(Dropout(0.25))
# &

# In[15]:


from keras.layers import Dropout

# define the model architecture
model = Sequential()

# Add the first convolutional layer
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))

# Add the second convolutional layer
model.add(Conv2D(64, (3, 3), activation='relu'))

# Add the first MaxPooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add Dropout after the first MaxPooling layer
model.add(Dropout(0.25))

# Flatten the feature maps
model.add(Flatten())

# Add the first Dense hidden layer
model.add(Dense(128, activation='relu'))

# Add Dropout after the first Dense hidden layer
model.add(Dropout(0.5))

# Add the output layer
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=200, verbose=2)

# Evaluate the model on the test set
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test accuracy:', score[1])


# from tensorflow.keras import regularizers
# 
# model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.05)))
# 
