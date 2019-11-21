# Convolutional Neural Network

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
import tensorflow as tf
# from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Dropout, Dense
from tensorflow.python.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

no_of_classes = 3

# Initialising the CNN
classifier = tf.keras.Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(16, (3, 3), input_shape=(224, 224, 1), activation='relu'))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2), strides=2))

classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2), strides=2))

classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2), strides=2))

classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2), strides=2))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units=1024, activation='relu'))
classifier.add(Dense(units=no_of_classes, activation='softmax'))

# Compiling the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Part 2 - Fitting the CNN to the images

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1. / 255)

test_datagen = ImageDataGenerator(rescale=1. / 255)

Training = train_datagen.flow_from_directory('dataset/train',
                                             target_size=(224, 224),
                                             batch_size=32,
                                             color_mode='grayscale',
                                             class_mode='categorical')

Validation = test_datagen.flow_from_directory('dataset/test',
                                              target_size=(224, 224),
                                              batch_size=32,
                                              color_mode='grayscale',
                                              class_mode='categorical')

history = classifier.fit_generator(Training,
                         steps_per_epoch=50,
                         epochs=20,
                         validation_data=Validation,
                         validation_steps=50)

classifier.summary()

classifier.save('trace60(2).h5')
converter = tf.lite.TFLiteConverter.from_keras_model_file("trace60(2).h5")
tflite_model = converter.convert()
open("trace60(2).tflite", "wb").write(tflite_model)


# Plotting graphs
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()