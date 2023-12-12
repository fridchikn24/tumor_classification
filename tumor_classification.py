import keras.preprocessing.image
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout , BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, Adamax
from keras import regularizers
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report



path = Path("C:/Users/fridc/Documents/Data_Projects/tumor_classification")

training_generator = ImageDataGenerator(rescale=1./255,
                                        rotation_range=7,
                                        horizontal_flip=True,
                                        zoom_range=0.2)
training_dataset = training_generator.flow_from_directory(path /'Training',
                                                        target_size = (64, 64),
                                                        batch_size = 32,
                                                        class_mode = 'categorical',
                                                        shuffle = True)

print(training_dataset.class_indices)

test_generator = ImageDataGenerator(rescale=1./255)
test_dataset = test_generator.flow_from_directory(path /'Testing',
                                                     target_size = (64, 64),
                                                     batch_size = 1,
                                                     class_mode = 'categorical',
                                                     shuffle = False)

cnn = Sequential()
cnn.add(Conv2D(32, (3,3), input_shape = (64,64,3), activation='relu'))
cnn.add(MaxPooling2D(2,2))
cnn.add(Conv2D(32, (3,3), activation='relu'))
cnn.add(MaxPooling2D(2,2))
cnn.add(Flatten())
cnn.add(Dense(units = 731, activation='relu'))
cnn.add(Dense(units = 64, activation='relu'))
cnn.add(Dense(units = 4, activation='softmax'))

cnn.summary()

cnn.compile(optimizer='Adam', loss='categorical_crossentropy', metrics = ['accuracy'])
historic = cnn.fit(training_dataset, epochs = 10)


predictions = cnn.predict(test_dataset)
predictions = np.argmax(predictions, axis = 1)

print(training_dataset.class_indices)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_dataset.classes, predictions)
print(cm)


from sklearn.metrics import classification_report
print(classification_report(test_dataset.classes, predictions))