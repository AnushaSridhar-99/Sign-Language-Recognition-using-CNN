import os, cv2, math
from keras.models import Sequential, model_from_json
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator, load_img, image, img_to_array
from keras.utils import plot_model
import numpy as np
from sklearn.model_selection import train_test_split
from shutil import copyfile
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import Augmentor
from PIL import Image

dataset_path = 'asl_dataset_augmented'

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

training_data = train_datagen.flow_from_directory(os.path.join(dataset_path, 'training_set'),
                                                 target_size = (64,64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

validation_data = validation_datagen.flow_from_directory(os.path.join(dataset_path, 'validation_set'),
                                            target_size = (64,64),
                                            batch_size = 32,
                                            class_mode = 'categorical')

test_data = test_datagen.flow_from_directory(os.path.join(dataset_path, 'test_set'),
                                            target_size = (64,64),
                                            batch_size = 32,
                                            class_mode = 'categorical')


classifier = Sequential()

classifier.add(Conv2D(32, (3,3), input_shape = (64,64,3), activation= 'relu'))
# BatchNormalization()
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Dropout(0.25))

classifier.add(Conv2D(32, (3,3), activation = 'relu'))
# BatchNormalization()
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Dropout(0.25))

classifier.add(Conv2D(32, (3,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Dropout(0.25))

classifier.add(Flatten())
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 39, activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

history = classifier.fit_generator(training_data,
                                  steps_per_epoch= math.ceil(training_data.n/training_data.batch_size),
                                  epochs=30,
                                  validation_data= validation_data,
                                  validation_steps= math.ceil(validation_data.n/validation_data.batch_size))



classifier_json = classifier.to_json()
with open("model.json", "w") as json_file:
    json_file.write(classifier_json)

classifier.save_weights("model.h5")
print("Model saved to disk")

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model.h5")
print('Loaded model from disk')
loaded_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])



