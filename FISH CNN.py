# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
import numpy as np
import cv2
from keras.preprocessing import image

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(output_dim = 3, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1. / 255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('C:/Users/aryad/Desktop/dataset/training_set',
                                                 target_size = (64,64),
                                                 batch_size = 10,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory( 'C:/Users/aryad/Desktop/dataset/test_set', 
                                            target_size = (64,64),
                                            batch_size = 1,
                                            class_mode = 'categorical')

classifier.fit_generator(training_set, 
                         samples_per_epoch = 210,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 4)

#classifier.save("arya.h5")
#classifier.load("arya.h5")



test_image = image.load_img('C:/Users/aryad/Desktop/dataset/validation/3.jpg', target_size = (64, 64))
test_image1 = np.array(test_image)
#test_image2 = np.expand_dims(test_image1, axis = 0)
result = classifier.predict(test_image1)
training_set.class_indices

print (result)

if result[0][0] == 1:
    print ("fish")
else:
    print ("not fish")
    


#import pickle
# Open the file to save as pkl file
# The wb stands for write and binary
#model_pkl = open("model.pkl", "wb")

# Write to the file (dump the model)
# Open the file to save as pkl file
#pickle.dump(classifier, model_pkl)

# Close the pickle file
#model_pkl.close()