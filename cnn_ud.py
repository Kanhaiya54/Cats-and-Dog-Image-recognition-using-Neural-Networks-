import keras

from keras.models import Sequential # or graphs
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


classifier = Sequential()
classifier.add(Conv2D(32,(3,3),input_shape=(64,64,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2),padding='same'))
classifier.add(Flatten())
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=1,activation='sigmoid'))
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
                                                                                    'dataset/training_set',
                                                                                    target_size=(64, 64),#input image size to the CNN layer
                                                                                    batch_size=32,
                                                                                    class_mode='binary')#2 classes output

test_set = test_datagen.flow_from_directory(
                                                                                                'dataset/test_set',
                                                                                                target_size=(64, 64),
                                                                                                batch_size=32,
                                                                                                class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,#no of images in the training set.
        epochs=2,
        validation_data=test_set,
        validation_steps=2000)