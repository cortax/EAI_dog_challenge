import keras
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, load_img, list_pictures, array_to_img, img_to_array
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.regularizers import l2
from keras.layers.core import Activation, Dense
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
from keras.callbacks import LearningRateScheduler
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import wget
import os
import tarfile
from PIL import Image
import PIL as pil_image
import glob
import numpy as np
from math import floor
import h5py
from load_dataset import prepare_dataset

# Download dataset from web, uncompress it and divide into a train and test folder
prepare_dataset()
train_data_dir = 'Images/train'
validation_data_dir = 'Images/test'

# Learning parameters
HEIGHT = 299
WIDTH = 299
LEARNING_RATE = 0.0001
BATCH_SIZE = 8
NB_EPOCH = 15
MOMENTUM = 0.9

# Load pretrained InceptionV3
base_model = InceptionV3(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(120, activation='softmax')(x)
model = Model(input=base_model.input, output=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Create data generators
N_train = sum([len(files) for r, d, files in os.walk(os.getcwd() + '/' + train_data_dir)])
N_test = sum([len(files) for r, d, files in os.walk(os.getcwd() + '/' + validation_data_dir)])

train_datagen = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.2,
        rotation_range=90,
        width_shift_range=5/WIDTH,
        height_shift_range=5/HEIGHT,
        shear_range=2.0*np.pi*5/360,
        fill_mode='reflect',
        horizontal_flip=True,
        vertical_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(WIDTH, HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(WIDTH, HEIGHT),
        batch_size=BATCH_SIZE,
        shuffle=False,
        class_mode='categorical')

# Preparing callbacks
tb = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=False)
try:
    os.remove('./results.csv')
except:
    pass
csvlog = CSVLogger('./results.csv', separator=',', append=True)
m_checkpoint = ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, period=1)

# Training the top layers
print('Phase 1 - train a new fully connected layer')
model.fit_generator(
        train_generator,
        samples_per_epoch=N_train,
        nb_epoch=1,
        callbacks=[tb, csvlog, m_checkpoint],
        validation_data=validation_generator,
        nb_val_samples=N_test)

# Retraining first inception blocks
print('Phase 2 - retrain inception blocks - depth(1)')
for layer in model.layers[:172]:
   layer.trainable = False
for layer in model.layers[172:]:
   layer.trainable = True

from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=LEARNING_RATE, momentum=MOMENTUM), loss='categorical_crossentropy', metrics=['accuracy'])

# defining learning rate ladder
def f_sched(epoch):
    lr_decay = 0.1
    lr_schedule = [5, 10]
    return LEARNING_RATE * lr_decay**(np.array(lr_schedule) <= epoch).sum()

lrs = LearningRateScheduler(f_sched)

model.fit_generator(
        train_generator,
        samples_per_epoch=N_train,
        nb_epoch=NB_EPOCH,
        callbacks=[tb, lrs, csvlog, m_checkpoint],
        validation_data=validation_generator,
        nb_val_samples=N_test)

# Retraining second inception blocks
print('Phase 3 - retrain inception blocks - depth(2)')
for layer in model.layers[:158]:
   layer.trainable = False
for layer in model.layers[158:]:
   layer.trainable = True

from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=LEARNING_RATE, momentum=MOMENTUM), loss='categorical_crossentropy', metrics=['accuracy'])

# REdefining learning rate ladder
def f_sched(epoch):
    lr_decay = 0.1
    lr_schedule = [5, 10]
    return LEARNING_RATE * lr_decay**(np.array(lr_schedule) <= epoch).sum()

lrs = LearningRateScheduler(f_sched)

model.fit_generator(
        train_generator,
        samples_per_epoch=N_train,
        nb_epoch=NB_EPOCH,
        callbacks=[tb, lrs, csvlog, m_checkpoint],
        validation_data=validation_generator,
        nb_val_samples=N_test)

# Retraining second inception blocks
print('Phase 4 - retrain inception blocks - depth(3)')
for layer in model.layers[:136]:
   layer.trainable = False
for layer in model.layers[136:]:
   layer.trainable = True

from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=LEARNING_RATE, momentum=MOMENTUM), loss='categorical_crossentropy', metrics=['accuracy'])

# REdefining learning rate ladder
def f_sched(epoch):
    lr_decay = 0.1
    lr_schedule = [5, 10]
    return LEARNING_RATE * lr_decay**(np.array(lr_schedule) <= epoch).sum()

lrs = LearningRateScheduler(f_sched)

model.fit_generator(
        train_generator,
        samples_per_epoch=N_train,
        nb_epoch=NB_EPOCH,
        callbacks=[tb, lrs, csvlog, m_checkpoint],
        validation_data=validation_generator,
        nb_val_samples=N_test)
