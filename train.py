# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import keras
from keras.preprocessing.image import ImageDataGenerator, load_img, list_pictures, array_to_img, img_to_array
from keras.datasets import cifar10
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

# <codecell>

if os.path.isdir("./Images"):
    print('Image folder found')
else:
    if os.path.isfile("images.tar"):
        print('Image archive found')
    else:
        url = 'http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar'
        filename = wget.download(url)
    filename = 'images.tar'
    opener, mode = tarfile.open, 'r'
    file = opener('./' + filename, mode)
    file.extractall()
    file.close()

cwd = os.getcwd()
os.makedirs(cwd + '/Images/train', exist_ok=True)
os.makedirs(cwd + '/Images/test', exist_ok=True)

for d in [x[0] for x in os.walk('./Images/')]:
    basedir = os.path.split(d)
    if not basedir[1] == '' and not basedir[1] == 'train' and not basedir[1] == 'test' and basedir[0] == './Images':
        print(basedir[1])
        L = glob.glob(str(d) + '/*.jpg')
        lastfile = floor(0.85*len(L))
        for pathname in L[0:lastfile]:
            parsed_filename = pathname.split('/')
            parsed_filename[0] = cwd
            oldpathname = '/'.join(parsed_filename)
            parsed_filename.insert(2, 'train')
            newpathname = '/'.join(parsed_filename)
            os.makedirs('/'.join(parsed_filename[:-1]), exist_ok=True)
            os.rename(oldpathname, newpathname)
        for pathname in L[lastfile:]:
            parsed_filename = pathname.split('/')
            parsed_filename[0] = cwd
            oldpathname = '/'.join(parsed_filename)
            parsed_filename.insert(2, 'test')
            newpathname = '/'.join(parsed_filename)
            os.makedirs('/'.join(parsed_filename[:-1]), exist_ok=True)
            os.rename(oldpathname, newpathname)
        os.rmdir(cwd + '/Images/' + basedir[1])

# <codecell>


# <codecell>

train_data_dir = 'Images/train'
validation_data_dir = 'Images/test'

HEIGHT = 300
WIDTH = 300
CHAN = 3
WEIGHT_DECAY = 5e-4
LEARNING_RATE = 2
BATCH_SIZE = 4
NB_EPOCH = 10
MOMENTUM = 0.99

N_train = sum([len(files) for r, d, files in os.walk(os.getcwd() + '/Images/train')])
N_test = sum([len(files) for r, d, files in os.walk(os.getcwd() + '/Images/test')])

# <codecell>


# <codecell>

model = Sequential()

model.add(Convolution2D(96, 3, 3, init='he_normal', W_regularizer=l2(WEIGHT_DECAY), bias=True, input_shape=(WIDTH, HEIGHT, 3), border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid'))
model.add(Dropout(0.5))

model.add(Convolution2D(64, 3, 3, init='he_normal', W_regularizer=l2(WEIGHT_DECAY), bias=False, input_shape=(WIDTH, HEIGHT, 3), border_mode='same'))
model.add(BatchNormalization(epsilon=0.001, mode=0, axis=3, momentum=MOMENTUM, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=l2(WEIGHT_DECAY), beta_regularizer=l2(WEIGHT_DECAY)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid'))
model.add(Dropout(0.5))

model.add(Convolution2D(32, 3, 3, init='he_normal', W_regularizer=l2(WEIGHT_DECAY), bias=False, input_shape=(WIDTH, HEIGHT, 3), border_mode='same'))
model.add(BatchNormalization(epsilon=0.001, mode=0, axis=3, momentum=MOMENTUM, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=l2(WEIGHT_DECAY), beta_regularizer=l2(WEIGHT_DECAY)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid'))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(output_dim=512, init='glorot_uniform', W_regularizer=l2(WEIGHT_DECAY), bias=False))
model.add(BatchNormalization(epsilon=0.001, mode=0, momentum=MOMENTUM, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=l2(WEIGHT_DECAY), beta_regularizer=l2(WEIGHT_DECAY)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(output_dim=120, init='glorot_uniform', W_regularizer=l2(WEIGHT_DECAY), bias=True))
model.add(Activation('softmax'))

sgd = keras.optimizers.SGD(lr=LEARNING_RATE, momentum=MOMENTUM, decay=0.0, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

print(model.summary())

# <codecell>

train_datagen = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.2,
        rotation_range=5,
        width_shift_range=4/WIDTH,
        height_shift_range=4/HEIGHT,
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
        class_mode='categorical')



tb = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=False)
csvlog = CSVLogger('./results.csv', separator=',', append=False)
m_checkpoint = ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, period=1)

def f_sched(epoch):
    lr_decay = 0.1
    lr_schedule = [30, 70, 200]
    return LEARNING_RATE * lr_decay**(np.array(lr_schedule) <= epoch).sum()

lrs = LearningRateScheduler(f_sched)

model.fit_generator(
        train_generator,
        samples_per_epoch=N_train,
        nb_epoch=NB_EPOCH,
        callbacks=[tb, lrs, csvlog, m_checkpoint],
        validation_data=validation_generator,
nb_val_samples=N_test)

# <codecell>


# <codecell>


