# -*- coding: utf-8 -*-
# File: cnn_transfer_example.py
# ---------------------
# Train a deep convolutional neural network by fine-tuning on VGG16.
#
# Code adapted from
# https://github.com/fchollet/keras/blob/master/examples/mnist_transfer_cnn.py

from __future__ import print_function
import numpy as np
import datetime
import os

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils

import h5py
import logging
from cnn_baseline import CNN

now = datetime.datetime.now

DATA_DIR = 'data'

batch_size = 128
nb_classes = 7
nb_epoch = 8

# input image dimensions
img_rows, img_cols = 48, 48
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3


def train_model(model, train, test, nb_classes):
  X_train = train[0]

  X_test = test[0]
  X_train = X_train.astype('float32')
  X_test = X_test.astype('float32')
  X_train /= 255
  X_test /= 255

  # Convert from 1-channel images to 3-channel images by duplicating channel.
  # TODO: Find a model that is trained in gray scale so we don't have to do
  # this hack.
  X_train_expanded = np.zeros((X_train.shape[0], 3, X_train.shape[2], X_train.shape[3]))
  X_train_expanded[:, 0, :, :] = X_train[:, 0, :, :]
  X_train_expanded[:, 1, :, :] = X_train[:, 0, :, :]
  X_train_expanded[:, 2, :, :] = X_train[:, 0, :, :]

  X_test_expanded = np.zeros((X_test.shape[0], 3, X_test.shape[2], X_test.shape[3]))
  X_test_expanded[:, 0, :, :] = X_test[:, 0, :, :]
  X_test_expanded[:, 1, :, :] = X_test[:, 0, :, :]
  X_test_expanded[:, 2, :, :] = X_test[:, 0, :, :]

  X_train = X_train_expanded
  X_test = X_test_expanded
  print('SHAPE:', X_train.shape)
  print('SHAPE:', X_test.shape)

  print('X_train shape:', X_train.shape)
  print(X_train.shape[0], 'train samples')
  print(X_test.shape[0], 'test samples')

  # convert class vectors to binary class matrices
  Y_train = np_utils.to_categorical(train[1], nb_classes)
  Y_test = np_utils.to_categorical(test[1], nb_classes)

  model.compile(loss='categorical_crossentropy', optimizer='adadelta')

  t = now()
  model.fit(X_train, Y_train,
        batch_size=batch_size, nb_epoch=nb_epoch,
        show_accuracy=True, verbose=2,
        validation_data=(X_test, Y_test))
  print('Training time: %s' % (now() - t))
  score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=1)
  print('Test score:', score[0])
  print('Test accuracy:', score[1])


def load_vgg_model(weights_path):

  # build the VGG16 network with our input_img as input
  first_layer = ZeroPadding2D((1, 1), input_shape=(3, img_cols, img_rows))
  # first_layer.input = input_img

  # Model replicates VGG16 architecture:
  # https://gist.github.com/jimmie33/27c1c0a7736ba66c2395
  model = Sequential()
  model.add(first_layer)
  model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
  model.add(MaxPooling2D((2, 2), strides=(2, 2)))

  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
  model.add(MaxPooling2D((2, 2), strides=(2, 2)))

  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
  model.add(MaxPooling2D((2, 2), strides=(2, 2)))

  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
  model.add(MaxPooling2D((2, 2), strides=(2, 2)))

  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
  # model.add(MaxPooling2D((2, 2), strides=(2, 2)))

  # load the weights of the VGG16 networks
  # (trained on ImageNet, won the ILSVRC competition in 2014)
  # note: when there is a complete match between your model definition
  # and your weight savefile, you can simply call model.load_weights(filename)
  assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
  f = h5py.File(weights_path)
  for k in range(f.attrs['nb_layers']):
    if k >= len(model.layers):
      # we don't look at the last (fully-connected) layers in the savefile
      break
    g = f['layer_{}'.format(k)]
    weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    model.layers[k].set_weights(weights)
    model.layers[k].trainable = False # Freeze these layers.
  f.close()
  print('VGG 16 weights loaded.')

  # Add classification layers.
  model.add(Flatten())
  model.add(Dense(128))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(nb_classes))
  model.add(Activation('softmax'))

  return model

def run_cnn():
  # Define file paths.
  train_data_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), DATA_DIR, 'train.csv')
  val_data_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), DATA_DIR, 'val.csv')
  weights_data_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), DATA_DIR, 'vgg16_weights.h5')

  # Use a dummy CNN to load data.
  cnn = CNN(params={}, verbose=True)
  cnn.load_data(train_data_file, val_data_file, num_train=800, num_val=200)
  X_train, y_train = cnn.X_train, cnn.y_train
  X_val, y_val = cnn.X_val, cnn.y_val
  print('Loaded input data.')

  # Train and evaluate the model.
  model = load_vgg_model(weights_data_file)
  train_model(model, (X_train, y_train), (X_val, y_val), nb_classes)

def main():
  logging.basicConfig(format="[%(name)s %(asctime)s]\t%(msg)s", level=logging.INFO)
  run_cnn()

if __name__ == '__main__':
  main()
