# -*- coding: utf-8 -*-
# File: cnn_face_transfer.py
# ---------------------
# Train a deep convolutional neural network by fine-tuning on VGG FACE.
#
# Note that VGG FACE expects input images with 3 channels.
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

# Import Caffe Libraries
import numpy as np
import matplotlib.pyplot as plt
import sys
import caffe
import caffe.io
import collections

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

def get_caffe_params(netname, paramname):
  # load the model in
  net = caffe.Net(netname, paramname, caffe.TEST)
  params = collections.OrderedDict()

  # Read all the parameters into numpy arrays
  for layername in net.params:
    caffelayer = net.params[layername]
    params[layername] = []
    for sublayer in caffelayer:
      params[layername].append( sublayer.data ) 
    print("layer " + layername + " has " + str(len(caffelayer)) + " sublayers, shape " + str(params[layername][0].shape))

  return params, net

def train_model(model, train, test, nb_classes):
  X_train = train[0]

  X_test = test[0]
  X_train = X_train.astype('float32')
  X_test = X_test.astype('float32')
  X_train /= 255
  X_test /= 255

  # Convert from 1-channel images to 3-channel images by duplicating channel.
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

  # build the VGG_FACE_16 network with our input_img as input
  first_layer = ZeroPadding2D((1, 1), input_shape=(3, img_cols, img_rows))

  # Model replicates VGG_FACE_16 architecture:
  # http://www.robots.ox.ac.uk/~vgg/software/vgg_face/
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
#  model.add(MaxPooling2D((2, 2), strides=(2, 2)))

  # load the weights of the VGG_FACE_16 networks
  weightlayers=[]
  layerindex = 0
  for layer in model.layers:
    if len(layer.get_weights()) > 0:
      weightlayers.append(layerindex)
    layerindex+=1
  print("There are " + str(len(weightlayers)) + " layers in the model with weights")

  paramkeys = params.keys()

  for i in xrange(f.attrs['nb_layers']):
    if i >= len(model.layers):
      # we don't look at the last (fully-connected) layers in the savefile
      break
    layer = model.layers[ weightlayers[i] ]
    weights = params[paramkeys[i]]

    # Dense layers are specified as Input-Output in Keras
    if type(layer) is Dense:
      weights[0] = weights[0].transpose(1,0)
      weights[1] = weights[1]
    # Convolution 2D is specified as flip and then multiply
    elif type(layer) is Convolution2D:
      weights[0] = weights[0].transpose(0,1,2,3)[:,:,::-1,::-1]
    layer.set_weights( weights )
    layer.trainable = False # Freeze these layers

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
  weights_data_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), DATA_DIR, 'VGG_FACE.caffemodel')

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
