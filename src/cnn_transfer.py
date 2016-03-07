# -*- coding: utf-8 -*-
# File: cnn_transfer.py
# ---------------------
# Train a deep convolutional neural network by fine-tuning on VGG16
# using different fine-tuning methods.
#
# Some code adapted from
# https://github.com/fchollet/keras/blob/master/examples/mnist_transfer_cnn.py

from __future__ import print_function
import numpy as np
import datetime
import os

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import History

import argparse
import h5py
import logging
import pdb
from cnn_baseline import CNN

now = datetime.datetime.now

batch_size = 128
nb_classes = 7

IMG_DIM = 48
DATA_DIR = 'data'
DEFAULT_OUT_DIR = '../outputs/'
DEFAULT_NB_EPOCH = 3

# Mode enumeration.
FULL_FROZEN_VGG = 1
PART_FROZEN_VGG = 2
FULL_TRAINABLE_VGG = 3

# Input image dimensions.
img_rows, img_cols = 48, 48

# Number of convolutional filters to use.
nb_filters = 32

# Size of pooling area for max pooling.
nb_pool = 2

# Convolution kernel size
nb_conv = 3

def parse_args():
  """
  Parse the command line input.
  """
  train_data_file_default = os.path.join(os.path.dirname(os.path.dirname(__file__)), DATA_DIR, 'train.csv')
  val_data_file_default = os.path.join(os.path.dirname(os.path.dirname(__file__)), DATA_DIR, 'val.csv')
  default_num_train = 28709
  default_num_val = 3589

  parser = argparse.ArgumentParser()
  parser.add_argument('-td', default = train_data_file_default, help = 'training data file')
  parser.add_argument('-vd', default = val_data_file_default, help = 'validation data file')
  parser.add_argument('-e', default = DEFAULT_NB_EPOCH, help = 'number of epochs', type=int)
  parser.add_argument('-o', default = DEFAULT_OUT_DIR, help = 'location of output directory')
  parser.add_argument('-nt', default = default_num_train, help = 'number of training examples to use', type=int)
  parser.add_argument('-nv', default = default_num_val, help = 'number of validation examples to use', type=int)
  parser.add_argument('-m', required = True, type=int, help = 'mode: ' +
    '1 -- train linear classifier on top of full VGG16 network; ' +
    '2 -- train linear classifier on top of part of VGG16 network; ' + 
    '3 -- train entire network, using VGG16 weights as initializations')

  args = parser.parse_args()
  params = {'nb_epoch': args.e, 'output_dir': args.o, 'mode': args.m}
  return args.td, args.vd, args.nt, args.nv, args.m, params

def prepare_data(train, val):
  '''
  Return formatted input data.
  '''
  X_train = train[0]
  X_val = val[0]
  X_train = X_train.astype('float32')
  X_val = X_val.astype('float32')
  X_train /= 255
  X_val /= 255

  # Convert from 1-channel images to 3-channel images by duplicating across
  # color channels
  X_train_expanded = np.zeros((X_train.shape[0], 3, X_train.shape[2], X_train.shape[3]))
  X_train_expanded[:, 0, :, :] = X_train[:, 0, :, :]
  X_train_expanded[:, 1, :, :] = X_train[:, 0, :, :]
  X_train_expanded[:, 2, :, :] = X_train[:, 0, :, :]

  X_val_expanded = np.zeros((X_val.shape[0], 3, X_val.shape[2], X_val.shape[3]))
  X_val_expanded[:, 0, :, :] = X_val[:, 0, :, :]
  X_val_expanded[:, 1, :, :] = X_val[:, 0, :, :]
  X_val_expanded[:, 2, :, :] = X_val[:, 0, :, :]

  X_train = X_train_expanded
  X_val = X_val_expanded

  # Convert class vectors to binary class matrices.
  Y_train = np_utils.to_categorical(train[1], nb_classes)
  Y_val = np_utils.to_categorical(val[1], nb_classes)

  return X_train, X_val, Y_train, Y_val

def augment_data(X_train):
  '''
  Perform data augmentation on the input data and return an ImageDataGenerator
  object.
  '''
  datagen = ImageDataGenerator(
    featurewise_center=True,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=True,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

  datagen.fit(X_train)
  return datagen

def train_model(model, train, val, nb_classes, params):
  '''
  Train the CNN and output results to a file.
  '''
  X_train, X_val, Y_train, Y_val = prepare_data(train, val)
  datagen = augment_data(X_train)

  nb_epoch = params.get('nb_epoch', DEFAULT_NB_EPOCH)

  model.compile(loss='categorical_crossentropy', optimizer='adadelta')

  print('X_train shape:', X_train.shape)
  print('Y_train shape:', Y_train.shape)
  print('X_val shape:', X_val.shape)
  print('Y_val shape:', Y_val.shape)
  print(X_train.shape[0], 'train samples')
  print(X_val.shape[0], 'val samples')

  # Fit the model on the batches generated by datagen.flow().
  history = History()
  model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
            samples_per_epoch=X_train.shape[0],
            nb_epoch=nb_epoch, show_accuracy=True,
            validation_data=(X_val, Y_val),
            nb_worker=1, callbacks=[history], verbose=2)

  # Print the results to the console.
  for key in history.history:
      print(key, history.history[key])

  final_acc = history.history["acc"][-1] 

  # Write the results to a file
  out_location = params['output_dir']
  out_file = out_location + 'mode_' + str(params['mode']) + '_' + str(final_acc) + "_transfer.txt"

  print('Writing to file:', out_file)
  f = open(out_file, "w")
  for key in history.history:
      f.write(key + ": " + str(history.history[key]) + "\n")

  # Print parameters to the file.
  for key in params:
      f.write(key + ": " + str(params[key]) + "\n")

  f.close()

def get_part_vgg_model(weights_path, trainable=False):
  '''
  Return a model that includes the first few layers from VGG with linear
  layers added on top.

  Args:
    weights_path: path to file containing VGG16 weights.
    trainable: boolean representing whether the VGG layers should be
      trainable or fixed.
  '''

  # Keep 3 layers of convolution from VGG16.
  model = Sequential()
  model.add(ZeroPadding2D((1, 1), input_shape=(3, img_cols, img_rows)))
  model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
  model.add(MaxPooling2D((2, 2), strides=(2, 2)))

  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))

  # Load the weights of the VGG16 network.
  assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
  f = h5py.File(weights_path)
  for k in range(f.attrs['nb_layers']):
    if k >= len(model.layers):
      # We don't look at the last (fully connected) layers in the savefile.
      break
    g = f['layer_{}'.format(k)]
    weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    model.layers[k].set_weights(weights)

    # Freeze these layers if trainable = False.
    model.layers[k].trainable = trainable
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

def get_full_vgg_model(weights_path, trainable=False):
  '''
  Return a model that includes the all layers from VGG with linear
  layers added on top.

  Args:
    weights_path: path to file containing VGG16 weights.
    trainable: boolean representing whether the VGG layers should be
      trainable or fixed.
  '''

  # Model replicates VGG16 architecture:
  # https://gist.github.com/jimmie33/27c1c0a7736ba66c2395
  model = Sequential()
  model.add(ZeroPadding2D((1, 1), input_shape=(3, img_cols, img_rows)))
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

  # Load the weights of the VGG16 network.
  assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
  f = h5py.File(weights_path)
  for k in range(f.attrs['nb_layers']):
    if k >= len(model.layers):
      # We don't look at the last (fully connected) layers in the savefile.
      break
    g = f['layer_{}'.format(k)]
    weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    model.layers[k].set_weights(weights)

    # Freeze these layers if trainable = False.
    model.layers[k].trainable = trainable
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


def load_data(train_data_file, val_data_file, num_train, num_val):
  # Use a dummy CNN to load data.
  cnn = CNN(params={}, verbose=True)
  cnn.load_data(train_data_file, val_data_file, num_train=num_train, num_val=num_val)
  X_train, y_train = cnn.X_train, cnn.y_train
  X_val, y_val = cnn.X_val, cnn.y_val
  print('Loaded input data.')
  return X_train, y_train, X_val, y_val

def run_cnn():
  # Define file paths.
  train_data_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), DATA_DIR, 'train.csv')
  val_data_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), DATA_DIR, 'val.csv')
  weights_data_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), DATA_DIR, 'vgg16_weights.h5')

  # Load data.
  train_data_file, val_data_file, num_train, num_val, mode, params = parse_args()
  X_train, y_train, X_val, y_val = load_data(train_data_file, val_data_file, num_train, num_val)

  logging.info('Running with mode {}'.format(mode))
  logging.info('Running with params: {}'.format(params))

  # Define appropriate model depending on input mode.
  model = None
  if mode == FULL_FROZEN_VGG:
    model = get_full_vgg_model(weights_data_file, trainable=False)
  elif mode == PART_FROZEN_VGG:
    model = get_part_vgg_model(weights_data_file, trainable=False)
  elif mode == FULL_TRAINABLE_VGG:
    model = get_full_vgg_model(weights_data_file, trainable=True)

  # Train and evaluate model.
  train_model(model, (X_train, y_train), (X_val, y_val), nb_classes, params)

def main():
  logging.basicConfig(format="[%(name)s %(asctime)s]\t%(msg)s", level=logging.INFO)

  run_cnn()

if __name__ == '__main__':
  main()
