#!/usr/bin/python

# File: cnn_baseline.py
# ---------------------
# Train a baseline deep convolutional neural network on a sample of the data
# from the Facial Expression Recognition Challenge.
#
# Run this script from the scripts/ directory.
# python ../src/cnn_baseline.py -l learning_rate -r regularization ...
#
# Code adapted from: 
# https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py
# GPU run command:
#  THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnn_baseline.py

from __future__ import print_function
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.callbacks import History
from keras.regularizers import l2, activity_l2
from pool import FractionalMaxPooling2D
from filter_visualization import generate_filter_visualizations, generate_class_visualizations, load_custom_cnn

import os
import numpy as np
import argparse
import logging
import h5py

IMG_DIM = 48
DATA_DIR = 'data'
DEFAULT_LR = 1e-4
DEFAULT_REG = 0
DEFAULT_NB_EPOCH = 2
DEFAULT_LAYER_SIZE_1 = 32
DEFAULT_LAYER_SIZE_2 = 64
DEFAULT_DROPOUT1 = 0.1
DEFAULT_DROPOUT2 = 0.25
DEFAULT_OUT_DIR = '../outputs/'
DEFAULT_DEPTH1 = 1
DEFAULT_DEPTH2 = 2
DEFAULT_FRAC_POOLING = False
DEFAULT_SAVE_WEIGHTS = False
DEFAULT_USE_BATCHNORM = False

def parse_args():
  """
  Parses the command line input.

  """
  train_data_file_default = os.path.join(os.path.dirname(os.path.dirname(__file__)), DATA_DIR, 'train.csv')
  val_data_file_default = os.path.join(os.path.dirname(os.path.dirname(__file__)), DATA_DIR, 'val.csv')
  default_num_train = 28709
  default_num_val = 3589

  parser = argparse.ArgumentParser()
  parser.add_argument('-td', default = train_data_file_default, help = 'training data file')
  parser.add_argument('-vd', default = val_data_file_default, help = 'validation data file')
  parser.add_argument('-w', default = '', help = 'path to weights file from which to load model')
  parser.add_argument('-l', default = DEFAULT_LR, help = 'learning rate', type=float)
  parser.add_argument('-r', default = DEFAULT_REG, help = 'regularization', type=float)
  parser.add_argument('-e', default = DEFAULT_NB_EPOCH, help = 'number of epochs', type=int)
  parser.add_argument('-nt', default = default_num_train, help = 'number of training examples to use', type=int)
  parser.add_argument('-nv', default = default_num_val, help = 'number of validation examples to use', type=int)
  parser.add_argument('-nf1', default = DEFAULT_LAYER_SIZE_1, help = 'number of filters in the first set of layers', type=int)
  parser.add_argument('-nf2', default = DEFAULT_LAYER_SIZE_2, help = 'number of filters in the second set of layers', type=int)
  parser.add_argument('-d1', default = DEFAULT_DROPOUT1, help = 'dropout rate for first conv layer block', type=float)
  parser.add_argument('-d2', default = DEFAULT_DROPOUT2, help = 'dropout rate for second conv layer block', type=float)
  parser.add_argument('-o', default = DEFAULT_OUT_DIR, help = 'location of output directory')
  parser.add_argument('-dp1', default = DEFAULT_DEPTH1, help = 'depth of first set of network', type=int)
  parser.add_argument('-dp2', default = DEFAULT_DEPTH2, help = 'depth of second set of network', type=int)
  parser.add_argument('-frac', default = DEFAULT_FRAC_POOLING, help = 'pass to use fractional max pooling', dest='frac', action = 'store_true')
  parser.add_argument('-save', action='store_true', default = DEFAULT_SAVE_WEIGHTS, help = 'whether to save model weights at each epoch')
  parser.add_argument('-bn', action='store_true', default = DEFAULT_USE_BATCHNORM, help = 'whether to use batchnorm')

  args = parser.parse_args()
  params = {
    'lr': args.l, 'reg': args.r, 'nb_epoch': args.e, 'nb_filters_1': args.nf1, 'nb_filters_2': args.nf2,
    'dropout1': args.d1, 'dropout2': args.d2, 'output_dir': args.o, 'depth1': args.dp1, 'depth2':args.dp2,
    'use_batchnorm': args.bn, 'save_weights': args.save, 'fractional_pooling': args.frac,
    'weights_path': args.w
  }

  return args.td, args.vd, args.nt, args.nv, params

class CNN:
  """
  Convolutional Neural Network model.

  """

  def __init__(self, params={}, verbose=True):
    """
    Initialize the CNN model with a set of parameters.

    Args:
      params: a dictionary containing values of the models' parameters.

    """
    self.verbose = verbose
    self.params = params

    # An empty (uncompiled and untrained) model may be used for visualizations.
    self.empty_model = None

    logging.info('Initialized with params: {}'.format(params))

  def load_data(self, train_data_file, val_data_file, num_train=None, num_val=None):
    """
    Load training and validation data from files.

    Args:
      train_data_file: path to the file containing training examples.
      val_data_file: path to the file containing validation examples.

    """
    logging.info('Reading {} training examples from {}...'.format(num_train, train_data_file))
    self.X_train, self.y_train = self._load_data_from_file(train_data_file, num_train)
    logging.info('Reading {} validation examples from {}...'.format(num_val, val_data_file))
    self.X_val, self.y_val = self._load_data_from_file(val_data_file, num_val)

  def _load_data_from_file(self, filename, num_examples=None):
    if num_examples is None:
      num_examples = sum(1 for line in open(filename))

    X_data = np.zeros((num_examples, 1, 48, 48))
    y_data = np.zeros((num_examples, 1))
    with open(filename, 'r') as f:
      for i, line in enumerate(f):
        label, pixels, usage = line.split(',')
        # Reformat image from array of pixels to square matrix.
        pixels = np.array([int(num) for num in pixels.split(' ')]).reshape((IMG_DIM, IMG_DIM))
        X_data[i][0] = pixels
        y_data[i][0] = int(label)

        if num_examples is not None and i == num_examples - 1:
          return X_data, y_data

    return X_data, y_data

  def _add_batchnorm_layer(self, model):
    '''
    Add a batch normalization layer to the model if the params specify use batchnorm.
    '''
    if self.params.get('use_batchnorm', DEFAULT_USE_BATCHNORM):
      model.add(BatchNormalization())

  def _get_file_prefix(self):
    file_prefix = ''
    file_prefix += self.params['output_dir']
    param_names = ['lr', 'depth2', 'fractional_pooling', 'use_batchnorm', 'dropout1', 'dropout2']
    for idx, param_name in enumerate(param_names):
      file_prefix += param_name + '=' + str(self.params[param_name])
      if idx < len(param_names) - 1:
        file_prefix += '_'

    return file_prefix

  def _load_weights(self, model):
    weights_path = self.params['weights_path']
    logging.info('Loading weights from file: {}'.format(weights_path))
    assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
      if k >= len(model.layers):
        # We don't look at the last (fully connected) layers in the savefile.
        break
      g = f['layer_{}'.format(k)]
      weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
      model.layers[k].set_weights(weights)

    f.close()
    logging.info('Weights loaded.')

  def train(self):
    """
    Train the CNN model.

    """

    batch_size = 128
    nb_classes = 7

    nb_epoch = self.params.get('nb_epoch', DEFAULT_NB_EPOCH)
    lr = self.params.get('lr', DEFAULT_REG)
    reg = self.params.get('reg', DEFAULT_REG)
    nb_filters_1 = self.params.get('nb_filters_1', DEFAULT_LAYER_SIZE_1)
    nb_filters_2 = self.params.get('nb_filters_2', DEFAULT_LAYER_SIZE_2)
    dropout1 = self.params.get('dropout1', DEFAULT_DROPOUT1)
    dropout2 = self.params.get('dropout2', DEFAULT_DROPOUT2)
    depth1 = self.params.get('depth1', DEFAULT_DEPTH1)
    depth2 = self.params.get('depth2', DEFAULT_DEPTH2)
    fractional_pooling = self.params.get('fractional_pooling', DEFAULT_FRAC_POOLING)
    if fractional_pooling:
        print("Using fractional max pooling... \n")
    else:
        print("Using standard max pooling... \n")

    save_weights = self.params.get('save_weights', DEFAULT_SAVE_WEIGHTS)

    X_train, y_train = self.X_train, self.y_train
    X_val, y_val = self.X_val, self.y_val

    # Input image dimensions.
    img_rows, img_cols = IMG_DIM, IMG_DIM

    img_channels = 1

    # Convert class vectors to binary class matrices.
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_val = np_utils.to_categorical(y_val, nb_classes)

    model = Sequential()

    weight_init = 'he_normal'

    # Keep track of which convolutional layer we are at.
    conv_counter = 1

    model.add(Convolution2D(nb_filters_1, 3, 3, init=weight_init, border_mode='same',
      name='conv_%d' % (conv_counter),
      input_shape=(img_channels, img_rows, img_cols)))
    self._add_batchnorm_layer(model)
    conv_counter += 1
    model.add(Activation('relu'))

    for i in xrange(depth1):
        model.add(Convolution2D(nb_filters_1, 3, 3, init=weight_init, border_mode='same', W_regularizer=l2(reg),
          name='conv_%d' % (conv_counter)))
        self._add_batchnorm_layer(model)
        conv_counter += 1
        model.add(Activation('relu'))
        model.add(Convolution2D(nb_filters_1, 3, 3, init=weight_init, border_mode='same', W_regularizer=l2(reg),
          name='conv_%d' % (conv_counter)))
        self._add_batchnorm_layer(model)
        conv_counter += 1
        model.add(Activation('relu'))
        if fractional_pooling:
            model.add(FractionalMaxPooling2D(pool_size=(np.sqrt(2), np.sqrt(2))))
        else:
            model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout1))

    for i in xrange(depth2):
        model.add(Convolution2D(nb_filters_2, 3, 3, border_mode='same', init=weight_init, W_regularizer=l2(reg),
          name='conv_%d' % (conv_counter)))
        self._add_batchnorm_layer(model)
        conv_counter += 1
        model.add(Activation('relu'))
        model.add(Convolution2D(nb_filters_2, 3, 3, border_mode='same', init=weight_init, W_regularizer=l2(reg),
          name='conv_%d' % (conv_counter)))
        self._add_batchnorm_layer(model)
        conv_counter += 1
        model.add(Activation('relu'))
        if fractional_pooling:
            model.add(FractionalMaxPooling2D(pool_size=(np.sqrt(2), np.sqrt(2))))
        else:
            model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout1))

    model.add(Flatten(input_shape=(img_rows, img_cols)))

    # Add 3 fully connected layers.
    dense_sizes = [512, 256, 128]
    for idx, dense_size in enumerate(dense_sizes):
      model.add(Dense(dense_size))
      self._add_batchnorm_layer(model)
      model.add(Activation('relu'))

      # Use dropout2 only for the final dense layer.
      if idx == len(dense_sizes) - 1:
        model.add(Dropout(dropout2))
      else:
        model.add(Dropout(dropout1))

    model.add(Dense(nb_classes, init=weight_init))
    model.add(Activation('softmax'))

    if self.params['weights_path'] is not '':
      self._load_weights(model)

    # Use the Adam update rule.
    adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    logging.info('Starting compilation...')
    model.compile(loss='categorical_crossentropy', optimizer=adam)
    logging.info('Finished compilation.')

    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    X_train /= 255
    X_val /= 255

    # Settings for preprocessing.
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

    # Fit the model on the batches generated by datagen.flow().
    history = History()
    callbacks = [history]

    if save_weights:
      file_name = self._get_file_prefix() + '.hdf5'
      checkpointer = ModelCheckpoint(filepath=file_name, save_best_only=True, mode='auto', verbose=1, monitor="val_acc")
      callbacks.append(checkpointer)

    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
              samples_per_epoch=X_train.shape[0],
              nb_epoch=nb_epoch, show_accuracy=True,
              validation_data=(X_val, Y_val),
              nb_worker=1, callbacks=callbacks, verbose=2)

    # Print the results to the console
    for key in history.history:
        print(key, history.history[key])

    final_acc = history.history["acc"][-1] 

    # Print the results to a file
    out_file = self._get_file_prefix() + '_' + str(final_acc) + "_out.txt"
    f = open(out_file, "w")
    for key in history.history:
        f.write(key + ": " + str(history.history[key]) + "\n")

    # print parameters to the file
    for key in self.params:
        f.write(key + ": " + str(self.params[key]) + "\n")

    f.close()


def main():
  # Set up logging.
  logging.basicConfig(format="[%(name)s %(asctime)s]\t%(msg)s", level=logging.INFO)

  train_data_file, val_data_file, num_train, num_val, params = parse_args()

  cnn = CNN(params)
  cnn.load_data(train_data_file, val_data_file, num_train=num_train, num_val=num_val)
  cnn.train()

if __name__ == '__main__':
  main()
