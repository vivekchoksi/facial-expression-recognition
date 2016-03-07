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

IMG_DIM = 48
DATA_DIR = 'data'
DEFAULT_LR = 1e-4
DEFAULT_REG = 0
DEFAULT_NB_EPOCH = 2
DEFAULT_LAYER_SIZE_1 = 32
DEFAULT_LAYER_SIZE_2 = 64
DEFAULT_DROPOUT = 0.25
DEFAULT_OUT_DIR = '../outputs/'
DEFAULT_DEPTH1 = 1
DEFAULT_DEPTH2 = 2
DEFAULT_FRAC_POOLING = False
DEFAULT_SAVE_WEIGHTS = False

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
  parser.add_argument('-l', default = DEFAULT_LR, help = 'learning rate', type=float)
  parser.add_argument('-r', default = DEFAULT_REG, help = 'regularization', type=float)
  parser.add_argument('-e', default = DEFAULT_NB_EPOCH, help = 'number of epochs', type=int)
  parser.add_argument('-nt', default = default_num_train, help = 'number of training examples to use', type=int)
  parser.add_argument('-nv', default = default_num_val, help = 'number of validation examples to use', type=int)
  parser.add_argument('-nf1', default = DEFAULT_LAYER_SIZE_1, help = 'number of filters in the first set of layers', type=int)
  parser.add_argument('-nf2', default = DEFAULT_LAYER_SIZE_2, help = 'number of filters in the second set of layers', type=int)
  parser.add_argument('-d', default = DEFAULT_DROPOUT, help = 'dropout rate', type=float)
  parser.add_argument('-o', default = DEFAULT_OUT_DIR, help = 'location of output directory')
  parser.add_argument('-dp1', default = DEFAULT_DEPTH1, help = 'depth of first set of network', type=int)
  parser.add_argument('-dp2', default = DEFAULT_DEPTH2, help = 'depth of second set of network', type=int)
  parser.add_argument('-frac', default = DEFAULT_FRAC_POOLING, help = 'pass to use fractional max pooling', dest='frac', action = 'store_true')
  parser.add_argument('-save', action='store_true', default = DEFAULT_SAVE_WEIGHTS, help = 'whether to visualize filters')

  args = parser.parse_args()
  params = {
    'lr': args.l, 'reg': args.r, 'nb_epoch': args.e, 'nb_filters_1': args.nf1, 'nb_filters_2': args.nf2,
    'dropout': args.d, 'output_dir': args.o, 'depth1': args.dp1, 'depth2':args.dp2,
    'save_weights': args.save, 'fractional_pooling': args.frac
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

  def train(self):
    """
    Train the CNN model.

    """

    batch_size = 32
    nb_classes = 7

    nb_epoch = self.params.get('nb_epoch', DEFAULT_NB_EPOCH)
    lr = self.params.get('lr', DEFAULT_REG)
    reg = self.params.get('reg', DEFAULT_REG)
    nb_filters_1 = self.params.get('nb_filters_1', DEFAULT_LAYER_SIZE_1)
    nb_filters_2 = self.params.get('nb_filters_2', DEFAULT_LAYER_SIZE_2)
    dropout = self.params.get('dropout', DEFAULT_DROPOUT)
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
    conv_counter += 1
    model.add(Activation('relu'))

    for i in xrange(depth1):
        model.add(Convolution2D(nb_filters_1, 3, 3, init=weight_init, border_mode='same', W_regularizer=l2(reg),
          name='conv_%d' % (conv_counter)))
        conv_counter += 1
        model.add(Activation('relu'))
        model.add(Convolution2D(nb_filters_1, 3, 3, init=weight_init, border_mode='same', W_regularizer=l2(reg),
          name='conv_%d' % (conv_counter)))
        conv_counter += 1
        model.add(Activation('relu'))
        if fractional_pooling:
            model.add(FractionalMaxPooling2D(pool_size=(np.sqrt(2), np.sqrt(2))))
        else:
            model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout))

    for i in xrange(depth2):
        model.add(Convolution2D(nb_filters_2, 3, 3, border_mode='same', init=weight_init, W_regularizer=l2(reg),
          name='conv_%d' % (conv_counter)))
        conv_counter += 1
        model.add(Activation('relu'))
        model.add(Convolution2D(nb_filters_2, 3, 3, border_mode='same', init=weight_init, W_regularizer=l2(reg),
          name='conv_%d' % (conv_counter)))
        conv_counter += 1
        model.add(Activation('relu'))
        if fractional_pooling:
            model.add(FractionalMaxPooling2D(pool_size=(np.sqrt(2), np.sqrt(2))))
        else:
            model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout))

    model.add(Flatten(input_shape=(img_rows, img_cols)))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(nb_classes, init=weight_init))
    model.add(Activation('softmax'))

    # Use the Adam update rule.
    adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model.compile(loss='categorical_crossentropy', optimizer=adam)

    X_train = X_train.astype('float64')
    X_val = X_val.astype('float64')
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
    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
              samples_per_epoch=X_train.shape[0],
              nb_epoch=nb_epoch, show_accuracy=True,
              validation_data=(X_val, Y_val),
              nb_worker=1, callbacks=[history], verbose=2)

    # Print the results to the console
    for key in history.history:
        print(key, history.history[key])

    final_acc = history.history["acc"][-1] 

    # Print the results to a file
    out_location = self.params['output_dir']
    out_file = out_location + str(final_acc) + "_out.txt"
    f = open(out_file, "w")
    for key in history.history:
        f.write(key + ": " + str(history.history[key]) + "\n")

    # print parameters to the file
    for key in self.params:
        f.write(key + ": " + str(self.params[key]) + "\n")

    f.close()

    if save_weights:
      weights_path = out_location + str(final_acc) + '_weights.h5'
      print('Writing weights to file:', weights_path)
      model.save_weights(weights_path, overwrite=True)

  def visualize_filters(self, weights_path, layer_name):
    '''
    Output an image visualizing filters at a particular layer in the CNN.
    The generated images are input images that maximize the activations
    of filters at a particular layer in the CNN.

    Args:
      weights_path: path to the .h5 file storing the model's weights.
      layer_name: name of the layer (e.g. conv_3) to visualize.
    '''

    out_location = self.params['output_dir']
    output_image = out_location + layer_name + "_filters.png"

    # Load an empty (uncompiled) model from which to generate
    # visualizations.
    if not self.empty_model:
      self.empty_model = load_custom_cnn(self.params, weights_path)

    # Specify how many filters, of what size, at which layer, to which output
    # path to generate. See the docstring for generate_filter_visualizations.
    generate_filter_visualizations(self.empty_model, layer_name, output_image,
      num_channels=1, img_width=IMG_DIM, img_height=IMG_DIM, nb_filters=10, filter_grid_length=1)

  def visualize_class(self, weights_path):
    out_location = self.params['output_dir']
    output_image = out_location + str(0) + "_class.png"

    # Load an empty (uncompiled) model from which to generate
    # visualizations.
    if not self.empty_model:
      self.empty_model = load_custom_cnn(self.params, weights_path)

    # Specify how many filters, of what size, at which layer, to which output
    # path to generate. See the docstring for generate_filter_visualizations.
    generate_class_visualizations(self.empty_model, '', output_image,
      num_channels=1, img_width=IMG_DIM, img_height=IMG_DIM, nb_filters=10, filter_grid_length=1)


def main():
  # Set up logging.
  logging.basicConfig(format="[%(name)s %(asctime)s]\t%(msg)s", level=logging.INFO)

  train_data_file, val_data_file, num_train, num_val, params = parse_args()

  cnn = CNN(params)
  cnn.load_data(train_data_file, val_data_file, num_train=num_train, num_val=num_val)
  cnn.train()

  # weights_path = '../outputs/custom_weights.h5'
  # cnn.visualize_filters(weights_path, 'conv_3')
  # cnn.visualize_class(weights_path)

if __name__ == '__main__':
  main()
