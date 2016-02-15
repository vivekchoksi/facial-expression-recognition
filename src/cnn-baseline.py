# File: cnn-baseline.py
# ---------------------
# Train a baseline deep convolutional neural network on a sample of the data
# from the Facial Expression Recognition Challenge.
# 
# python cnn-baseline.py -l learning_rate -r regularization ...
#
# Code adapted from: 
# https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py
# GPU run command:
#  THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnn-baseline.py

from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.callbacks import History
from optparse import OptionParser

import os
import numpy as np
import argparse

IMG_DIM = 48
DATA_DIR = 'data'
DEFAULT_LR = 1e-8
DEFAULT_REG = 1e-6
DEFAULT_NB_EPOCH = 3

def parse_args():
  """
  Parses the command line input.

  """
  train_data_file_default = os.path.join(os.path.dirname(os.path.dirname(__file__)), DATA_DIR, 'train-small.csv')
  val_data_file_default = os.path.join(os.path.dirname(os.path.dirname(__file__)), DATA_DIR, 'train-small.csv')

  parser = argparse.ArgumentParser()
  parser.add_argument('-td', default = train_data_file_default, help = 'training data file')
  parser.add_argument('-vd', default = val_data_file_default, help = 'validation data file')
  parser.add_argument('-l', default = DEFAULT_LR, help = 'learning rate', type=float)
  parser.add_argument('-r', default = DEFAULT_REG, help = 'regularization', type=float)
  parser.add_argument('-e', default = DEFAULT_NB_EPOCH, help = 'number of epochs', type=int)

  args = parser.parse_args()
  params = {'lr': args.l, 'reg': args.r, 'nb_epoch': args.e}
  return args.td, args.vd, params

class CNN:
  """
  Convolutional Neural Network model.

  """

  def __init__(self, params={}):
    """
    Initialize the CNN model with a set of parameters.

    Args:
      params: a dictionary containing values of the models' parameters.

    """
    self.params = params

  def load_data(self, train_data_file, val_data_file, num_train=None, num_val=None):
    """
    Load training and validation data from files.

    Args:
        train_data_file: path to the file containing training examples.
        val_data_file: path to the file containing validation examples.

    """
    self.X_train, self.y_train = self._load_data_from_file(train_data_file, num_train)
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

    model.add(Convolution2D(32, 3, 3, init=weight_init, border_mode='same',
                input_shape=(img_channels, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3, init=weight_init))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same', init=weight_init))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, init=weight_init))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
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

    print(history.history)

    # Print the results to a file
    out_file = str(lr) + "_out.txt"
    f = open(out_file, "w")
    f.write(str(history.history))
    f.close()

def main():
  train_data_file, val_data_file, params = parse_args()

  cnn = CNN(params)
  cnn.load_data(train_data_file, val_data_file, num_train=800, num_val=200)
  cnn.train() 

if __name__ == '__main__':
  main()
