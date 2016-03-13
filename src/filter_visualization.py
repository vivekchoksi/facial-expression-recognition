'''Visualization of the filters of a model via gradient ascent in input space.

Script adapted from fchollet:
https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py

python src/filter_visualization.py -l 0.001 -d 0 -r 1e-6 -nf1 32 -nf2 64 -dp1 1 -dp2 2 -o ./
'''

from __future__ import print_function
from scipy.misc import imsave
import numpy as np
import time
import os
import h5py
import pdb
import argparse

from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.regularizers import l2, activity_l2
from pool import FractionalMaxPooling2D
from keras.models import Sequential
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras import backend as K

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
  parser = argparse.ArgumentParser()
  parser.add_argument('-l', default = DEFAULT_LR, help = 'learning rate', type=float)
  parser.add_argument('-r', default = DEFAULT_REG, help = 'regularization', type=float)
  parser.add_argument('-nf1', default = DEFAULT_LAYER_SIZE_1, help = 'number of filters in the first set of layers', type=int)
  parser.add_argument('-nf2', default = DEFAULT_LAYER_SIZE_2, help = 'number of filters in the second set of layers', type=int)
  parser.add_argument('-d', default = DEFAULT_DROPOUT, help = 'dropout rate', type=float)
  parser.add_argument('-o', default = DEFAULT_OUT_DIR, help = 'location of output directory')
  parser.add_argument('-dp1', default = DEFAULT_DEPTH1, help = 'depth of first set of network', type=int)
  parser.add_argument('-dp2', default = DEFAULT_DEPTH2, help = 'depth of second set of network', type=int)
  parser.add_argument('-frac', default = DEFAULT_FRAC_POOLING, help = 'pass to use fractional max pooling', dest='frac', action = 'store_true')

  args = parser.parse_args()
  params = {
    'lr': args.l, 'reg': args.r, 'nb_filters_1': args.nf1, 'nb_filters_2': args.nf2,
    'dropout': args.d, 'output_dir': args.o, 'depth1': args.dp1, 'depth2':args.dp2, 'fractional_pooling': args.frac
  }

  return params

def generate_class_visualizations(model, out_class, out_path, img_width=48, img_height=48, num_channels=1):
    '''
    Visualize the classifications of a model by generating an image that maximizes the
    score of a particular output class. Note: this method is not yet working.

    Args:
        model: Keras model with weights already loaded or trained.
        out_class: the class (0-6) to visualize.
        out_path: the output path for the filter visualization image.
        img_width: width of the filters to generate.
        img_height: height of the filters to generate.
    '''
    print('Starting class visualization...')

    # this will contain our generated image
    input_img = K.placeholder((1, num_channels, img_width, img_height))

    # build the network with our input_img as input
    model.layers[0].input = input_img

    start_time = time.time()

    # Build a loss function that maximizes the
    # last layer's output (softmax activation).
    layer_output = model.layers[-1].get_output()

    # NOTE: Not sure if this is correct.
    # Score for the class.
    loss = K.mean(layer_output[0][int(out_class)])

    # we compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]

    # normalization trick: we normalize the gradient
    grads = normalize(grads)

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])

    # step size for gradient ascent
    step = 1.

    # we start from a gray image with some random noise
    input_img_data = np.random.random((1, num_channels, img_width, img_height)) * 20 + 128.

    # we run gradient ascent for many iterations
    for i in range(20):
        loss_value, grads_value = iterate([input_img_data])

        # NOTE: Not sure how best to set step)val.
        step_val = np.abs(1 / np.max(grads_value))
        # pdb.set_trace()
        input_img_data += grads_value * step_val

        print('Current loss value:', loss_value)

    # decode the resulting input image
    if loss_value > 0:
        img = deprocess_image(input_img_data[0])
        end_time = time.time()
        print('Processed in %ds' % (end_time - start_time))

        img_expanded = np.zeros((3, img_width, img_height))
        img_expanded[0, :, :] = img[:, :, 0]
        img_expanded[1, :, :] = img[:, :, 0]
        img_expanded[2, :, :] = img[:, :, 0]


        # save the result to disk
        print('Saving image as file:', out_path)
        imsave(out_path, img_expanded)
    else:
        print('Loss value is < 0.')


def generate_filter_visualizations(model, layer_name, out_path, img_width=48, img_height=48,
    nb_filters=50, filter_grid_length=2, num_channels=1):
    '''
    Visualize the filters generated by a model at a particular layer, and
    save the output visualization as an image.

    Args:
        model: Keras model with weights already loaded or trained.
        layer_name: the name of the layer in the model for which to visualize filters.
        out_path: the output path for the filter visualization image.
        img_width: width of the filters to generate.
        img_height: height of the filters to generate.
        nb_filters: number of filters to consider (not the number of filters visualized).
        filter_grid_length: side length of the output square grid of filter visualizations.
    '''
    print('Starting filter visualization for layer:', layer_name)

    # this will contain our generated image
    input_img = K.placeholder((1, num_channels, img_width, img_height))

    # build the VGG16 network with our input_img as input
    model.layers[0].input = input_img

    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    kept_filters = []
    for filter_index in range(0, nb_filters):
        # we only scan through the first 200 filters,
        # but there are actually 512 of them
        print('Processing filter %d' % filter_index)
        start_time = time.time()

        # we build a loss function that maximizes the activation
        # of the nth filter of the layer considered
        layer_output = layer_dict[layer_name].get_output()
        loss = K.mean(layer_output[:, filter_index, :, :])

        # we compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_img)[0]

        # normalization trick: we normalize the gradient
        grads = normalize(grads)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])

        # step size for gradient ascent
        step = 2.

        # we start from a gray image with some random noise
        input_img_data = np.random.random((1, num_channels, img_width, img_height)) * 20 + 128.

        # we run gradient ascent for 20 steps
        for i in range(20):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * step

            print('Current loss value:', loss_value)
            # if loss_value <= 0.:
            #     # some filters get stuck to 0, we can skip them
            #     break

        # decode the resulting input image
        # if loss_value > 0:
        img = deprocess_image(input_img_data[0])
        kept_filters.append((img, loss_value))

        end_time = time.time()
        print('Filter %d processed in %ds' % (filter_index, end_time - start_time))

    # we will stich the best n^2 filters on an n x n grid.
    n = filter_grid_length

    # the filters that have the highest loss are assumed to be better-looking.
    # we will only keep the top n^2 filters.
    kept_filters.sort(key=lambda x: x[1], reverse=True)
    kept_filters = kept_filters[:n * n]

    # build a black picture with enough space for
    # our n x n filters of size img_width * img_height, with a 5px margin in between
    margin = 5
    width = n * img_width + (n - 1) * margin
    height = n * img_height + (n - 1) * margin
    stitched_filters = np.zeros((width, height, 3)) + 255.

    # fill the picture with our saved filters
    for i in range(n):
        for j in range(n):
            img, loss = kept_filters[i * n + j]
            stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
                             (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img

    # save the result to disk
    print('Saving image as file:', out_path)
    imsave(out_path, stitched_filters)


# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def load_custom_cnn(params, weights_path):
    '''
    Return a Keras Sequential model constructed according to
    the input parameters and loaded with weights from weights_path.
    '''

    reg = params.get('reg')
    nb_filters_1 = params.get('nb_filters_1')
    nb_filters_2 = params.get('nb_filters_2')
    dropout = params.get('dropout')
    depth1 = params.get('depth1')
    depth2 = params.get('depth2')
    img_channels = 1

    model = Sequential()

    weight_init = 'he_normal'

    # Keep track of which convolutional layer we are at.
    conv_counter = 1

    model.add(Convolution2D(nb_filters_1, 3, 3, init=weight_init, border_mode='same',
      name='conv_%d' % (conv_counter),
      input_shape=(1, 48, 48)))
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
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(FractionalMaxPooling2D(pool_size=(np.sqrt(2), np.sqrt(2))))
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
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout))

    model.add(Flatten(input_shape=(48, 48)))

    # Add 3 fully connected layers.
    dense_sizes = [512, 256, 128]
    for idx, dense_size in enumerate(dense_sizes):
      model.add(Dense(dense_size))
      model.add(Activation('relu'))

      # Use dropout2 only for the final dense layer.
      if idx == len(dense_sizes) - 1:
        model.add(Dropout(0.2))
      else:
        model.add(Dropout(0))

    model.add(Dense(7, init=weight_init))
    model.add(Activation('softmax'))

    assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')

    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    return model


def load_vgg():
    # dimensions of the generated pictures for each filter.
    img_width = 64
    img_height = 64

    # path to the model weights file.
    weights_path = 'data/vgg16_weights.h5'

    # this will contain our generated image
    # input_img = K.placeholder((1, 3, img_width, img_height))

    # build the VGG16 network with our input_img as input
    first_layer = ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height))
    # first_layer.input = input_img

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
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

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
    f.close()
    print('Model loaded.')

    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    return model


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def visualize_vgg():
    for layer in ['conv1_1', 'conv2_1', 'conv2_2', 'conv5_1']:
        model = load_vgg()
        generate_filter_visualizations(model, layer,
            'outputs/filters_vgg' + layer + '.png',
            img_width=64, img_height=64, nb_filters=20,
            filter_grid_length=2, num_channels=3)


def visualize_filters_custom_model(params, weights_path, layer_name):
    '''
    Output an image visualizing filters at a particular layer in the CNN.
    The generated images are input images that maximize the activations
    of filters at a particular layer in the CNN.

    Args:
        params: parameters of the CNN to load.
        weights_path: path to the .h5 file storing the model's weights.
        layer_name: name of the layer (e.g. conv_3) to visualize.
    '''

    out_location = params['output_dir']
    output_image = out_location + layer_name + "_filters.png"

    # Load an empty (uncompiled) model from which to generate
    # visualizations.
    empty_model = load_custom_cnn(params, weights_path)

    # Specify how many filters, of what size, at which layer, to which output
    # path to generate. See the docstring for generate_filter_visualizations.
    # try:
    generate_filter_visualizations(empty_model, layer_name, output_image,
        num_channels=1, img_width=48, img_height=48,
        nb_filters=32, filter_grid_length=2)
    # except Exception:
        # print('Could not visualize weights for layer:', layer_name)


def main():
    params = parse_args()
    weights_path = 'final_outputs/lr=0.0005_depth2=2_fractional_pooling=False_use_batchnorm=False_dropout1=0.0_dropout2=0.2.hdf5'
    for layer_index in xrange(8, 12):
        visualize_filters_custom_model(params, weights_path, 'conv_%d' % layer_index)

if __name__ == '__main__':
    main()
