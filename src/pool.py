from keras.layers.convolutional import *
from keras.layers.convolutional import _Pooling2D
from keras import backend as K
import numpy as np
import theano
import theano.tensor as T

class _FractionalPooling2D(_Pooling2D):
    '''Abstract class for fractional max-pooling 2D layer.
    '''
    @property
    def output_shape(self):
        input_shape = self.input_shape
        if self.dim_ordering == 'th':
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.dim_ordering == 'tf':
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        rows = fractional_conv_output_length(rows, self.pool_size[0])
        cols = fractional_conv_output_length(cols, self.pool_size[1])

        if self.dim_ordering == 'th':
            return (input_shape[0], input_shape[1], rows, cols)
        elif self.dim_ordering == 'tf':
            return (input_shape[0], rows, cols, input_shape[3])
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

class FractionalMaxPooling2D(_FractionalPooling2D):
    '''Fractional Max pooling operation for spatial data, using random
    overlapping mode as described in Benjamin Graham's paper:
    http://arxiv.org/abs/1412.6071.
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.
    # Output shape
        4D tensor with shape:
        `(nb_samples, channels, pooled_rows, pooled_cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, pooled_rows, pooled_cols, channels)` if dim_ordering='tf'.
    # Arguments
        pool_size: tuple of 2 integers,
            factors by which to downscale (vertical, horizontal).
            (2, 2) will halve the image in each dimension.
        strides: tuple of 2 integers, or None. Strides values.
        border_mode: 'valid' or 'same'.
            Note: 'same' will only work with TensorFlow for the time being.
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode is it at index 3.
    '''

    def __init__(self, pool_size=(2, 2), strides=None, border_mode='valid',
                 dim_ordering='th', **kwargs):
        super(FractionalMaxPooling2D, self).__init__(pool_size, strides, border_mode,
                                           dim_ordering, **kwargs)

    def _pooling_function(self, inputs, pool_size, strides,
                          border_mode, dim_ordering):
        output = self.pool(inputs)
        return output

    def pool(self, inputs):
        '''Convert the inputs into a fractionally max-pooled output tensor.
        Implementation adapted from ebenolson:
        https://github.com/Lasagne/Lasagne/pull/171
        '''
        _, _, n_in0, n_in1 = self.input_shape

        n_out0 = fractional_conv_output_length(n_in0, self.pool_size[0])
        n_out1 = fractional_conv_output_length(n_in1, self.pool_size[1])

        # Variable stride across the input creates fractional reduction.
        a = theano.shared(np.array([2] * (n_in0 - n_out0) + [1] * (2 * n_out0 - n_in0)))
        b = theano.shared(np.array([2] * (n_in1 - n_out1) + [1] * (2 * n_out1 - n_in1)))

        # Randomize the input strides.
        a = theano_shuffled(a)
        b = theano_shuffled(b)

        # Convert to input positions, starting at 0.
        a = T.concatenate(([0], a[:-1]))
        b = T.concatenate(([0], b[:-1]))
        a = T.cumsum(a)
        b = T.cumsum(b)

        # Positions of the other corners.
        c = T.clip(a + 1, 0, n_in0 - 1)
        d = T.clip(b + 1, 0, n_in1 - 1)

        # Index the four positions in the pooling window and stack them.
        temp = T.stack(inputs[:, :, a, :][:, :, :, b],
                       inputs[:, :, c, :][:, :, :, b],
                       inputs[:, :, a, :][:, :, :, d],
                       inputs[:, :, c, :][:, :, :, d])

        out = T.max(temp, axis=0)

        return out

_srng = T.shared_randomstreams.RandomStreams()

def theano_shuffled(input):
    n = input.shape[0]
    shuffled = T.permute_row_elements(input.T, _srng.permutation(n=n)).T
    return shuffled

def fractional_conv_output_length(input_length, filter_size):
    return int(np.ceil(float(input_length) / filter_size))
