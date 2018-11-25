from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Lambda
from keras.utils import np_utils
from keras import layers
from keras import backend as K
from keras.engine.topology import Layer,InputSpec
from keras import initializers,regularizers,constraints, activations
from keras.initializers import Initializer
from keras import regularizers

class MaxPlusDense(Layer):
    """A MaxPlus layer. TESTING MODE
    A `MaxPlus` layer takes the (element-wise + Bias) maximum of
      # Arguments
        units: int > 0.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights
            initialization. This parameter is only relevant
            if you don't pass a `weights` argument.
        weights: list of Numpy arrays to set as initial weights.
            The list should have 2 elements, of shape `(input_dim, units)`
            and (units,) for weights and biases respectively.
        bias_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        bias_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        bias: whether to include a bias
            (i.e. make the layer affine rather than linear).
        input_dim: dimensionality of the input (integer). This argument
            (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
    # Input shape
        2D tensor with shape: `(nb_samples, input_dim)`.
    # Output shape
        2D tensor with shape: `(nb_samples, units)`.
    """

    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 input_dim=None,
                 **kwargs):
        self.input_dim = input_dim
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(MaxPlusDense, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.input_dim = input_shape[-1]
        self.input_spec = InputSpec(dtype=K.floatx(),shape=(None, self.input_dim))
        self.kernel = self.add_weight(shape=(self.input_dim, self.units,),
                                initializer=self.kernel_initializer,
                                name='kernel',
                                regularizer=self.kernel_regularizer,
                                constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None  
        self.built = True

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def call(self, inputs):
        output=K.concatenate([K.reshape(inputs, [-1, self.input_dim, 1])]*self.units)
        output -=self.kernel
        output = K.max(output, axis=1)
        if self.use_bias:
            bias = K.reshape(K.concatenate([K.reshape(self.bias, [1, -1]) ]*K.int_shape(output)[0], axis=0),(-1, self.units, 1))
            output = K.reshape(output, (-1, self.units, 1))
            output = K.concatenate([output, bias])
            output = K.max(output, axis=-1)
        if self.activation is not None:
            output = self.activation(output)
        return output
        
    def get_config(self):
        config = {'units': self.units,
                  'use_bias': self.use_bias,
                  'activation': activations.serialize(self.activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'input_dim': self.input_dim}
        base_config = super(MaxPlusDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
