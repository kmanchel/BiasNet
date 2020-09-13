import tensorflow as tf
from tensorflow.keras import regularizers, initializers
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.layers import Layer, InputSpec


class Attention(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get("normal")
        self.supports_masking = True
        self.attention_dim = 50
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight(
            name="W",
            initializer=self.init,
            shape=tf.TensorShape((input_shape[-1], input_shape[1])),
            trainable=True,
        )
        self.b = self.add_weight(
            name="b", initializer=self.init, shape=tf.TensorShape((input_shape[1],)), trainable=True
        )
        self.u = self.add_weight(
            name="u",
            initializer=self.init,
            shape=tf.TensorShape((input_shape[1], 1)),
            trainable=True,
        )
        super(Attention, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)
        ait = K.exp(ait)

        if mask is not None:
            ait *= K.cast(mask, K.floatx())

        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])