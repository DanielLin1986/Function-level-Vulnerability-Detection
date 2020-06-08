# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 14:36:16 2019

@author: DanielLin's PC
"""
import tensorflow as tf
import keras.backend as K
import tensorflow_hub as hub
from keras.engine import Layer

# Initialize session
sess = tf.Session()
K.set_session(sess)

# Create a custom layer that allows us to update weights (lambda layers do not have trainable parameters!)
class ElmoEmbeddingLayer(Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable=True
#        self.max_length = 1000
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,name="{}_module".format(self.name))
        #self.elmo = hub.Module('C:\\Users\\yuyu-\\.keras\\elmo_3\\', trainable=self.trainable,name="{}_module".format(self.name))

        self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),
                      as_dict=True,
                      signature='default',
                      )['default']
        return result

    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, '--PAD--')

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dimensions)