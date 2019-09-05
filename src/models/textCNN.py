# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 11:49:37 2019

@author: Daniel Lin
"""

from keras import initializers
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D, Reshape, Concatenate,Flatten 
from keras.layers.core import Dropout

class textCNN(object):
    
    def __init__ (self, config):
        
        self.MAX_LEN = config['model_settings']['model_para']['max_sequence_length']
        self.use_dropout = config['model_settings']['model_para']['use_dropout']
        self.dropout_rate = config['model_settings']['model_para']['dropout_rate']
        self.LOSS_FUNCTION = config['model_settings']['loss_function']
        self.OPTIMIZER = config['model_settings']['optimizer']['type']
        self.dnn_size = config['model_settings']['model_para']['dnn_size']
        self.embedding_trainable = config['model_settings']['model_para']['embedding_trainable']
        
    def buildModel(self, word_index, embedding_matrix, embedding_dim):
        filter_sizes = [3,4,5,6]
        num_filters = 16
        inputs = Input(shape=(self.MAX_LEN,))
        sharable_embedding = Embedding(len(word_index) + 1,
                                   embedding_dim,
                                   weights=[embedding_matrix],
                                   input_length=self.MAX_LEN,
                                   trainable= self.embedding_trainable)(inputs)
        reshape = Reshape((self.MAX_LEN, embedding_dim, 1))(sharable_embedding)
        conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                        bias_initializer=initializers.constant(value=0.1), activation='relu')(reshape)
        maxpool_0 = MaxPool2D(pool_size=(self.MAX_LEN - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
        
        conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                            bias_initializer=initializers.constant(value=0.1), activation='relu')(reshape)
        maxpool_1 = MaxPool2D(pool_size=(self.MAX_LEN - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
        
        conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                            bias_initializer=initializers.constant(value=0.1), activation='relu')(reshape)
        maxpool_2 = MaxPool2D(pool_size=(self.MAX_LEN - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)
        
        conv_3 = Conv2D(num_filters, kernel_size=(filter_sizes[3], embedding_dim), padding='valid', kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                            bias_initializer=initializers.constant(value=0.1), activation='relu')(reshape)
        maxpool_3 = MaxPool2D(pool_size=(self.MAX_LEN - filter_sizes[3] + 1, 1), strides=(1,1), padding='valid')(conv_3)
    
        concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3])
        flatten = Flatten()(concatenated_tensor)
        
        if self.use_dropout:
            dropout_layer_2 = Dropout(self.dropout_rate)(flatten)
            dense_1 = Dense(self.dnn_size, activation='relu')(dropout_layer_2)
        else:
            dense_1 = Dense(self.dnn_size, activation='relu')(flatten)
            
        if self.use_dropout:
            dropout_layer_3 = Dropout(self.dropout_rate)(dense_1)
            dense_2 = Dense(int(self.dnn_size/2), activation='relu')(dropout_layer_3)
        else:
            dense_2 = Dense(int(self.dnn_size/2), activation='relu')(dense_1)
    
        dense_4 = Dense(1, activation='sigmoid')(dense_2)
        
        model = Model(inputs=inputs, outputs = dense_4, name='text-CNN')
        
        model.compile(loss=self.LOSS_FUNCTION,
                 optimizer=self.OPTIMIZER,
                 metrics=['accuracy'])
        
        return model
        
        