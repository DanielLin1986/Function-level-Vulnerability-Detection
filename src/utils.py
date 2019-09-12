# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 15:07:15 2019

"""
import os
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K

def plot_history(config, network_history): 
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(network_history.history['loss'])
    plt.plot(network_history.history['val_loss'])
    plt.legend(['Training', 'Validation'])
    plt.savefig(config['training_settings']['model_save_path'] + os.sep + config['training_settings']['model_saved_name'] + '_Epoch_loss' + '.jpg') 


def visualize_attention(test_seq, i, attention_layer, model, word_index, n):
    """
    Visualize that in the i th sequence, the top n words that the model pays attention to. 
    We first do a forward pass and get the output of the LSTM layer.
    THen we apply the function of the Attention layer and get the weights.
    Finally we obtain and print the words of the input sequence 
    that have these weights.

    """
    
    new_word_index = dict((v, k) for k, v in word_index.items()) # Exchange key with values.

    get_layer_output = K.function([model.layers[0].input, K.learning_phase()], [model.layers[4].output])
    out = get_layer_output([test_seq, ])[0]  # test mode

    att_w = model.layers[attention_layer].get_weights() # The attention layer

    eij = np.tanh(np.dot(out[i], att_w[0]))
#    print("1 eij is ............")
#    print(eij)
#    ListToCSV(eij, 'eij.csv') # added by Shigang
    ai = np.exp(eij)
##    print("1 ai is ............")
##    print(ai)
#    ListToCSV(ai, 'ai.csv') # added by Shigang
    weights = ai/np.sum(ai)
##    print("1 weights is ............")
##    print(weights)
#    ListToCSV(weights, '1weights.csv') # added by Shigang
#    weights = np.sum(weights, axis=1)
##    print("2 weights is ............")
##    print(weights)
#    ListToCSV(weights, '2weights.csv') # added by Shigang
    topKeys = np.argpartition(weights,-n)[-n:]

    print (' '.join([new_word_index[wrd_id] for wrd_id in test_seq[i] if wrd_id != 0.]))
    print ('--------------------------Attentive Words start: ------------------------------------')
    
    for k in test_seq[i][topKeys]:
        if k != 0.:
            print (new_word_index[k])
    print ('--------------------------Attentive Words end: --------------------------------------')
    
    return

def printAttentionWords(test_set_x, test_set_y, model, word_index, num_word_pay_atten):
    
    vul_index = []
    
    for index, item in enumerate(test_set_y):
#        print(item)
        if item == 1:
#            print('item1')
#            print(item)
            vul_index.append(index)
#            print(vul_index)
#            print(index)
    
    # Only visualize the vulnerable sequences. 
    for item in vul_index:
#        print('item2')
#        print(item)
        visualize_attention(test_set_x, item, model, word_index, num_word_pay_atten)

def plot_attention_graph(selected_samples, attention_coefs, selected_values):
    for sample_id, attention, y in zip(selected_samples, attention_coefs, selected_values):
        plt.figure(figsize=(20, 3))
        plt.title(f'Sample #{sample_id}. Correct value @ index {y}')
        plt.xlabel('Timesteps')
        plt.ylabel('Attention coefficient')

        plt.grid(alpha=0.4)
        plt.plot(attention)
        
        max_attention = np.max(attention)

        U = np.linspace(0, max_attention, 20)
        y = np.ones_like(U) * y
        plt.plot(y, U, 'r', alpha=0.8, label='Correct timestep')

        plt.legend()
