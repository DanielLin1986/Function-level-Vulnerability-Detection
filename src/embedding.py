# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 10:38:49 2019

@author: Daniel Lin

"""
import os
import pickle
import numpy as np
from src.DataLoader import LoadPickleData

"""
Tokenization:
    1. Perform tokenization and save tokenizer.
    2. Load tokenizer.
"""
class Embedding_Model():
    
    def __init__ (self, config):
        self.config = config
        self.tokenizer_path = self.config['embedding_settings']['embedding_model_saved_path'] 
        self.tokenizer_saved_path = self.config['embedding_settings']['embedding_model_saved_path']
        assert os.path.exists(self.tokenizer_path)
        assert os.path.exists(self.tokenizer_saved_path)
        self.n_workers = self.config['embedding_settings']['n_workers']
        self.seed = self.config['embedding_settings']['seed']
        
    def LoadTokenizer(self, data_list):
        tokenizer = LoadPickleData(self.tokenizer_path + 'tokenizer.pickle')
        total_sequences = tokenizer.texts_to_sequences(data_list)
        word_index = tokenizer.word_index
        
        return total_sequences, word_index            

class WordToVec(Embedding_Model):
    ''' Handler for Word2vec training progress...'''
    def __init__(self,config):
        super(WordToVec, self).__init__(config)
        
        self.wordtovec_size = self.config['embedding_settings']['word2vec']['size']
        self.wordtovec_window = self.config['embedding_settings']['word2vec']['window']
        self.wordtovec_min_count = self.config['embedding_settings']['word2vec']['min_count']
        self.wordtovec_algorithm = self.config['embedding_settings']['word2vec']['algorithm']
        
    def TrainWordToVec(self, data_list):
        from gensim.models import Word2Vec
        
        print ("----------------------------------------")
        print ("Start training the Word2Vec model. Please wait.. ")
        # 2. Train a Vocabulary with Word2Vec -- using the function provided by gensim
        w2vModel = Word2Vec(data_list, workers = self.n_workers, size = self.wordtovec_size, window = self.wordtovec_window, min_count = self.wordtovec_min_count, sg = self.wordtovec_algorithm, seed = self.seed)
        print ("Model training completed!")
        print ("----------------------------------------")
        print ("The trained word2vec model: ")
        print (w2vModel)
        
        w2vModel.wv.save_word2vec_format(self.tokenizer_saved_path + "w2v_model.txt", binary=False)
        
    def ApplyWordToVec(self, word_index):
        
        print ("-------------------------------------------------------")
        print ("Loading trained Word2vec model. ")
        w2v_model = open(self.tokenizer_saved_path + "w2v_model.txt")        
        print ("The trained word2vec model: ")
        print (w2v_model)
        
        embeddings_index = {} # a dictionary with mapping of a word i.e. 'int' and its corresponding 100 dimension embedding.

        # Use the loaded model
        for line in w2v_model:
           if not line.isspace():
               values = line.split()
               word = values[0]
               coefs = np.asarray(values[1:], dtype='float32')
               embeddings_index[word] = coefs
        w2v_model.close()
        
        print ('Found %s word vectors.' % len(embeddings_index))
        
        embedding_matrix = np.zeros((len(word_index) + 1, self.wordtovec_size))
        for word, i in word_index.items():
           embedding_vector = embeddings_index.get(word)
           if embedding_vector is not None:
               # words not found in embedding index will be all-zeros.
               embedding_matrix[i] = embedding_vector
               
        return embedding_matrix, self.wordtovec_size
    
class Glove(Embedding_Model):
    ''' Handler for Glove training progress...'''
    def __init__(self,config):
        super(Glove, self).__init__(config)
        
        self.components = self.config['embedding_settings']['glove']['components']
        self.glove_window = self.config['embedding_settings']['glove']['window']
        self.glove_epoch = self.config['embedding_settings']['glove']['epoch']
        self.glove_learning_rate = self.config['embedding_settings']['glove']['learning_rate']
        
    def TrainGlove(self, data_list):
        
        from glove import Corpus, Glove
        # creating a corpus object
        print ("----------------------------------------")
        print ("Start training the GLoVe model. Please wait.. ")
        corpus = Corpus()
        corpus.fit(data_list, window=self.glove_window)
        glove = Glove(no_components=self.components, learning_rate=self.glove_learning_rate)
 
        glove.fit(corpus.matrix, epochs=self.glove_epoch, no_threads=self.n_workers, verbose=True)
        glove.add_dictionary(corpus.dictionary)
        glove.save(self.tokenizer_saved_path + os.sep + 'glove.model') # This is to save the model as a pkl file.
        
        # Save Glove model as .txt format for checking content
        vector_size = self.components
        with open(self.tokenizer_saved_path + 'results_glove.txt', "w", encoding = 'latin-1') as f:
            for word in glove.dictionary:
                f.write(word)
                f.write(" ")
                for i in range(0, vector_size):
                    f.write(str(glove.word_vectors[glove.dictionary[word]][i]))
                    f.write(" ")
                f.write("\n")
            f.close()

        print("GLOVE SAVE HERE ", self.tokenizer_saved_path + 'glove.model')
        
        print ("Model training completed!")
        print ("----------------------------------------")
        
    def ApplyGlove(self, word_index):
        with open(self.tokenizer_saved_path + os.sep + 'glove.model', 'rb') as f:
            glove_model = pickle.load(f, encoding = 'latin-1')
            
        key_list = list(glove_model['dictionary'].keys())
        word_vector_list = glove_model['word_vectors'].tolist()
        
        embeddings_index = {}
        for index, item in enumerate(key_list):
            word = key_list[index]
            coefs = np.asarray(word_vector_list[index], dtype='float32')
            embeddings_index[word] = coefs
        print('Loaded %s word vectors.' % len(embeddings_index))
        
        embedding_matrix = np.zeros((len(word_index) + 1, self.components))
        for word, i in word_index.items():
           embedding_vector = embeddings_index.get(word)
           if embedding_vector is not None:
               # words not found in embedding index will be all-zeros.
               embedding_matrix[i] = embedding_vector
        
        return embedding_matrix, self.components
            
class FastText(Embedding_Model):
    ''' Handler for Word2vec training progress...'''
    def __init__(self,config):
        super(FastText, self).__init__(config)
        
        self.fasttext_size = self.config['embedding_settings']['fasttext']['size']
        self.fasttext_window = self.config['embedding_settings']['fasttext']['window']
        self.fasttext_min_count = self.config['embedding_settings']['fasttext']['min_count']
        self.fasttext_algorithm = self.config['embedding_settings']['fasttext']['algorithm']
        
    def TrainFastText(self, data_list):
        from gensim.models import FastText
        
        print ("----------------------------------------")
        print ("Start training the FastText model. Please wait.. ")
        # 2. Train a Vocabulary with Word2Vec -- using the function provided by gensim
        ft_Model = FastText(data_list, workers = self.n_workers, size = self.fasttext_size, window = self.fasttext_window, min_count = self.fasttext_min_count, sg = self.fasttext_algorithm, seed = self.seed)
        print ("Model training completed!")
        print ("----------------------------------------")
        print ("The trained FastText model: ")
        print (ft_Model)
        
        ft_Model.wv.save_word2vec_format(self.tokenizer_saved_path + "ft_model.txt")
        
    def ApplyFastText(self, word_index):
        
        #from gensim.models.wrappers import FastText
        print ("-------------------------------------------------------")
        print ("Loading trained model. ")
        ft_model = open(self.tokenizer_saved_path + "ft_model.txt")        
        print ("The trained word2vec model: ")
        print (ft_model)
        
        embeddings_index = {} # a dictionary with mapping of a word i.e. 'int' and its corresponding 100 dimension embedding.

        # Use the loaded model
        for line in ft_model:
           if not line.isspace():
               values = line.split()
               word = values[0]
               coefs = np.asarray(values[1:], dtype='float32')
               embeddings_index[word] = coefs
        ft_model.close()
        print ('Found %s word vectors.' % len(embeddings_index))
        
        embedding_matrix = np.zeros((len(word_index) + 1, self.fasttext_size))
        for word, i in word_index.items():
           embedding_vector = embeddings_index.get(word)
           if embedding_vector is not None:
               # words not found in embedding index will be all-zeros.
               embedding_matrix[i] = embedding_vector
               
        return embedding_matrix, self.fasttext_size