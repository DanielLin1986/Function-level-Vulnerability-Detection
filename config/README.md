# Config File Documentation

**Please note that the suggested settings in the configuration file provided is not fine-tuned**, one has to adjust the settings based on the data. The description of parameters in the configuration file:

## Embedding settings

| Parameter     | Description                                                                                                                 | Note                                                               |
|---------------|-----------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| data_path         | The path of the code base for training the embedding model.                                                                                 | By default, the data for training the embedding model stores in `data\`                                        |
| embedding_model_saved_path | The path of the trained embedding model. When the embedding training is completed, the trained embedding model i.e., the dictionary file will be placed in the path specified.                                                                                           | By default, the trained embedding files will be stored in `embedding\` |
| seed | The seed for replicating the result. | By default, the seed is 1. |
| n_workders | The number of threads for training. | By default, we use 4 threads. Using more threads to speed up training.|
| size/components | The dimensionality of the word vectors. | By default, we use 100. After embedding, each input sequence will be a tensor with the shape of (1000, 100).  |
| window   | The maximum distance between the current and predicted word within a sentence. | In this code, we use a value of 5. |
| epoch (GloVe & FastText)   | The epcoh used for training. | In this code, we use the value of 40.|
| min_count (Word2vec & FastText)   | Ignores all words with total frequency lower than this. | In this code, we set this value to 5.|
| algorithm (Word2vec & FastText)   | Training algorithm: 1 for skip-gram; otherwise Continouse Bag-Of-Word (CBOW).| In this code, we use CBOW.|
| learning_rate (GloVe) | The learning rate used for training. | In this code, we set this value to 0.001.


## Model settings

| Parameter     | Description                                                                                                                 | Note                                                               |
|---------------|-----------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| model          | The name of the neural network models for training.                                                                                 | Currently, the code supports the DNN, RNNs (i.e., LSTM and GRU ), BiRNN (i.e., bidirectional LSTM and bidirectional GRU), and textCNN)                                          |
| optimizer | The optimizer used.                                                                                                | A user can choose different optimizers based on their tasks. In this code, we use the SGD with its default settings.                                                                     |
| loss_function    | The loss function to minimize. | We use the binary cross entropy. |
| handle_data_imbalance    | Whether to handle the data imbalance issue. | The cost-sensitive learning will be applied if it is set to True|
| max_sequence_length   | The length for each input code sequence. | In this code, we use 1,000 as the maximal input length.|
| use_dropout   | Whether to use dropout | In this code, we use a dropout to prevent overfitting. |
| dropout_rate   | The dropout value. | In this code, we use the value of 0.5.|
| dnn_size   | The number of neurons used for DNN (the first layer) | In this code, we set this value to 128.|
| rnn_size   | The number of neurons used for RNNs (the first layer) | In this code, we set this value to 128.|
| birnn_size   | The number of neurons used for bidirectional RNNs (the first layer) | In this code, we set this value to 64.|
| embedding_trainable | Whether allows the trained embedding layer to be tuned. | By default, we set this value to False. 

## Training settings

| Parameter     | Description                                                                                                                 | Note                                                               |
|---------------|-----------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| Test_set_ratio   | If not using a separate test set, set the test set ratio.  | In this code, we partition the dataset into training, validation and test sets with a ratio of 6:2:2. Users can use their own test set. If users use their own test set, they should set the 'using_separate_test_set' to True and ignore this value. |
| using_separate_test_set   | Suggest whether use a separate test set. | If this is set to True, please specify the path of test set.  |
| test_set_path   | The path contains the test data.  | If a user uses a separate test set, the user should specify a path leading to the test data. |
| Validation_set_ratio   | Suggest the percentage of the total dataset that is used as the validation set. | We use part of the training set as the validation set. |
| batch_size   | The size of mini batch | A relatively small batch size leads to better generalization. We use the value of 16. |
| epochs   | The number of forward and backward pass of all the training examples. | In this code, we set this value to 150.|
| patcience   | The number of epochs with no improvement after which training will be stopped. | In this code, we set this value to 35.|
| save_training_history   | Whether to save the training history. | The default value is True.|
| plot_training_history   | Whether to plot the training/validation curve. | The default value is True.|
| validation_metric   | The quantity to be monitored. | In this code, we choose to monitor validation loss.|
| save_best_model   | If save_best_only=True, the latest best model according to the quantity monitored will not be overwritten | In this code, we set this value to True.|
| period_of_saving   | Interval (number of epochs) between checkpoints. | In this code, we set this value to 1.|
| log_path   | The path where the log files are stored. | By default, the log files are stored in the `logs/`.|
| model_save_path   | The path where the trained models are stored. | By default, the trained models are stored in the `result/models/`.|
| model_saved_name   | The name of the trained model. | By default, the trained model is called 'test_model'.
