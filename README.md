# NextWordPredictor

The Jupyter notebook "Next_word_prediction_using_LSTM" focuses on predicting the next word using an LSTM (Long Short-Term Memory) model. Below is a summary of its key components:

## Libraries Used
**NumPy** for numerical operations.
**Pandas** for data manipulation.
**NLTK** (Natural Language Toolkit) for text preprocessing, specifically tokenization.
**TensorFlow Keras** for building and training the LSTM model.
## Data Preprocessing
Tokenization of the input text using **RegexpTokenizer** from NLTK to split the text into tokens.
The input data is a collection of text data from a CSV file **(fake_or_real_news.csv)**, which is read and then combined into a single string for processing.
## Parameters
**n_words**: The number of words to consider as input for the model.
Various parameters for data processing like **unique_tokens**, **unique_token_index**, and structures for storing input and output sequences for the model.
## Model Architecture
The model is a Sequential model consisting of two LSTM layers followed by a Dense layer with a softmax activation function. The first LSTM layer has 128 units and returns sequences, while the second LSTM layer also has 128 units but does not return sequences. This is designed to predict the probability distribution over the unique tokens for the next word.
The model is compiled with the RMSprop optimizer and categorical crossentropy as the loss function. It is trained on the input and output sequences with a batch size of 128 for 50 epochs.
## Main Functions
**predict_next_word(input_text, n_best)**: Predicts the next word given an input text sequence, returning the **n_best** possible next words.
**generate_text(input_text, text_length, creativity=3)**: Generates text of a specified length starting from an input text, using the model's predictions. The **creativity** parameter influences the randomness of the next word selection.

This notebook demonstrates how to preprocess text data for a sequence prediction task, construct an LSTM-based model for next word prediction, and apply the model to generate text sequences.
