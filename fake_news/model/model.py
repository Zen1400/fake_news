
from pyexpat import model
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn import set_config
from fake_news.model.clean import cleaning; set_config(display='diagram')
from tensorflow.keras.preprocessing.text import Tokenizer
import string
import os
import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, Sequential
from tensorflow.keras.callbacks import EarlyStopping




def initialize(vocab_size) :
    """
    Initialize a neural network model

    """
    model = Sequential([
    layers.Embedding(
    input_dim= vocab_size + 1,
    output_dim= 30,
    mask_zero= True, ),
    layers.LSTM(20),
    layers.Dense(10, activation = 'relu'),
    layers.Dense(1, activation="sigmoid")
    ])

    return model

def compile(model) :
    """
    Compile the initialized model
    """
    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

    return

def train(model, X, y, patience = 4, epochs = 2, validation_split = 0.3, batch_size =32) :
    """
    Train the model on X_train, y_train
    """
    # Early stopping and train the model

    es = EarlyStopping(patience = patience)

    history = model.fit(X,
                        y,
                        batch_size =batch_size,
                        validation_split= validation_split,
                        epochs= epochs,
                        callbacks= [es])

    return history, model
