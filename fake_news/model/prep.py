"""
Split the data (train-test), tokenize, padding
"""
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
from fake_news.model.params import MAXLEN
from fake_news.model.clean import (cleaning, prep)
import os
import pickle
from fake_news.model.params import LOCAL_REGISTRY_PATH


def tokenizer(df : pd.DataFrame) -> pd.DataFrame :
    """
    Fits the tokenizer on the column text and transform it
    """
    # initialize the tokenizer
    tokenizer = Tokenizer()

    # fit on x_train
    tokenizer.fit_on_texts(df['text'])

    # transform X_train and X_test
    df['text'] = tokenizer.texts_to_sequences(df['text'])
    vocab_size = len(tokenizer.word_index)

    # Saving the tokenizer
    path = os.path.join(LOCAL_REGISTRY_PATH, "token", ".pickle")
    with open(path, "wb") as file:
        pickle.dump(tokenizer, file)

    return df, vocab_size


def pad(df : pd.DataFrame, maxlen= 300) -> pd.DataFrame :
    """
    Pad tokenized column based on maxlen specified in the .env
    """
    X = df['text']
    y = df['true']
    X = pad_sequences(X, dtype='float32', padding='post', maxlen=maxlen)

    return X, y


def predict(X) :
    """
    Prepares a text(image) for prediction by the loaded model in main.py
    """
    cleaned = cleaning(X)

    preped = [prep(cleaned)]


    # loading the tokenizer

    path = os.path.join(LOCAL_REGISTRY_PATH, "token", ".pickle")
    with open(path, 'rb') as file:
        tokenizer = pickle.load(file)

    tokenized = tokenizer.texts_to_sequences(preped)

    padded = pad_sequences(tokenized, dtype='float32', padding='post', maxlen = 300)

    return padded
