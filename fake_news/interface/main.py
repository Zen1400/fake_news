from json import load
from re import X
from fake_news.data.local import get_data
from fake_news.model.clean import (cleaning, prep)
from fake_news.model.model import (initialize, compile, train)
from fake_news.model.prep import (tokenizer, pad, predict)
import pandas as pd
import numpy as np
from fake_news.model.registry import load_model, save_model
from fake_news.model.params import MAXLEN
def preprocess() :
    """
    Get the data, clean, and preprocess
    """
    # Get the data from the folder raw_data
    df = get_data()

    # Clean the data

    print("Started Cleaning")

    df['text'] = df['text'].map(cleaning)

    df['text'] = df['text'].map(prep)

    # Tokenization
    print("Tokenizing")
    df, vocab_size = tokenizer(df)

    # Padding
    print("Padding")
    X, y = pad(df, maxlen = 300)

    return X, y, vocab_size


def training() :
    """
    Initialize a model, compile it and train it on X_train and y_train and save it with its metrics and params
    """
    X, y, vocab_size = preprocess()
    print("We got the preprocessed data frame")

    model = initialize(vocab_size)

    compile(model)
    print(" Initializing and compiling are done")
    print(model)
    validations_split = 0.3
    epochs = 5
    patience = 1
    batch_size = 32
    print("Starting the training")
    history, model = train(model, X, y, epochs = epochs, validation_split=validations_split, patience = patience, batch_size= batch_size)

    params = dict(validations_split = validations_split,
                  patience = patience,
                  batch_size = batch_size,
                  epochs = epochs)

    metrics = history.history['accuracy'][-1]

    save_model(model, params, metrics)
    print("Model saved")
    return print("Training done")

def pred(X_new) :
    """
    Prepare X for prediction using prep.predict then predict with loaded model
    """
    preprocessed = predict(X_new)

    model = load_model()

    prediction = model.predict(preprocessed)

    return prediction


if __name__ == '__main__':
    try:
        pred()
    except:
        import ipdb, traceback, sys

        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
