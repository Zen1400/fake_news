import pandas as pd
import string
import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def cleaning(sentence : str) -> str:
    """
    Removes numbers and punctuations, and makes all letters small
    """
    # making all letters lower_case
    sentence = sentence.lower()

    # Removing numbers
    sentence = ''.join(char for char in sentence if not char.isdigit())

    # Removing punctuation
    for punctuation in string.punctuation:
        sentence = sentence.replace(punctuation, '')

    sentence = sentence.strip()

    return sentence



def prep(sentence : str)->list :
    """
    Removes stop words and transform a text into a list of words
    """

    stop_words = set(stopwords.words('english'))
    sentence = word_tokenize(sentence)
    sentence = [w for w in sentence if not w in stop_words]

    return sentence
