""""
Solve Classification problem (Sentiment Analysis in NLP) with RNN (Deep Learning based Language Model)

motion analyse -> labeling a sentence

"""

# import libraries
import pandas as pd
import numpy as np

from gensim.models import Word2Vec

from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Embedding

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
