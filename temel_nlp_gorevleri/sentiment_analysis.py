"""
Problem definition and dataset:
    https://github.com/pycaret/pycaret/blob/master/datasets/amazon.csv

    classification comments in amazon dataset to positive or negative
"""

# import libraries
import pandas as pd
import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download("vader_lexicon")
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")

# load dataset
df = pd.read_csv("../datasets/amazon.csv")

# text cleaning, preprocessing
lemmatizer = WordNetLemmatizer()
def clean_preprocess_data(text):
    # tokenize
    tokens = nltk.word_tokenize(text.lower())

    # stop words
    filtered_tokens = [token for token in tokens if token not in stopwords.words("english")]

    # lemmatize
    lemmatized_tokes = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # join words 
    process_text = " ".join(lemmatized_tokes)

    return process_text

df["reviewText2"] = df["reviewText"].apply(clean_preprocess_data)
