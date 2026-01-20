# %%
"""
Problem definition and dataset:
    https://github.com/pycaret/pycaret/blob/master/datasets/amazon.csv

    classification comments in amazon dataset to positive or negative
"""

# %%
# import libraries
import pandas as pd
import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix, classification_report

# %%
nltk.download("vader_lexicon")
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")

# %%
# load dataset
df = pd.read_csv("../datasets/amazon.csv")

# %%
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

# %%
df["reviewText2"] = df["reviewText"].apply(clean_preprocess_data)
df[["reviewText", "reviewText2"]].head()

# %%
# sentiment analysis (nltk)
analyzer = SentimentIntensityAnalyzer()

def get_sentiments(text):
    score = analyzer.polarity_scores(text)
    sentiment = 1 if score["pos"] > 0 else 0

    return sentiment

df["sentiment"] = df["reviewText2"].apply(get_sentiments)
df[["reviewText", "reviewText2"]].head()

# %%
cm = confusion_matrix(df["Positive"], df["sentiment"])
print(f"Confusion matrix: {cm}")

cr = classification_report(df["Positive"], df["sentiment"])
print(f"Classification report: {cr}")
