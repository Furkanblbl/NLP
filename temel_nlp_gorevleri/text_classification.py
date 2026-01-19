"""
spam dataset -> spam and ham -> binary classification with Decision Tree
"""

# import libraries
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# load dataset
data = pd.read_csv("../datasets/spam.csv", encoding="Latin-1")

data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1, inplace=True)
data.columns = ["label", "text"]
print(data)

# EDA: Exploratory data analysis
print(data.isna().sum()) # data is not missing

# text cleaning and preprocessing: special characters, lowercase, tokenization, stopwords, lemmatize
nltk.download("stopwords") # discard common used and not have meaning words
nltk.download("wordnet") # necessary for find lemma
nltk.download("omw-1.4") # have different language words meaning dataset

text = list(data.text)
lemmatizer = WordNetLemmatizer()

corpus = []

text = "Fırkan 2134rq fas241 54wefgw"
for i in range(len(text)):

    r = re.sub("[a-zA-Z]", " ",text[i]) # discard all characters without letter

    r = r.lower()

    r = r.split()
    
    r = [word for word in r if word not in stopwords.words("english")] # discard stopwords
    
    r = [lemmatizer.temmatize(word) for word in r]
    
    r = "".join(r)

    corpus.append(r)

data["text2"] = corpus


# model tranining and evalutaion