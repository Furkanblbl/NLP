"""
spam dataset -> spam and ham -> binary classification with Decision Tree
"""

# import libraries
import pandas as pd

# load dataset
data = pd.read_csv("../datasets/spam.csv", encoding="Latin-1")

data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1, inplace=True)
data.columns = ["label", "text"]
print(data)

# EDA: Exploratory data analysis
print(data.isna().sum()) # data is not missing

# text cleaning and preprocessing: special characters, lowercase, tokenization, stopwords, lemmatize


# model tranining and evalutaion