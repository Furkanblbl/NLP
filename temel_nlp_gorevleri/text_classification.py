"""
spam dataset -> spam and ham -> binary classification with Decision Tree
"""

# import libraries
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

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

for i in range(len(text)):

    r = re.sub("[a-zA-Z]", " ",text[i]) # discard all characters without letter

    r = r.lower()

    r = r.split()
    
    r = [word for word in r if word not in stopwords.words("english")] # discard stopwords
    
    r = [lemmatizer.lemmatize(word) for word in r]
    
    r = "".join(r)

    corpus.append(r)

data["text2"] = corpus


# model tranining and evalutaion

X = data["text2"]
y = data["label"] # target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# feature extraction: bag of words
cv = CountVectorizer()
X_train_cv = cv.fit_transform(X_train)

# classifier training: model training and evaluation
dt = DecisionTreeClassifier()
dt.fit(X_train_cv, y_train) # training

x_test_cv = cv.transform(X_test)

# prediction
prediction = dt.predict(x_test_cv)

c_matrix = confusion_matrix(y_test, prediction)
print(f"Confusion matrix:\n{c_matrix}")

accuracy = 100*(sum(sum(c_matrix)) - c_matrix[1,0] - c_matrix[0,1]) / sum(sum(c_matrix))
print(accuracy)