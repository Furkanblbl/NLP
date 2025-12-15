# import libraries
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# veri setinin iceriye aktarilmasi
df = pd.read_csv("../datasets/IMDB_Dataset.csv")
documents = df["review"]
labels = df["sentiment"]
# print(df)

# metin temizleme
def clean_text(text):
    # kucuk harfe cevir
    text = text.lower()

    # rakamlari temizle
    text = re.sub(r"\d+", "", text)

    # ozel karakterleri kaldir
    text = re.sub(r"[^\w\s]", "", text)

    # stop words listesi
    stop_words = ENGLISH_STOP_WORDS

    # kisa kelimeler + stop words temizleme
    text = " ".join([
        word for word in text.split()
        if len(word) > 2 and word not in stop_words
    ])

    return text

cleaned_doc = [clean_text(row) for row in documents]
# print(f"cleaned_doc {cleaned_doc}")

# bow
#vectorizer tanimla
vectorizer = CountVectorizer()

# metin -> sayisal hale getir
X = vectorizer.fit_transform(cleaned_doc[:75])

# kelime kumesi goster
feature_names = vectorizer.get_feature_names_out()

# vektor temsili goster
vektor_temsili2 = X.toarray()
# print(f"vektor_temsili2: {vektor_temsili2}")

df_bow = pd.DataFrame(vektor_temsili2, columns=feature_names)
print(f"df_bow: {df_bow}")

# kelime frekanslarini goster
word_counts = X.sum(axis=0).A1
word_freq = pd.DataFrame({
    "word": feature_names,
    "count": word_counts
}).sort_values(by="count", ascending=False)

# en sık geçen 20 kelime
top_words = word_freq.head(20)

plt.figure(figsize=(10,5))
plt.bar(top_words["word"], top_words["count"])
plt.xticks(rotation=90)
plt.title("En Sık Geçen 20 Kelime (BoW)")
plt.tight_layout()
plt.show()