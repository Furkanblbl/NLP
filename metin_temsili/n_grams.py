from sklearn.feature_extraction.text import CountVectorizer

# ornek metin
documents = [
    "Bu çalışma NGram çalışmasıdır.",
    "Bu çalışma doğal dil işleme çalışmasıdır."
]

# unigram, bigram, trigram
vectorizer_unigram = CountVectorizer(ngram_range=(1,1))
vectorizer_bigram = CountVectorizer(ngram_range=(2,2))
vectorizer_trigram = CountVectorizer(ngram_range=(3,3))

# unigram
X_unigram = vectorizer_unigram.fit_transform(documents)
unigram_features = vectorizer_unigram.get_feature_names_out()
print(f"X_unigram:\n{X_unigram.toarray()}")
print(f"unigram_features:\n{unigram_features}")

# bigram
X_bigram = vectorizer_bigram.fit_transform(documents)
bigram_features = vectorizer_bigram.get_feature_names_out()
print(f"X_bigram:\n{X_bigram.toarray()}")
print(f"bigram_features:\n{bigram_features}")

# trigram
X_trigram = vectorizer_trigram.fit_transform(documents)
trigram_features = vectorizer_trigram.get_feature_names_out()
print(f"X_trigram:\n{X_trigram.toarray()}")
print(f"trigram_features:\n{trigram_features}")
