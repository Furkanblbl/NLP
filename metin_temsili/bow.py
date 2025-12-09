from sklearn.feature_extraction.text import CountVectorizer

# veri seti olustur
documents = [
    "kedi bahçede",
    "kedi evde"
]

# vectorizer tanimla
vectorizer = CountVectorizer()

# meti sayisal vektorlere cevir
X = vectorizer.fit_transform(documents)

# kelime kumesi olusturma
feature_names = vectorizer.get_feature_names_out() # kelime kumesini oluturma
print(f"feature_names: {feature_names}")

# vektor temsili
vector_temsili = X.toarray()
print(f"X: {X}")