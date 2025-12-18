import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("../datasets/spam.csv", encoding="latin-1")

# Sadece gerekli kolonlari al (spam datasetleri genelde boyle olur)
df = df[["v1", "v2"]]
df.columns = ["label", "text"]

# vektorlestirme
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=3000
)

X = vectorizer.fit_transform(df["text"])

# kelime kumesi
feature_names = vectorizer.get_feature_names_out()

# ortalama tf-idf skorlarini al
tfidf_scores = X.mean(axis=0).A1

# DataFrame olustur
df_tfidf = pd.DataFrame({
    "word": feature_names,
    "tfidf_score": tfidf_scores
})

# skora gore sirala
df_tfidf_sorted = df_tfidf.sort_values(
    by="tfidf_score",
    ascending=False
)

# En yuksek TF-IDF kelimeler
top_n = 20
top_words = df_tfidf_sorted.head(top_n)

plt.figure(figsize=(10, 5))
plt.bar(top_words["word"], top_words["tfidf_score"])
plt.xticks(rotation=90)
plt.xlabel("Kelimeler")
plt.ylabel("TF-IDF Skoru")
plt.title("En Yüksek TF-IDF Skoruna Sahip 20 Kelime")
plt.tight_layout()
plt.show()

# TF-IDF skor dagilimi
plt.figure(figsize=(8, 5))
plt.hist(df_tfidf["tfidf_score"], bins=50)
plt.xlabel("TF-IDF Skoru")
plt.ylabel("Kelime Sayısı")
plt.title("TF-IDF Skor Dağılımı")
plt.tight_layout()
plt.show()

# Dusuk skorlari filtreleyerek grafik
filtered_df = df_tfidf_sorted[df_tfidf_sorted["tfidf_score"] > 0.01].head(20)

plt.figure(figsize=(10, 5))
plt.bar(filtered_df["word"], filtered_df["tfidf_score"])
plt.xticks(rotation=90)
plt.xlabel("Kelimeler")
plt.ylabel("TF-IDF Skoru")
plt.title("TF-IDF > 0.01 Olan En Anlamlı Kelimeler")
plt.tight_layout()
plt.show()

# Konsol ciktilari (kontrol icin)
print("Top 10 TF-IDF Kelime:")
print(df_tfidf_sorted.head(10))
