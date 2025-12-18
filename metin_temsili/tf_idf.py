# import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "Köpek tatlı bir hayvandır.",
    "Kuşlar ve köpek çok tatlı hayvanlardır.",
    "İnekler süt üretirler."
]

vectorizer = TfidfVectorizer(
    lowercase=True,
    token_pattern=r"(?u)\b\w+\b"
)

X = vectorizer.fit_transform(documents)
feature_names = vectorizer.get_feature_names_out()

df_tfidf = pd.DataFrame(X.toarray(), columns=feature_names)

mean_tfidf = df_tfidf.mean(axis=0).sort_values(ascending=False)
mean_df = mean_tfidf.reset_index()
mean_df.columns = ["Kelime", "Ortalama TF-IDF"]

plt.figure()
plt.axis("off")
plt.title("Ortalama TF-IDF Değerleri")

table = plt.table(
    cellText=mean_df.round(3).values,
    colLabels=mean_df.columns,
    loc="center"
)

table.scale(1, 1.5)
plt.show()

plt.figure()
plt.bar(mean_tfidf.index, mean_tfidf.values)
plt.xticks(rotation=90)
plt.title("Ortalama TF-IDF Değerleri")
plt.tight_layout()
plt.show()

plt.figure()
for i in range(len(df_tfidf)):
    plt.plot(
        df_tfidf.columns,
        df_tfidf.iloc[i],
        marker="o",
        label=f"Doküman {i+1}"
    )

plt.xticks(rotation=90)
plt.title("Dokümanlara Göre TF-IDF Dağılımı")
plt.legend()
plt.tight_layout()
plt.show()
