# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import re

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# load dataset
df = pd.read_csv('../datasets/IMDB_Dataset.csv')  # assuming the dataset is in a CSV file
documents = df['review']

# text cleaning
def clean_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = re.sub(r"[^\w\s]", '', text)  # remove punctuation
    text = " ".join([word for word in text.split() if len(word) > 2])  # remove short words

    return text

test = clean_text("This is a sample review! It has numbers 123 and short words a an the.")
# print(test)
cleaned_documents = [clean_text(doc) for doc in documents]

# text tokenization
tokenized_documents = [simple_preprocess(doc) for doc in cleaned_documents]
print(tokenized_documents[0:2])

# word2vec model
model = Word2Vec(sentences=tokenized_documents, vector_size=50, window=5, min_count=1, sg=0)
word_vectors = model.wv

words = list(word_vectors.index_to_key)[:500]
vectors = [word_vectors[word] for word in words]

# clustering kmeans k=2
kmeans = KMeans(n_clusters=2)
kmeans.fit(vectors)
clusters = kmeans.labels_ # 0, 1

# PCA 50 -> 2
pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(vectors)

# 2d plotting
plt.figure(figsize=(10, 8))
scatter = plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=clusters, cmap='viridis')
plt.title('Word2Vec Kelime Temsilleri ve Kümelenmesi (KMeans)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2') 
for i, word in enumerate(words):
    plt.text(reduced_vectors[i, 0], reduced_vectors[i, 1], word)
plt.colorbar(scatter)
plt.show()