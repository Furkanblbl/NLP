"""
    word2vec: google
    fasttext: facebook
"""
# import libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA # pricipal component analysis: dimensionality reduction technique

from gensim.models import Word2Vec, FastText
from gensim.utils import simple_preprocess

# create a dataset for word embeddings
sentences = {
    "Köpek çok tatlı bir hayvandır.",
    "Köpekler evcil hayvanlardır.",
    "Kediler genellikle bağımsız hareket etmeyi severler.",
    "Köpekler sadık ve dost canlısı hayvanlardır.",
    "Hayvanlar insanlar için iyi arkadaşlardır.", 
}

tokenized_sentences = [simple_preprocess(sentence) for sentence in sentences]

# word2vec
word2_vec_model = Word2Vec(sentences=tokenized_sentences, vector_size=50, window=5, min_count=1, sg=0)

# fasttext
fasttext_model = FastText(sentences=tokenized_sentences, vector_size=50, window=5, min_count=1, sg=1)

# visualization
def plot_word_embeddings(model, title):
    word_vectors = model.wv
    words = list(word_vectors.index_to_key)[:1000]
    vectors = [word_vectors[word] for word in words]
    
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(vectors)
    
    # 3d plotting
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # draw vectors
    ax.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], zs=0, zdir='z', s=20, c='b', depthshade=True)
    
    # label words
    for i, word in enumerate(words):
        ax.text(reduced_vectors[i, 0], reduced_vectors[i, 1], 0, word)
        
    ax.set_title(title)
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('Z')
    plt.show()


# plot word2vec embeddings
plot_word_embeddings(word2_vec_model, "Word2Vec Kelime Temsilleri")
# plot fasttext embeddings
plot_word_embeddings(fasttext_model, "FastText Kelime Temsilleri")