# import libraries
from transformers import BertTokenizer, BertModel

import numpy as np
import sklearn.metrics.pairwise as cosine_similarity

# tokenizer and model create
model_name  = "bert-base-uncased" # lower size bert model
tokenizer   = BertTokenizer.from_pretrained(model_name)
model       = BertModel.from_pretrained(model_name)

# create dataset
documents = [
    "Machine learning is a field of artificial intelligence that uses statistical techniques to give computer systems the ability to 'learn' from data, without being explicitly programmed.",
    "Natural language processing (NLP) is a subfield of artificial intelligence that focuses on the interaction between computers and humans through natural language.",
    "Deep learning is a subset of machine learning that uses neural networks with many layers (hence 'deep') to model complex patterns in data.",
    "Transformers are a type of deep learning model that have revolutionized natural language processing tasks by enabling models to understand context and relationships in text more effectively.",
    "BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained transformer model that has been widely used for various NLP tasks.",
    "Deep learning is a subset of machine learning that uses neural networks with many layers (hence 'deep') to model complex patterns in data.",
    "I go to shop"
]

query = "What is Deep Learning?"

# get information with bert model

def get_embedding(text):

    # tokenization
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # run the model
    outputs = model(**inputs)

    # last generated hidden states
    last_hidden_state = outputs.last_hidden_state

    # text representation by averaging token embeddings
    embedding = last_hidden_state.mean(dim=1)

    # return numpy array
    return embedding.detach().numpy()

# get embeddings for documents
doc_embeddings = np.vstack([get_embedding(doc) for doc in documents])
query_embedding = get_embedding(query)

# compute cosine similarities
similarities = cosine_similarity.cosine_similarity(query_embedding, doc_embeddings)

# get the most similar document
for i, score in enumerate(similarities[0]):
    print(f"Document {i+1} similarity score: {score:.4f}")

most_similar_doc_index = np.argmax(similarities)
print("\nMost similar document:")
print(documents[most_similar_doc_index])