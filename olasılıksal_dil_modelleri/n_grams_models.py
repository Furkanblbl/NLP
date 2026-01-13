# import libraries

import nltk # natural language toolkit
from nltk.util import ngrams # n-grams
from nltk.tokenize import word_tokenize # tokenization

from collections import Counter # frequency distribution

# create dataset
corpus = [
    "I love apple",
    "I love him",
    "I love NLP",
    "You love me",
    "He loves me",
    "They love apple",
    "I love you and you love me"
]

"""
Problem tanimi:
    dil modeli yapmak istiyoruz.
    amac 1 kelimeden sonra gelecek kelimeyi tahmin etmek.: metin turetmek/olusturmak
    bunun icin n-grams modelleri kullanacagiz.

    ex: I ....(Love),,,(apple)
"""

# tokenization
tokens = [word_tokenize(sentence.lower()) for sentence in corpus]
# print(f"tokens:\n{tokens}")


# bigram model
bigrams = []
for token_list in tokens:
    bigrams.extend(list(ngrams(token_list, 2)))
# print(f"bigrams:\n{bigrams}")

bigrams_freq = Counter(bigrams)
print(f"bigrams_freq:\n{bigrams_freq}")

# trigram model
trigrams = []
for token_list in tokens:
    trigrams.extend(list(ngrams(token_list, 3)))

print(f"trigrams:\n{trigrams}")
trigrams_freq = Counter(trigrams)
print(f"trigrams_freq:\n{trigrams_freq}")

bigram = ('i', 'love') # example bigram
# i love you probability
prob_you = trigrams_freq[bigram + ('you',)] / bigrams_freq[bigram]
print(f"P(you|i love) = {prob_you}")
