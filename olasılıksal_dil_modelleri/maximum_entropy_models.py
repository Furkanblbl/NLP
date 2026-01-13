"""
    classification problem: duygu analizi -> olumlu veya olumsuz olarak siniflandirma
"""

# import libraries
from nltk.classify import MaxentClassifier

# dataset definition
train_data = [
    ({"Love":True, "amazing":True, "happy":True, "terrible":False}, "positive"),
    ({"hate":True, "terrible":True}, "negative"),
    ({"joy":True, "happy":True, "hate":False},"positive"),
    ({"sad":True, "depressed":True, "love":False}, "negative")
]

# train maximum entropy classifier
classifier = MaxentClassifier.train(train_data,max_iter=10)

# test
test_sentence = "I do not hate this movie"
features = {word: (word in test_sentence.lower()) for word in ["love","amazing","terrible","happy","joy","depressed","sad","hate"]}
print(f"features: {features}")

label = classifier.classify(features)
print(f"Result: {label}")
