# import libraries
import nltk
from nltk.tag import hmm
from nltk.corpus import conll2000

# include datasets
nltk.download("conll2000")

train_data = conll2000.tagged_sents("train.txt")
test_data = conll2000.tagged_sents("test.txt")

print(f"train_data = {train_data[:1]}")

# train hmm
trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train(train_data)

# new sentence and test
test_sentences = "I like going to school".split()
tags = hmm_tagger.tag(test_sentences)
