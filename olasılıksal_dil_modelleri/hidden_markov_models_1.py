# import libraries
import nltk
from nltk.tag import hmm

# training data
train_data = [
    [("I","PRP"),("am","VBP"),("a","DT"),("teacher","NN")],
    [("You","PRP"),("are","VBP"),("a","DT"),("student","NN")]
]

# train HMM
trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train(train_data)

# test sentences
test_sentences= "I am a student".split()

tags = hmm_tagger.tag(test_sentences)
print(f"New sentence {tags}") 