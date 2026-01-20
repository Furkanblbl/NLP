import spacy

nlp = spacy.load("en_core_web_sm")

sentence1 = "What is the weather like today or tomorrow"
sentence2 = "I worked on computers for a long time."
doc1 = nlp(sentence2) # analysis sentence

for token in doc1:
    print(token.text, token.pos_)

"""
What PRON
is AUX
the DET
weather NOUN
like ADP
today NOUN
or CCONJ
tomorrow NOUN
"""