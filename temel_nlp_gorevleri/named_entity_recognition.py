"""
named entity recognation: text -> names entity recognition in text
"""

# import libraries
import pandas as pd
import spacy

# ned with spacy
nlp = spacy.load("en_core_web_sm") # spacy library english lang model

content = "Alice works at Amazon and lives in London. She visited the British Museum Last weekend"

doc = nlp(content) # this process analyse entity in text

for ent in doc.ents:
    # ent.text: alice, amazon
    # ent.start_char ve ent.end_char: start end finish characters 
    # ent.label_: presence type
    print(ent.text, ent.start_char, ent.end_char, ent.label_)

entities = [(ent.text, ent.label_, ent.lemma_) for ent in doc.ents]

# convert entities to pandas df
df = pd.DataFrame(entities, columns=["text", "type", "lemma"])
print(f"df:\n{df} ")
