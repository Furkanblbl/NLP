import spacy

nlp = spacy.load("en_core_web_sm")

word = "I go to schools"

doc = nlp(word)
# print(doc)

for token in doc:
    print(f"Text: {token.text}")           # the word itself
    print(f"Lemma: {token.lemma_}")        # the root form of the word
    print(f"POS: {token.pos_}")            # grammatical feature of the word
    print(f"Tag: {token.tag_}")            # detailed grammatical features of the word
    print(f"Dependency: {token.dep_}")     # the role of the word
    print(f"Shape: {token.shape_}")        # character structure
    print(f"Is alpha: {token.is_alpha}")   # checks if the word consists only of alphabetical characters.
    print(f"Is stop: {token.is_stop}")     # whether the word is a stopword
    print(f"Morphologic: {token.morph}")   # morphological features
    print(f"Is plural: {'Number=Plur' in token.morph}") # check word is plural

    print()
