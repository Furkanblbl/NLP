import nltk # natural language toolkit

nltk.download("punkt_tab") # metni kelime ve ümle bazında tokenlara ayirmaya yarar

text = "Hello, World!, How are you? Hello, hi ..."

# kelime tokenizasyonu: word_tokenize: metni kelimelere ve ayirir,
# noktalama isaretlerini ve bosluklar ayri birer token olarak elde edilir

word_tokens = nltk.word_tokenize(text)
print(word_tokens)

# cumle tokenizasyonu: sent_tokenize: metni cumlelere ayirir. her bir cümle birer token olarak alir

sent_tokens = nltk.sent_tokenize(text)
print(sent_tokens)