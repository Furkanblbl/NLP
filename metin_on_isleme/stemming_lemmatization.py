import nltk

nltk.download("wordnet") # wordnet: lemmatization islemi icin gerekli veri tabani

from nltk.stem import PorterStemmer

# porter stemmer nesnesi olustur
stemmer = PorterStemmer()
words = ["running", "runner", "ran", "runs", "better", "go", "went"]

# kelimelerin stemleri bulunur, porter stammerin stem() fonksiyonu kullanılır
stems = [stemmer.stem(w) for w in words]
print(f"Stem: {stems}")

# lemmatization

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

lemmas = [lemmatizer.lemmatize(w, pos="v") for w in words]
print(f"lemmas: {lemmas}")