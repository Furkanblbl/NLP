import nltk
from nltk.corpus import stopwords

nltk.download("stopwords") # farkli dillerde en cok kullanilan stop words listesi

# ingilizce stop words analizi
stop_words_eng = set(stopwords.words("english"))
print(stop_words_eng)

text = "There are some examples of handling stop words from some texts."
text_list = text.split()
# eger word ingilizce stop words listesinde yoksa kelimeyi ekle
filtered_words = [word for word in text_list if word.lower() not in stop_words_eng]
print(f"filtered_words: {filtered_words}")

#turkce stop words analizi
stop_words_tr = set(stopwords.words("turkish"))
metin = "merhaba arkadaslar burada birkaç işimize yarayan güzel kelimeler yakalamaya çalışacağız. Bu nedenle şuan bu cümleyi kullanacağız."

metin_list = metin.split()
filtered_words_tr = [word for word in metin_list if word.lower() not in stop_words_tr]
print(f"filtered_words_tr: {filtered_words_tr}")

# kutuphanesiz stop words cikarimi

tr_stopwords = ["için", "bu", "ile", "mu", "mi", "özel"]
metin = "Bu bir denemedir. Amacımız bu metinde bulunan belirlediğimiz özel karakterleri elemek mi acaba?"

filtered_words = [word for word in metin.split() if word.lower() not in tr_stopwords]
print(f"filtered_words: {filtered_words}")
