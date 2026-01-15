""""
Solve Classification problem (Sentiment Analysis in NLP) with RNN (Deep Learning based Language Model)

motion analyse -> labeling a sentence

"""

# import libraries
import pandas as pd
import numpy as np

from gensim.models import Word2Vec

from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Embedding

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# create dataset
data = {
    "text": [
        "Yemekler harikaydı, servis çok hızlıydı.",
        "Garson çok ilgisizdi ve yemek geç geldi.",
        "Atmosfer çok keyifliydi, tekrar gelirim.",
        "Yemekler soğuktu ve tatsızdı.",
        "Tatlılar gerçekten mükemmeldi.",
        "Porsiyonlar çok küçüktü ve pahalıydı.",
        "Servis kalitesi üst seviyedeydi.",
        "Yemek bekleme süresi aşırı uzundu.",
        "Lezzetli bir akşam yemeğiydi.",
        "Hijyen konusunda ciddi sorunlar vardı.",

        "Garsonlar güler yüzlüydü.",
        "Yemekler bayattı.",
        "Menü oldukça zengindi.",
        "Servis berbattı.",
        "Etler tam kıvamındaydı.",
        "Fiyat performans çok kötüydü.",
        "Mekan temiz ve düzenliydi.",
        "Siparişimiz yanlış geldi.",
        "Tatlar çok dengeliydi.",
        "Bir daha asla gitmem.",

        "Yemekler taze ve lezzetliydi.",
        "Garsonlar kaba davrandı.",
        "Sunumlar çok şıktı.",
        "Ortam aşırı gürültülüydü.",
        "Hizmetten çok memnun kaldık.",
        "Yemeklerin tadı berbattı.",
        "Servis hızlı ve sorunsuzdu.",
        "Çok geç servis yapıldı.",
        "Ailece keyifli vakit geçirdik.",
        "Masalar kirliydi.",

        "Tatlı menüsü çok başarılıydı.",
        "İçecekler bayattı.",
        "Mekan ambiyansı çok güzeldi.",
        "Yemekler çok yağlıydı.",
        "Garson önerileri çok yerindeydi.",
        "Servis hiç ilgilenmedi.",
        "Her şey kusursuzdu.",
        "Beklediğimize hiç değmedi.",
        "Yemekler ağızda dağılıyordu.",
        "Fiyatlar aşırı pahalıydı.",

        "Hizmet kalitesi gerçekten yüksekti.",
        "Yemekler yanmıştı.",
        "Tekrar gelmeyi düşünüyoruz.",
        "Sunum çok özensizdi.",
        "Lezzetler bizi mutlu etti.",
        "Servis rezaletti.",
        "Mekan ferah ve rahattı.",
        "Yemekler hiç sıcak değildi.",
        "Kesinlikle tavsiye ederim.",
        "Garsonlar hiç yardımcı olmadı.",

        "Yemek deneyimi çok başarılıydı.",
        "Hijyen çok zayıftı.",
        "Tatlılar efsaneydi.",
        "Siparişler sürekli karıştı.",
        "Mutfak gerçekten başarılıydı.",
        "Lezzet sıfırdı.",
        "Servis ekibi çok iyiydi.",
        "Bekleme süresi aşırı fazlaydı.",
        "Her şey çok lezzetliydi.",
        "Parasını hak etmiyor.",

        "Ortam huzurluydu.",
        "Yemekler tatsızdı.",
        "Garsonlar çok profesyoneldi.",
        "Mekan beklentimi karşılamadı.",
        "Yemekler tam zamanında geldi.",
        "Tatlılar berbattı.",
        "Sunum ve lezzet harikaydı.",
        "Bir daha gelmem.",
        "Çok keyif aldık.",
        "Hizmet çok kötüydü."
    ],

    "label": [
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",

        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",

        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",

        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",

        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",

        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",

        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative"
    ]
}

df = pd.DataFrame(data)

# tokenization, padding, label encoding, train test split

# tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df["text"])
sequences = tokenizer.text_to_sequences(df["text"])
word_index = tokenizer.word_index
