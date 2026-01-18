"""
Sentence generation

Lstm traib with text data
generate test data with gpt
"""

# import libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences 

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Create train dataset
texts = [
    "Bugün hava çok güzel.",
    "Sabah erkenden yürüyüşe çıktım.",
    "Kahvaltıda peynir ve zeytin yedim.",
    "Dışarıda hafif bir rüzgar var.",
    "Akşam yemeğinde makarna yaptık.",
    "Bilgisayar başında uzun süre çalıştım.",
    "Kahve içmeden güne başlayamıyorum.",
    "Bugün işler beklediğimden yoğundu.",
    "Yağmur yağınca hava serinledi.",
    "Müzik dinlerken ders çalışmayı seviyorum.",

    "Hafta sonu arkadaşlarımla buluştum.",
    "Sabahları erken uyanmak zor geliyor.",
    "Kitap okumak beni çok rahatlatıyor.",
    "Telefonumun şarjı hızlı bitiyor.",
    "Bugün trafikte çok bekledim.",
    "Öğle yemeğinde çorba içtim.",
    "Akşam yürüyüş yapmak iyi geliyor.",
    "Yeni bir dizi izlemeye başladım.",
    "Bilgisayarım biraz yavaş çalışıyor.",
    "Bugün kendimi daha enerjik hissediyorum.",

    "Markete gitmem gerekiyordu.",
    "Evde temizlik yapmak zaman aldı.",
    "Sabah kahvemi balkonda içtim.",
    "Güneşli havalar beni mutlu ediyor.",
    "Bugün çok fazla mail attım.",
    "Akşam erken uyumayı planlıyorum.",
    "Uzun süredir spor yapamıyorum.",
    "Ders notlarını tekrar gözden geçirdim.",
    "Yeni bir şeyler öğrenmek hoşuma gidiyor.",
    "Hava akşam saatlerinde serinledi.",

    "Toplantı beklediğimden uzun sürdü.",
    "Bugün dışarı çıkmak istemedim.",
    "Bilgisayarda kod yazmak keyifliydi.",
    "Sabah alarmı duymadım.",
    "Öğleden sonra kısa bir mola verdim.",
    "Kulaklıkla müzik dinledim.",
    "Bugün çok fazla su içtim.",
    "Akşam yemeği biraz geç oldu.",
    "Hafif bir baş ağrım var.",
    "Gün hızlı geçti.",

    "Yeni bir proje üzerinde çalışıyorum.",
    "Bugün notlarımı düzenledim.",
    "Kahve makinesi bozuldu.",
    "Sabah yürüyüşü iyi hissettirdi.",
    "Bilgisayar ekranına uzun süre baktım.",
    "Akşam film izlemeyi düşünüyorum.",
    "Bugün daha verimli çalıştım.",
    "Telefon görüşmeleri çok vakit aldı.",
    "Dışarıda hava kapalıydı.",
    "Günü planlamak işimi kolaylaştırıyor.",

    "Sabahları sessizlik hoşuma gidiyor.",
    "Bugün biraz yorgun hissediyorum.",
    "Yeni bir şey denemek istiyorum.",
    "Öğleden sonra kahve içtim.",
    "Bilgisayarda dosyaları düzenledim.",
    "Akşam erken uyudum.",
    "Bugün motivasyonum yüksekti.",
    "Uzun bir gün oldu.",
    "Kısa bir yürüyüş yaptım.",
    "Gün sonunda dinlenmek iyi geldi.",
        "Sabah pencereyi açınca serin hava geldi.",
    "Bugün yapılacak işlerimi listeledim.",
    "Bilgisayarı açınca gün başladı.",
    "Kahvaltıdan sonra kısa bir mola verdim.",
    "Hava bulutlu ama yağmur yok.",
    "Öğle arasında biraz dinlendim.",
    "Telefonuma çok fazla bildirim geldi.",
    "Yeni bir şeyler denemek istiyorum.",
    "Bugün evde çalışmak daha rahattı.",
    "Akşamüstü kahve içtim.",

    "Bilgisayar başında saatler geçti.",
    "Sabah alarmı biraz geç çaldı.",
    "Bugün daha sakin bir gündü.",
    "Notlarımı düzenlemek iyi hissettirdi.",
    "Kısa bir yürüyüş yaptım.",
    "Akşam yemeği hafif oldu.",
    "Hava akşama doğru serinledi.",
    "Bugün daha az konuşmak istedim.",
    "Uzun süre ekrana baktım.",
    "Günü erken kapattım.",

    "Sabah güneşli bir hava vardı.",
    "Kahve kokusu mutfağı sardı.",
    "Bugün evden çıkmadım.",
    "Bilgisayarda eski dosyaları sildim.",
    "Öğleden sonra müzik dinledim.",
    "Telefonumu sessize aldım.",
    "Bugün zaman hızlı geçti.",
    "Kısa notlar aldım.",
    "Akşam dışarı çıkmadım.",
    "Sessiz bir gün oldu.",

    "Sabahları erken kalkmak zor geliyor.",
    "Bugün biraz dalgındım.",
    "Yapılacak işler birikti.",
    "Kahve içince daha iyi hissettim.",
    "Hava kapalı ama sıcak.",
    "Bilgisayarım güncelleme yaptı.",
    "Akşam planlarım iptal oldu.",
    "Bugün fazla yoruldum.",
    "Biraz dinlenmeye ihtiyacım var.",
    "Gün sonunda rahatladım.",

    "Sabah haberleri okudum.",
    "Bugün dışarı çıkmak istemedim.",
    "Bilgisayarda kod yazdım.",
    "Öğle yemeğini geç yedim.",
    "Hava temizdi.",
    "Telefonla uzun bir konuşma yaptım.",
    "Bugün üretken geçti.",
    "Akşam erken uyumayı düşünüyorum.",
    "Sessizlik iyi geldi.",
    "Gün yavaş ilerledi.",

    "Sabah pencere açıktı.",
    "Bugün kendimi daha iyi hissediyorum.",
    "Kahvaltı hazırlamak zaman aldı.",
    "Bilgisayar fanı çok ses yapıyor.",
    "Öğleden sonra biraz uyudum.",
    "Akşamüstü hava karardı.",
    "Bugün daha odaklıydım.",
    "Telefonumun şarjı bitti.",
    "Biraz mola verdim.",
    "Günü toparladım.",

    "Sabahları kahve içmek alışkanlık oldu.",
    "Bugün daha az stresliydi.",
    "Bilgisayarda uzun süre kaldım.",
    "Öğle arasında dışarı baktım.",
    "Hava hafif rüzgarlıydı.",
    "Akşam evdeydim.",
    "Bugün not almadım.",
    "Sessiz bir ortam vardı.",
    "Gün içinde çok düşünmedim.",
    "Akşam dinlenmek iyi geldi.",

    "Sabah erken uyandım.",
    "Bugün biraz halsizdim.",
    "Bilgisayarım hızlı çalıştı.",
    "Öğleden sonra kahve içmedim.",
    "Hava çok sıcak değildi.",
    "Akşam yemeğini evde yedim.",
    "Bugün planlar değişti.",
    "Kısa bir ara verdim.",
    "Gün sakin geçti.",
    "Akşam sessizdi.",

    "Sabah hafif bir serinlik vardı.",
    "Bugün kendime zaman ayırdım.",
    "Bilgisayarda yazı yazdım.",
    "Öğle yemeği basitti.",
    "Hava açık ve sakindi.",
    "Telefonla fazla ilgilenmedim.",
    "Bugün daha dengeliydi.",
    "Akşam evde kaldım.",
    "Yorgunluk biraz geçti.",
    "Günü kapattım."
]

# tokenization, padding, label encoding

#tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts) # learn frequences on the texts
total_words = len(tokenizer.word_index) + 1 

# Create n-gram arras
input_sequences = []
for text in texts:
    token_list = tokenizer.texts_to_sequences([text])[0]
    # create n-gram array every text
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)
# print(input_sequences)

# padding
max_sequence_lengt = max(len(x) for x in input_sequences)
# print(f"max_sequence_lengt: {max_sequence_lengt}")
input_sequences = pad_sequences(input_sequences, maxlen = max_sequence_lengt, padding="pre")

# X(input) and y(target)
X = input_sequences[:,:-1]

y = input_sequences[:,-1]
y = tf.keras.utils.to_categorical(y, num_classes = total_words) # one hot encoding


# Create lstm model, compile, train, evaluate

model = Sequential()

# embedding
model.add(Embedding(total_words, 50, input_length=X.shape[1]))

# lstm
model.add(LSTM(100, return_sequences = False))

# output
model.add(Dense(total_words, activation="softmax"))

# model compile
model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

# model training
model.fit(X, y, epochs=100, verbose=1)

# model prediction