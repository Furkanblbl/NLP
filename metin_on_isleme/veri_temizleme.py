# bosluklari kaldir
text = "Hello,    World!, 2023"
cleaned_text1 = " ".join(text.split())

print(f"text: {text}\ncleaned_text1: {cleaned_text1}")

# buyuk -> kucuk harf cevir
text = "HEllo, World! 2023"
cleaned_text2 = text.lower()
print(f"text: {text}\ncleaned_text2: {cleaned_text2}")

# noktalama isaretlerini kaldir
import string

text = "Hello, World! 2023"
cleaned_text3 = text.translate(str.maketrans("","", string.punctuation))
print(f"text: {text}\ncleaned_text3: {cleaned_text3}")

# ozel karaketerleri kaldir
import re

text = "Hello, World! 2023%"

cleaned_text4 = re.sub(r"[^A-Za-z0-9\s]", "",text)
print(f"text: {text}\ncleaned_text4: {cleaned_text4}")

# yazim hatalarini duzelt
from textblob import TextBlob # metin analizlerinde kullanilir

text = "Hellio, Wirld!, 2023"
cleaned_text5 = TextBlob(text).correct() #correct: yazim hatalarini duzeltir
print(f"text: {text}\ncleaned_text5: {cleaned_text5}")

# html yada url etiketlerini kaldir
from bs4 import BeautifulSoup

html_text = "<div>Hello, World! 2023</div>"
cleaned_text6 = BeautifulSoup(html_text, "html.parser").getText()

print(f"text: {html_text}\ncleaned_text5: {cleaned_text6}")
