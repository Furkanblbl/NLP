from transformers import MarianMTModel, MarianTokenizer

model_name = "Helsinki-NLP/opus-tatoeba-en-tr"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

text = "Hello, what is going on?"

translated_text = model.generate(**tokenizer(text, return_tensors="pt", padding=True))

translated_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated_text]
print(f"Translated Text: {translated_text[0]}")