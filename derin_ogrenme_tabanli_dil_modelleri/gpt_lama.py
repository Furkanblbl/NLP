"""
text generate with gpt-2
"""

# import libraries
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# model definition
model_name = "gpt2"

# tokenizer definition and model creation
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

model = GPT2LMHeadModel.from_pretrained(model_name)

# text generation
text = "I was reading a book"

# tokenization
inputs = tokenizer.encode(text, return_tensors="pt")

# text generation
outputs = model.generate(inputs, max_length=50)

# deacode
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True) # discard special tokens(sentence start, finish token EOS)

print(generated_text)