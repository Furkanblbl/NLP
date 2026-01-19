"""
text generate with gpt-2
"""

# import libraries
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM # llama

# model definition
model_name = "gpt2"
model_name_llama = "huggyLLama/llama-7b"

# tokenizer definition and model creation
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer_llama = AutoTokenizer.from_pretrained(model_name_llama)

model = GPT2LMHeadModel.from_pretrained(model_name)
model_llama = AutoModelForCausalLM.from_pretrained(model_name_llama)

# text generation
text = "I was reading a book"

# tokenization
inputs = tokenizer.encode(text, return_tensors="pt")
inputs_llama = tokenizer_llama(text, return_tensors="pt")

# text generation
outputs = model.generate(inputs, max_length=50)
outputs_llama = model_llama.generate(inputs_llama.input_ids, max_length=50)

# deacode
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True) # discard special tokens(sentence start, finish token EOS)
generated_text_llama = tokenizer_llama.decode(outputs_llama[0], skip_special_tokens=True)

print(generated_text)
print("**************************")
print(generated_text_llama)