from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2ForQuestionAnswering
import torch

import warnings
warnings.filterwarnings("ignore")

# finetuned gpt model on squad dataset
model_name = "gpt2"

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def generate_answer(context, question):

    input_text = f"Context: {context} Question: {question}. Please answer the question based on the context."

    # tokenize
    inputs = tokenizer.encode(input_text, return_tensors="pt")

    # run the model
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=512)

    # decode the answer
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    answer = answer.split("Answer:")[-1].strip()

    return answer

question = "What is the capital of France?"
context = "France, in Western Europe, encompasses medieval cities, alpine villages and Mediterranean beaches. Paris, its capital, is famed for its fashion houses, classical art museums including the Louvre and monuments like the Eiffel Tower."

answer = generate_answer(context, question)
print("Question:", question)
print("Answer:", answer)
