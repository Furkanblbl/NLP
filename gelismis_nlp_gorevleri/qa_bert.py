from transformers import BertTokenizer, BertForQuestionAnswering
import torch

import warnings
warnings.filterwarnings("ignore")

# finetuned bert model on squad dataset
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"

# bert tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)

model = BertForQuestionAnswering.from_pretrained(model_name)

# function to answer questions based on context
def predict_answer(context, question):
    """_summary_
        aim: answer the question based on the given context using bert model
        1) tokenize the inputs
        2) run the model
        3) get the most highlighted start and end indexes
        4) take answer tokens from token ids
        5) combine tokens to form the final answer string

    Args:
        context (str): context paragraph
        question (str): question related to the context
    """

    # tokenize the inputs
    encoding = tokenizer.encode_plus(question, context, return_tensors="pt", max_length=512, truncation=True)

    # input tensors
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"] # attention mask to avoid attending to padding tokens

    # run the model
    with torch.no_grad():
        start_scores, end_scores = model(input_ids, attention_mask=attention_mask, return_dict = False)
    
    # most highlighted start and end indexes
    start_index = torch.argmax(start_scores, dim=1).item() # start index
    end_index = torch.argmax(end_scores, dim=1).item() # end index

    # take answer tokens from token ids
    answer_tokens = tokenizer.convert_ids_to_tokens(input_ids[0][start_index:end_index+1])

    # combine tokens to form the final answer string
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    return answer

question = "What is the capital of France?"
context = "France, in Western Europe, encompasses medieval cities, alpine villages and Mediterranean beaches. Paris, its capital, is famed for its fashion houses, classical art museums including the Louvre and monuments like the Eiffel Tower."

answer = predict_answer(context, question)
print("Question:", question)
print("Answer:", answer)
