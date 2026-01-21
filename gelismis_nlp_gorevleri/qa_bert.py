from transformers import BertTokenizer, BertForQuestionAnswering
import torch

import warnings
warnings.filterwarnings("ignore")

# finetuned bert model on squad dataset
model_name = "bert-large-uncased_whole-word-masking-finetuned-squad"

# bert tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)

model = BertForQuestionAnswering.from_pretrained(model_name)
