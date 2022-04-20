# Question Answering on SQUAD

import transformers
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer

# Load the dataset
datasets = load_dataset("squad_v2")
print(datasets)
print("Context: ", datasets["train"][0]["context"])
print("Question: ", datasets["train"][0]["question"])
print("Answers: ", datasets["train"][0]["answers"])

# Dataset.filter(), during training, there is only one possible answer.
# However, for the evaluation, there are several possible answers for each sample, which may be the same or different.
datasets["train"].filter(lambda x:len(x["answers"]["text"]) != 1)

# Preprocessing the training data
model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Pass to our tokenizer the question and the context together
context = datasets["train"][0]["context"]
question = datasets["train"][0]["question"]

tokenizer_inputs = tokenizer(question, context)
print(tokenizer_inputs)
tokenizer.decode(tokenizer_inputs["input_ids"])

