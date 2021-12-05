import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

def getNumberAnswer(explanation):
    """
    """
    for index, value in enumerate(reversed(explanation)):
        if value == '#':
            return explanation[-index+1:]

    
# Open the test.jsonl as a json list.  The data set
# is from https://github.com/openai/grade-school-math.
with open('./test.jsonl', 'r') as json_file:
    jsonl = list(json_file)

questions = []
numberAnswers = []
wordAnswers = []

# Loop for all the data entries in test.jsonl and
# use the question and answer fields to construct a sentence.
for dataEntry in jsonl:
    result = json.loads(dataEntry)
    questions.append(result["question"])
    numberAnswers.append(getNumberAnswer(result["answer"]))
    wordAnswers.append(result["answer"])

print("Preprocess completed")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = GPT2Tokenizer.from_pretrained('finetuned')
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained('finetuned').to(device)
print("Model loaded")

# this is a single input batch with size 3
texts = questions[:3]

encoding = tokenizer(texts, padding=True, return_tensors='pt').to(device)
with torch.no_grad():
    generated_ids = model.generate(**encoding, max_length=100)
generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

accuracy = 0
for index, value in enumerate(texts):
    print(value)
    print(generated_texts[index])


