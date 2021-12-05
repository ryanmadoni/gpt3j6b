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

# Sampling from test.jsonl
sampling = 3
texts = questions[:sampling]
maxAnswerLength = max([len(sentence) for sentence in wordAnswers[:sampling]])

print(f"Sampling {sampling} from {len(questions)} testing data points")
print("Preprocess completed")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = GPT2Tokenizer.from_pretrained('finetuned')
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained('finetuned').to(device)
print("Model loaded")

encoding = tokenizer(texts, padding=True, return_tensors='pt').to(device)
with torch.no_grad():
    generated_ids = model.generate(**encoding, max_length=maxAnswerLength)
generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

accuracy = 0
for index, value in enumerate(texts):
    print(value)
    print(generated_texts[index])

with open(f"./results.jsonl", "w") as file:

    # Loop for each entry to be saved in the JSONL file.
    for index, value in enumerate(texts):
        generatedWordAnswer = generated_texts[index]
        generatedNumberAnswer = getNumberAnswer(generatedWordAnswer)

        datapoint = {"question": value,
                     "answer": wordAnswers[index],
                     "generatedWordAnswer": generatedWordAnswer,
                     "numberAnswer": numberAnswers[index],
                     "generatedNumberAnswer": generatedNumberAnswer}
        
        accuracy += (numberAnswers[index] == generatedNumberAnswer)

        json.dump(datapoint, file) # Save each entry.
        file.write("\n") # pad each entry with a '\n'.

        print(accuracy)
        print(accuracy / sampling)


