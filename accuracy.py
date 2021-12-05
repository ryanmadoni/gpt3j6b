import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from nltk.translate.bleu_score import sentence_bleu

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
samples = 3
texts = questions[:samples]

print(f"Sampling {samples} from {len(questions)} testing data points")
print("Preprocess completed")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = GPT2Tokenizer.from_pretrained('finetuned')
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained('finetuned').to(device)
print("Model loaded\nProcessing:")

generated_texts = []

for index, text in enumerate(texts):
    print(f"Porcessing text {index + 1} / {samples}")

    encoding = tokenizer([text], padding=True, return_tensors='pt').to(device)
    with torch.no_grad():
        generated_ids = model.generate(**encoding, max_length=len(text))

    generated_texts.append(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))

print("Calculating accuracy and BLEU Scores")

accuracy = 0.0
bleu = 0.0

with open(f"./results.jsonl", "w") as file:

    # Loop for each entry to be saved in the JSONL file.
    for index, value in enumerate(texts):
        generatedWordAnswer = generated_texts[index]
        generatedNumberAnswer = getNumberAnswer(generatedWordAnswer)
        bleuScore = sentence_bleu([wordAnswers[index].split(' ')], generatedWordAnswer.split(' '))

        datapoint = {"question": value,
                     "wordAnswer": wordAnswers[index],
                     "generatedWordAnswer": generatedWordAnswer,
                     "numberAnswer": numberAnswers[index],
                     "generatedNumberAnswer": generatedNumberAnswer,
                     "BLEU": bleuScore}

        bleu += bleuScore
        
        accuracy += (numberAnswers[index] == generatedNumberAnswer)

        json.dump(datapoint, file) # Save each entry.
        file.write("\n") # pad each entry with a '\n'.

    print(f"Accuracy: {accuracy / samples}, {accuracy} / {samples}")
    print(f"Average BLEU Score: {bleu / samples}")


