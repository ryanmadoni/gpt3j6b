import sys
import torch
import json
from transformers import (GPT2Tokenizer, GPT2LMHeadModel,
                          GPTNeoForCausalLM, AutoTokenizer)
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# "gpt2xl" or "gptneo"
model = "gptneo"  # The name of the model to be used.


def getNumberAnswer(explanation):
    """
    Takes as input an explanation for an answer, then computes and
    returns the numerical solution to the problem without explanation.
    """
    for index, value in enumerate(reversed(explanation)):

        # Check for the sentinel value, '#'.
        if value == '#':
            return explanation[-index+1:]  # Return the numerical solution.


# Open the test.jsonl as a json list.  The data set
# is from https://github.com/openai/grade-school-math.
with open('./test.jsonl', 'r') as json_file:
    jsonl = list(json_file)  # Open the jsonl file.

# Containers for the questions, the numerical answers
# and the word explanations with numerical answers.
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

# Update user messages.
print(f"Sampling {samples} from {len(questions)} testing data points")
print("Preprocessing completed")

generated_texts = []  # A container to store the generated texts.

# Check for what model to generate from.
if model == "gpt2xl":

    # Load the model, gpt2xl.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = GPT2Tokenizer.from_pretrained('finetuned')
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained('finetuned').to(device)
    print(f"Model, {model}, loaded\nProcessing Texts:")  # Update user message.

    # Loop for all the texts.
    for index, text in enumerate(texts):
        print(f"Processing text {index + 1}/{samples}")  # Update user message.

        # Generate the answer from the text.
        encoding = tokenizer([text], padding=True,
                             return_tensors='pt').to(device)

        with torch.no_grad():
            generated_ids = model.generate(**encoding, max_length=len(text))

        # Save the generated text for evaluation.
        generated_texts += tokenizer.batch_decode(generated_ids,
                                                  skip_special_tokens=True)

elif model == "gptneo":

    # Load the model, gptneo.
    model = GPTNeoForCausalLM.from_pretrained("finetuned").to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("finetuned")

    print(f"Model, {model}, loaded\nProcessing Texts:")  # Update user message.

    # Loop for all the texts.
    for index, text in enumerate(texts):
        print(f"Processing text {index + 1}/{samples}")  # Update user message.

        ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")

        # Add the length of the prompt tokens
        # to match with the mesh-tf generation.
        max_length = len(text) + ids.shape[1]

        # Generate the answer from the text.
        gen_tokens = model.generate(
                                    ids,
                                    do_sample=True,
                                    min_length=max_length,
                                    max_length=max_length,
                                    temperature=0.9,
                                    use_cache=True
                                    )

        # Save the generated text for evaluation.
        generated_texts += tokenizer.batch_decode(gen_tokens)[0]

else:
    print(f"Model, {model}, not supported")  # Print error message.
    sys.exit(1)  # Exit gracefully.

print("Calculating accuracy and BLEU Scores")  # Update user message.

# Variables to store the
# accuracy and BLEU scores.
accuracy = 0.0
bleu = 0.0

# Open a json file to store the results.
with open(f"./results.jsonl", "w") as file:

    # Load the smoothing function.
    smoothed = SmoothingFunction().method4

    # Loop for each entry to be saved in the JSONL file.
    for index, value in enumerate(texts):
        generatedWordAnswer = generated_texts[index]
        generatedNumberAnswer = getNumberAnswer(generatedWordAnswer)
        bleuScore = sentence_bleu([wordAnswers[index].split(' ')],
                                  generatedWordAnswer.split(' '),
                                  smoothing_function=smoothed)

        # Construct a dictionary to store in the jsonl file.
        datapoint = {"question": value,
                     "wordAnswer": wordAnswers[index],
                     "generatedWordAnswer": generatedWordAnswer,
                     "numberAnswer": numberAnswers[index],
                     "generatedNumberAnswer": generatedNumberAnswer,
                     "BLEU": bleuScore}

        # Add the BLEU score and accuracy to their runnning counts.
        bleu += bleuScore
        accuracy += (numberAnswers[index] == generatedNumberAnswer)

        json.dump(datapoint, file)  # Save each entry.
        file.write("\n")  # Pad each entry with a '\n'.

    # Calculate and display the average accuracy and BLEU scores.
    print(f"Accuracy: {accuracy / samples}, {accuracy} / {samples}")
    print(f"Average BLEU Score: {bleu / samples}")
