import pandas as pd
import re
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM

file_path = "dialogs.txt"
data = pd.read_csv(file_path, sep="\t", header=None, names=["input", "response"])

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip()

data['input'] = data['input'].apply(clean_text)
data['response'] = data['response'].apply(clean_text)

if data.isnull().sum().any():
    data = data.dropna()

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

print(f"Training samples: {len(train_data)}")
print(f"Testing samples: {len(test_data)}")

model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_response(user_input):
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    output_ids = model.generate(input_ids, max_length=50, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

test_input = "Hello, how are you?"
test_response = generate_response(test_input)
print(f"Bot: {test_response}")

save_path = "./saved_dialoGPT_model"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"Model and tokenizer saved to: {save_path}")
