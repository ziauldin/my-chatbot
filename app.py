import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("saved_dialoGPT_model")
    model = AutoModelForCausalLM.from_pretrained("saved_dialoGPT_model")
    return tokenizer, model

tokenizer, model = load_model()

# Generate response
def generate_response(user_input):
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    output_ids = model.generate(input_ids, max_length=50, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

# Streamlit interface
st.title("💬 DialoGPT Chatbot")
user_input = st.text_input("You:", "")

if st.button("Send") and user_input:
    bot_response = generate_response(user_input)
    st.text_area("Bot:", value=bot_response, height=100)
