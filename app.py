import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

st.set_page_config(page_title="AI Chatbot", page_icon="💬", layout="centered")

# Load model
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    return tokenizer, model

tokenizer, model = load_model()

def generate_response(user_input):
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    output_ids = model.generate(input_ids, max_length=50, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

# CSS styling
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
        padding: 2rem;
    }
    .chat-container {
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 0 12px rgba(0,0,0,0.05);
    }
    .chat-bubble {
        padding: 0.8rem;
        margin-bottom: 1rem;
        border-radius: 10px;
    }
    .user-msg {
        background-color: #d9eaff;
        text-align: right;
    }
    .bot-msg {
        background-color: #f1f1f1;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="chat-container">', unsafe_allow_html=True)
st.title("💬 Smart Chatbot")

# Chat Interface
user_input = st.text_input("Type your message:", key="input")

if st.button("Send") and user_input:
    bot_response = generate_response(user_input)
    
    st.markdown(f'<div class="chat-bubble user-msg">{user_input}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="chat-bubble bot-msg">{bot_response}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
