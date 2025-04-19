import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from streamlit.components.v1 import html

# Initialize session state for chat history
if 'history' not in st.session_state:
    st.session_state.history = []

# Load model with caching and error handling
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        
        # Set pad token to eos token if not already set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Generate response function with improved error handling
def generate_response(user_input, tokenizer, model):
    try:
        if not tokenizer or not model:
            return "Model not loaded properly. Please try again."
            
        # Encode input with attention mask
        inputs = tokenizer.encode_plus(
            user_input + tokenizer.eos_token,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True
        )
        
        # Generate response with attention mask
        output_ids = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=1000,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8
        )
        
        response = tokenizer.decode(
            output_ids[:, inputs.input_ids.shape[-1]:][0], 
            skip_special_tokens=True
        )
        return response
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

# App layout configuration
st.set_page_config(
    page_title="AI Conversational Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Main container */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Chat container */
    .chat-container {
        background-color: white;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        padding: 2rem;
        margin-bottom: 2rem;
        min-height: 60vh;
        max-height: 60vh;
        overflow-y: auto;
    }
    
    /* User message */
    .user-message {
        background-color: #e3f2fd;
        border-radius: 15px 15px 0 15px;
        padding: 12px 16px;
        margin: 8px 0;
        max-width: 80%;
        float: right;
        clear: both;
    }
    
    /* Bot message */
    .bot-message {
        background-color: #f1f1f1;
        border-radius: 15px 15px 15px 0;
        padding: 12px 16px;
        margin: 8px 0;
        max-width: 80%;
        float: left;
        clear: both;
    }
    
    /* Input area */
    .stTextInput>div>div>input {
        border-radius: 20px !important;
        padding: 12px 16px !important;
    }
    
    /* Button */
    .stButton>button {
        border-radius: 20px !important;
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%) !important;
        color: white !important;
        font-weight: bold !important;
        border: none !important;
        padding: 10px 24px !important;
        width: 100%;
    }
    
    /* Title */
    .title-text {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: #2c3e50 !important;
        text-align: center;
        margin-bottom: 1.5rem !important;
    }
    
    /* Subheader */
    .subheader-text {
        font-size: 1.1rem !important;
        color: #7f8c8d !important;
        text-align: center;
        margin-bottom: 2rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Header section
st.markdown('<h1 class="title-text">AI Conversational Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="subheader-text">Powered by Microsoft DialoGPT • Natural Language Processing</p>', unsafe_allow_html=True)

# Load model
tokenizer, model = load_model()

# Chat container
chat_container = st.container()
with chat_container:
    st.markdown('<div class="chat-container" id="chat-container">', unsafe_allow_html=True)
    
    # Display chat history
    for message in st.session_state.history:
        if message['role'] == 'user':
            st.markdown(f'<div class="user-message"><b>You:</b> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message"><b>Assistant:</b> {message["content"]}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Input area with proper label
with st.form("chat_form"):
    col1, col2 = st.columns([4, 1])
    with col1:
        user_input = st.text_input(
            "Type your message", 
            placeholder="Type your message here...", 
            key="user_input",
            label_visibility="collapsed"
        )
    with col2:
        send_button = st.form_submit_button("Send")

# Handle user input
if send_button and user_input.strip() != "":
    # Add user message to history
    st.session_state.history.append({"role": "user", "content": user_input})
    
    # Generate and add bot response
    with st.spinner("Thinking..."):
        bot_response = generate_response(user_input, tokenizer, model)
        st.session_state.history.append({"role": "bot", "content": bot_response})
    
    # Rerun to update the display
    st.rerun()

# JavaScript to auto-scroll to bottom of chat
html("""
<script>
    window.onload = function() {
        var container = document.getElementById('chat-container');
        container.scrollTop = container.scrollHeight;
    }
    
    // Scroll to bottom after new message
    if (window.Streamlit) {
        Streamlit.onMessage(function() {
            var container = document.getElementById('chat-container');
            container.scrollTop = container.scrollHeight;
        });
    }
</script>
""")
