import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

st.set_page_config(
    page_title="AI Chatbot", 
    page_icon="💬", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load model
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    return tokenizer, model

tokenizer, model = load_model()

def generate_response(user_input):
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    output_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

# CSS styling
st.markdown("""
    <style>
    :root {
        --primary: #6e48aa;
        --secondary: #9d50bb;
        --dark: #1a1a2e;
        --light: #f8f9fa;
        --user: #6e48aa;
        --bot: #e9ecef;
    }
    
    body {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .main {
        max-width: 900px;
        margin: 0 auto;
        padding: 2rem 1rem;
    }
    
    .header {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .header h1 {
        color: var(--primary);
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .header p {
        color: var(--dark);
        opacity: 0.8;
    }
    
    .chat-container {
        background-color: white;
        border-radius: 18px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        height: 65vh;
        overflow-y: auto;
        margin-bottom: 1.5rem;
    }
    
    .chat-bubble {
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
        border-radius: 18px;
        max-width: 70%;
        position: relative;
        line-height: 1.5;
        animation: fadeIn 0.3s ease-out;
    }
    
    .user-msg {
        background: linear-gradient(135deg, var(--user) 0%, var(--secondary) 100%);
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 4px;
    }
    
    .bot-msg {
        background-color: var(--bot);
        color: var(--dark);
        margin-right: auto;
        border-bottom-left-radius: 4px;
    }
    
    .input-container {
        display: flex;
        gap: 0.5rem;
    }
    
    .stTextInput>div>div>input {
        border-radius: 12px !important;
        padding: 12px 16px !important;
        border: 2px solid #e0e0e0 !important;
    }
    
    .stButton>button {
        border-radius: 12px !important;
        padding: 0.5rem 1.5rem !important;
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%) !important;
        color: white !important;
        border: none !important;
        font-weight: 500 !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(110, 72, 170, 0.3);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #c1c1c1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #a8a8a8;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 2rem;
        color: var(--dark);
        opacity: 0.7;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header">
    <h1>✨ Smart AI Assistant</h1>
    <p>Ask me anything and I'll do my best to help!</p>
</div>
""", unsafe_allow_html=True)

# Chat Container
st.markdown('<div class="chat-container" id="chat-container">', unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    if role == "user":
        st.markdown(f'<div class="chat-bubble user-msg">{content}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-bubble bot-msg">{content}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# User input
input_container = st.container()
with input_container:
    user_input = st.text_input(
        "Type your message here...", 
        key="input",
        label_visibility="collapsed",
        placeholder="Ask me anything..."
    )
    col1, col2, col3 = st.columns([3,1,3])
    with col2:
        send_button = st.button("Send", use_container_width=True)

if (send_button or user_input) and user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Generate response
    with st.spinner('Thinking...'):
        bot_response = generate_response(user_input)
    
    # Add bot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    
    # Rerun to show new messages
    st.experimental_rerun()

# Footer
st.markdown("""
<div class="footer">
    <p>Powered by DialoGPT | Made with Streamlit</p>
</div>
""", unsafe_allow_html=True)

# Auto-scroll to bottom of chat
st.markdown("""
<script>
window.addEventListener('load', function() {
    var chatContainer = document.getElementById('chat-container');
    chatContainer.scrollTop = chatContainer.scrollHeight;
});
</script>
""", unsafe_allow_html=True)
