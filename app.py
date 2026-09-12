"""
CURAJ Admission Chatbot - Streamlit App
========================================
A simple ML-based chatbot for Central University of Rajasthan.
Uses TF-IDF + Logistic Regression for intent classification.
"""

import streamlit as st
import json
import pickle
import re
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---- Load the trained model, vectorizer, and dataset ----

@st.cache_resource
def load_resources():
    """Load model, vectorizer, and dataset once and cache them.
    Using st.cache_resource so they don't reload on every interaction.
    """
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    with open('dataset.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # build a lookup dict: tag -> list of responses
    responses = {}
    for intent in dataset['intents']:
        responses[intent['tag']] = intent['responses']
    
    # also build the full pattern matrix for cosine similarity fallback
    patterns = []
    tags = []
    for intent in dataset['intents']:
        for pattern in intent['patterns']:
            patterns.append(pattern)
            tags.append(intent['tag'])
    
    return model, vectorizer, responses, patterns, tags

model, vectorizer, responses, all_patterns, all_tags = load_resources()

# ---- Text preprocessing (same as training) ----

# hardcoded stopwords so we don't need nltk downloads
STOP_WORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'of',
    'it', 'its', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'we',
    'our', 'you', 'your', 'he', 'she', 'they', 'them', 'his', 'her', 'their',
    'has', 'have', 'had', 'was', 'were', 'be', 'been', 'being', 'will',
    'would', 'could', 'should', 'may', 'might', 'shall', 'am', 'just',
    'very', 'too', 'also', 'so', 'than', 'then', 'if', 'else', 'because',
    'as', 'with', 'by', 'from', 'up', 'down', 'out', 'off', 'over', 'under',
    'again', 'once', 'here', 'there', 'all', 'each', 'every', 'both',
    'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same',
    'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'between', 'did', 'doing', 'until', 'while'
}

def simple_lemmatize(word):
    """Basic lemmatization by removing common suffixes."""
    if word.endswith('ies') and len(word) > 4:
        return word[:-3] + 'y'
    if word.endswith('es') and len(word) > 4:
        return word[:-2]
    if word.endswith('s') and not word.endswith('ss') and len(word) > 3:
        return word[:-1]
    if word.endswith('ing') and len(word) > 5:
        return word[:-3]
    if word.endswith('ed') and len(word) > 4:
        return word[:-2]
    return word

def clean_text(text):
    """Preprocess user input the same way we preprocessed training data.
    This is important - the model expects the same format it was trained on.
    """
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9\s.]', '', text)
    words = text.split()
    cleaned_words = []
    for word in words:
        if word not in STOP_WORDS:
            lemma = simple_lemmatize(word)
            cleaned_words.append(lemma)
    return ' '.join(cleaned_words)


# ---- Prediction logic ----

# threshold below which we don't trust the model's prediction
CONFIDENCE_THRESHOLD = 0.35

# for cosine similarity fallback
all_patterns_cleaned = [clean_text(p) for p in all_patterns]
all_patterns_vectorized = vectorizer.transform(all_patterns_cleaned)

def get_response(user_input):
    """Predict the intent and return an appropriate response.
    
    Strategy:
    1. Clean the input text
    2. Use the LR model to predict intent
    3. If confidence is high enough, use the model's prediction
    4. If confidence is low, fall back to cosine similarity
    5. If cosine similarity is also low, return a generic response
    """
    cleaned = clean_text(user_input)
    
    # handle empty input
    if not cleaned.strip():
        return "Could you please type your question? I can help with CURAJ admissions, fees, courses, and eligibility."
    
    # get model prediction and confidence
    query_vec = vectorizer.transform([cleaned])
    predicted_tag = model.predict(query_vec)[0]
    proba = model.predict_proba(query_vec)[0]
    confidence = max(proba)
    
    # if model is confident enough, use its prediction
    if confidence >= CONFIDENCE_THRESHOLD:
        if predicted_tag in responses:
            return random.choice(responses[predicted_tag])
    
    # fallback: use cosine similarity to find the closest match
    similarities = cosine_similarity(query_vec, all_patterns_vectorized).flatten()
    best_idx = similarities.argmax()
    best_score = similarities[best_idx]
    
    if best_score >= 0.15:
        fallback_tag = all_tags[best_idx]
        if fallback_tag in responses:
            return random.choice(responses[fallback_tag])
    
    # if nothing matches well enough, give a generic helpful response
    return ("I'm not sure I understand that question. I can help you with:\n"
            "- **Admissions** - CUET process, how to apply\n"
            "- **Courses** - UG and PG programmes available\n"
            "- **Fees** - Fee structure for all programmes\n"
            "- **Eligibility** - Requirements for specific courses\n"
            "- **Hostel** - Accommodation and mess details\n"
            "- **Campus** - Facilities and contact info\n\n"
            "Try asking something like: *'What is the fee for MSc Computer Science?'*")


# ---- Streamlit UI ----

# page config
st.set_page_config(
    page_title="CURAJ Chatbot",
    page_icon="🎓",
    layout="centered"
)

# simple custom styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 10px 0;
    }
    .stChatMessage {
        padding: 8px;
    }
</style>
""", unsafe_allow_html=True)

# header section
st.markdown("<h1 style='text-align: center;'>🎓 CURAJ Admission Chatbot</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "Ask me about admissions, fees, courses, eligibility, and more at "
    "Central University of Rajasthan"
    "</p>",
    unsafe_allow_html=True
)
st.divider()

# initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! 👋 I'm the CURAJ Chatbot. I can help you with information about admissions, courses, fees, eligibility, CUET requirements, and campus facilities. What would you like to know?"}
    ]

# display all previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# chat input
if user_input := st.chat_input("Type your question here..."):
    # display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # get bot response
    response = get_response(user_input)
    
    # display bot response
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

# sidebar with useful info
with st.sidebar:
    st.markdown("### 💡 Sample Questions")
    st.markdown("""
    Try asking:
    - What is the PG fee structure?
    - Eligibility for MSc Computer Science?
    - CUET subject codes for PG?
    - What are the hostel charges?
    - How to apply to CURAJ?
    - Tell me about MBA programme
    - What UG courses are available?
    - BSc Computer Science eligibility?
    """)
    
    st.divider()
    
    st.markdown("### ℹ️ About")
    st.markdown("""
    This chatbot uses **TF-IDF + Logistic Regression** 
    to classify your questions and provide relevant 
    answers about CURAJ.
    
    **Data Sources:**
    - Official CURAJ PDFs
    - University website info
    
    **Tech Stack:**
    - Scikit-learn (ML)
    - Streamlit (UI)
    - NLTK (NLP)
    """)
    
    st.divider()
    
    # clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Chat cleared! How can I help you?"}
        ]
        st.rerun()
