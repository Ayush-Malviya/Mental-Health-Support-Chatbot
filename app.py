
import streamlit as st
import sqlite3
import hashlib
import numpy as np
import tensorflow as tf
import json
import pickle
import random
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('punkt_tab')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load model, words, classes, and intents data
model = tf.keras.models.load_model('model.keras')
with open('texts.pkl', 'rb') as f:
    words = pickle.load(f)
with open('labels.pkl', 'rb') as f:
    classes = pickle.load(f)
with open('intents.json') as f:
    intents = json.load(f)

# Initialize and configure SQLite database
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            username TEXT UNIQUE,
            password TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Hashing function for password security
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Register a new user
def register_user(name, username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    hashed_password = hash_password(password)
    try:
        c.execute("INSERT INTO users (name, username, password) VALUES (?, ?, ?)", (name, username, hashed_password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

# Authenticate user
def authenticate_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    hashed_password = hash_password(password)
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, hashed_password))
    user = c.fetchone()
    conn.close()
    return user

# NLP and Chatbot functions
def preprocess_input(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    bag = [1 if word in sentence_words else 0 for word in words]
    return np.array([bag])

def predict_class(sentence):
    bag = preprocess_input(sentence)
    prediction = model.predict(bag)[0]
    threshold = 0.25
    results = [{"intent": classes[i], "probability": str(pred)} for i, pred in enumerate(prediction) if pred > threshold]
    results.sort(key=lambda x: x["probability"], reverse=True)
    return results

def get_response(intents_list, intents_json):
    if not intents_list:
        return "I'm sorry, I didn't quite understand that. Could you rephrase?"
    tag = intents_list[0]['intent']
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I'm sorry, I didn't understand that."

def chatbot_response(text):
    intents_list = predict_class(text)
    return get_response(intents_list, intents)

# Initialize database on app start
init_db()

# Streamlit UI layout
st.title("Mental Health Support Chatbot")

# State management for user authentication
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = None

# Define login and registration screens
def login():
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        user = authenticate_user(username, password)
        if user:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.session_state.name = user[1]
            st.success(f"Welcome back, {st.session_state.name}!")
        else:
            st.error("Invalid username or password")

def register():
    st.subheader("Register")
    name = st.text_input("Full Name")
    username = st.text_input("Choose a Username")
    password = st.text_input("Choose a Password", type="password")
    if st.button("Register"):
        if register_user(name, username, password):
            st.success("Registration successful! Please log in.")
        else:
            st.error("Username already taken. Try a different one.")

# Define profile and chatbot screens
def profile():
    st.subheader("Profile")
    st.write("Name:", st.session_state.name)
    st.write("Username:", st.session_state.username)

def chatbot():
    st.subheader("Chatbot")

    # Display conversation history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    # Capture user input
    prompt = st.chat_input("Type your message here...")
    if prompt:
        # Display user message and append to session history
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate and display bot response
        response = chatbot_response(prompt)
        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# Main application logic
if not st.session_state.authenticated:
    page = st.sidebar.selectbox("Choose an option", ["Login", "Register"])
    if page == "Login":
        login()
    elif page == "Register":
        register()
else:
    page = st.sidebar.selectbox("Choose an option", ["Chatbot", "Profile"])
    if page == "Chatbot":
        chatbot()
    elif page == "Profile":
        profile()
