import streamlit as st
from openai import OpenAI
import os
import json
from rag_functions import generate_answer



# get api key from file
with open("../../../apikeys/openai-keys.json", "r") as key_file:
    api_key = json.load(key_file)["default_api_key"]
os.environ["OPENAI_API_KEY"] = api_key
token = api_key


# get model names from config
with open("./config/config.json", "r") as config_file:
    config = json.load(config_file)

st.title("ChartGPT - Talk, discover Chat with Charts")
client = OpenAI()
vector_store_path="./vector-stores/chroma_db"
llm_model = config["foundational-model"]


# Chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Source documents
if "source" not in st.session_state:
    st.session_state.source = []

# Display chats
for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Ask a question
if question := st.chat_input("Ask a question"):
    # Append user question to history
    st.session_state.history.append({"role": "user", "content": question})
    # Add user question
    with st.chat_message("user"):
        st.markdown(question)

    # Answer the question
    answer, doc_source = generate_answer(question, token, vector_store_path, llm_model)
    with st.chat_message("assistant"):
        st.write(answer)
    # Append assistant answer to history
    st.session_state.history.append({"role": "assistant", "content": answer})

    # Append the document sources
    st.session_state.source.append({"question": question, "answer": answer, "document": doc_source})


# Source documents
with st.expander("Source documents"):
    st.write(st.session_state.source)