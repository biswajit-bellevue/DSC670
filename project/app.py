# import streamlit as st
# import pandas as pd
#
# # with st.chat_message("user"):
# #     st.write("Hello 👋")
#
# df = pd.read_csv("fine-tuning-results.csv")
# message = st.chat_message("assistant")
# message.write("Loss curve")
# message.line_chart(data=df, x="step",y="train_loss")

import streamlit as st
from openai import OpenAI
import os
import random
import time
import json



# get api key from file
with open("../../../apikeys/openai-keys.json", "r") as key_file:
    api_key = json.load(key_file)["default_api_key"]
os.environ["OPENAI_API_KEY"] = api_key

# get model names from config
with open("./config/config.json", "r") as config_file:
    config = json.load(config_file)

st.title("SerenityBot - Your Mental Heath Virtual Assistant")
client = OpenAI()

# Streamed response emulator
def response_generator():
    response = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = config["fine-tuned-model"]

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("How can I help you to feel better?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # response = st.write_stream(response_generator())
        for m in st.session_state.messages:
            messages = [
                {"role": "system", "content": "You are an assistant that helps answer mental health question."},
                {"role": m["role"], "content": m["content"]}

            ]
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=messages,
            stream=True,
        )
        response = st.write_stream(stream)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
