import streamlit as st
from lida import Manager, TextGenerationConfig, llm
from dotenv import load_dotenv
import os
from openai import OpenAI
from PIL import Image
from io import BytesIO
import base64
import pandas as pd
import logging
import math
from pathlib import Path

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)

# Load environment variables and OpenAI key
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# Check if the OpenAI API key is loaded
if not openai_api_key:
    st.error("OpenAI API key not found. Please ensure it is set in the .env file.")
    st.stop()  # Stop the app if the API key is missing

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

# Initialize LIDA Manager and config without using cache to avoid old results
def init_lida():
    return Manager(text_gen=llm("openai"))

textgen_config = TextGenerationConfig(n=1, temperature=0.5, model="gpt-4o", use_cache=False)

# Set up page config for chatbot layout
st.set_page_config(page_title="Vizionary", layout="wide")

# Custom CSS to enhance the chatbot UI
st.markdown(
    """
    <style>
    /* General Styles */
    html, body {
        font-family: 'Rubik', sans-serif;
        background-color: #f9f9f9;
    }
    .stButton > button {
        background: linear-gradient(145deg, #6a11cb 0%, #2575fc 100%);
        border: none;
        color: white;
        padding: 10px 20px;
        font-size: 14px;
        border-radius: 10px;
        margin-top: 10px;
        transition: background 0.3s ease-in-out;
    }
    .stButton > button:hover {
        background: linear-gradient(145deg, #5b10a6 0%, #1e62b2 100%);
    }
    .stTextInput > div > div > input {
        border-radius: 10px;
        padding: 10px;
    }
    /* Chat Bubbles */
    .user-bubble {
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        display: inline-block;
        max-width: 70%;
    }
    .assistant-bubble {
        background-color: #e6e6e6;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        display: inline-block;
        max-width: 70%;
        color: black;
    }
    .chat-container {
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True
)

# Function to convert base64 string to image
def base64_to_image(base64_string):
    byte_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(byte_data))

# Function to batch data into manageable chunks
def batch_data(df, batch_size=50):
    num_batches = math.ceil(len(df) / batch_size)
    for i in range(num_batches):
        yield df.iloc[i * batch_size: (i + 1) * batch_size]

# Function to generate data for OpenAI API
def generate_data_for_model(df, max_rows=50):
    """
    Prepares a summarized version of the dataset for the model, respecting token limits.
    """
    return df.head(max_rows).to_csv(index=False)

# Track conversation history using session state
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar for file upload and API key entry
with st.sidebar:
    st.write('📊 **Dashboard**')
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("🔍 **Preview of the dataset:**")
        st.dataframe(df)

prompt = st.chat_input("Ask a question or request a visualization")

suggestions = [
    "List all the work orders completed within the Requested Date Time (RDT).",
    "List all the work orders whose response is met.",
    "List all the work orders with their delay code and time of delay.",
    "Make a pie chart for the top 5 managers based on their number of work orders.",
    "What are the delay reasons for work orders completed after RDT?",
    "How many work orders have not been completed (no response)?",
    "What was the average delay (in days) for work orders not completed on time?",
]

st.markdown("### 💡 **Suggested Queries**")
for suggestion in suggestions:
    if st.button(suggestion):
        prompt = suggestion

def reset_lida_cache():
    lida = init_lida()  # Reinitialize LIDA Manager to avoid using cached models
    return lida

if prompt and uploaded_file:
    st.session_state.history.append({"role": "user", "content": prompt})

    if any(word in prompt.lower() for word in ["graph", "plot", "visualize", "pie"]):
        with st.chat_message("assistant"):
            st.markdown("🛠️ Generating a graph based on your query...")

        try:
            lida = reset_lida_cache()
            csv_context = f"The user has uploaded a dataset. Here is the summarized data:\n{generate_data_for_model(df)}"
            summary = lida.summarize(df, summary_method="default", textgen_config=textgen_config)
            charts = lida.visualize(summary=summary, goal=prompt, textgen_config=textgen_config)

            for chart in charts:
                image_base64 = chart.raster
                img = base64_to_image(image_base64)
                st.session_state.history.append({"role": "assistant", "type": "chart", "image": img})
        except Exception as e:
            st.error(f"Error generating graph: {str(e)}")

    else:
        all_responses = []
        batch_size = 50

        for batch_df in batch_data(df, batch_size=batch_size):
            column_data = generate_data_for_model(batch_df, max_rows=batch_size)
            csv_context = f"The user has uploaded a dataset. Here is the data from the dataset:\n{column_data}"

            prompt_template = f"""
            You are an expert data analyst. The user has uploaded a CSV file containing the following data:
            {csv_context}

            Based on the user query, provide the exact figures or points required.

            ### User Query:
            {prompt}
            """

            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt_template}],
                    max_tokens=500
                )
                bot_response = response.choices[0].message.content
                all_responses.append(bot_response)

            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

        combined_response = "\n\n".join(all_responses)
        st.session_state.history.append({"role": "assistant", "content": combined_response})

# Display chat history with improved UI
for entry in st.session_state.history:
    if entry["role"] == "user":
        with st.chat_message("user"):
            st.markdown(f"<div class='user-bubble'>{entry['content']}</div>", unsafe_allow_html=True)
    elif entry.get("type") == "chart":
        with st.chat_message("assistant"):
            st.image(entry["image"])
    else:
        with st.chat_message("assistant"):
            st.markdown(f"<div class='assistant-bubble'>{entry['content']}</div>", unsafe_allow_html=True)