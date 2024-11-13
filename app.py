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

# # Convert image to bytes
# def img_to_bytes(img_path):
#     img_bytes = Path(img_path).read_bytes()
#     encoded = base64.b64encode(img_bytes).decode()
#     return encoded

# # Convert image to HTML
# def img_to_html(img_path):
#     img_html = "<img src='data:image/png;base64,{}' width='100' class='logo'>".format(
#         img_to_bytes(img_path)
#     )
#     return img_html

# # Path to your image
# image_path = 'hplogo.png'

# # Get the base64 image HTML
# image_html = img_to_html(image_path)

# # Create the HTML code for the logo and title
# html_code = f"""
#     <div style="display: flex; flex-direction:column; align-items: center; justify-contents:center; margin-bottom:40px">
#         {image_html}
#         <h1 style="font-size: 20px; margin: 0; position:absolute; top:40px ">HP Vizionary <span>MVP</span></h1>
#     </div>
#     """

# Display the HTML code in the sidebar
#st.sidebar.markdown(html_code, unsafe_allow_html=True)

# Custom CSS to style the entire file uploader, send button, and the bot icon
st.markdown(
    """
    <style>
       @import url('https://fonts.google.com/share?selection.family=Rubik:ital,wght@0,300..900;1,300..900');
    html, body {
        font-family: 'Rubik', sans-serif;
    }
    .stFileUploader {
        background: linear-gradient(107.91deg, #3673EA 7.37%, #4623E9 95.19%) !important;
        color: white !important;
        border-radius: 10px;
        padding: 20px;
    }
    .stFileUploader label {
        background: linear-gradient(107.91deg, #3673EA 7.37%, #4623E9 95.19%) !important;
        color: white !important;
        border: none !important;
    }
    div.stTextInput > div > div > button {
        background: linear-gradient(107.91deg, #3673EA 7.37%, #4623E9 95.19%) !important;
        color: white !important;
    }
    [data-testid="stSidebarHeader"] {
        display: none;
    }
    [data-testid="stMarkdownContainer"] p {
        font-size: 16px;
        font-weight: bold;
    }
    [data-testid="stMarkdownContainer"] h1 {
        font-size: 20px;
        font-weight: 700;
        text-align:center;
        font-family: 'Rubik', sans-serif;
    }
    [data-testid="stSidebarContent"] {
        box-shadow: rgba(0, 0, 0, 0.35) 0px 5px 15px;
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
    st.write('Dashboard')

    # File uploader for CSV files
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

    if uploaded_file:
        # Read the uploaded CSV file into a DataFrame
        df = pd.read_csv(uploaded_file)
        st.write("Preview of the dataset:")
        st.dataframe(df)

# Input for user query
prompt = st.chat_input("Ask a question or request a visualization")

# Clickable suggestions
suggestions = [
    "List all the work orders completed within the Requested Date Time (RDT).",
    "List all the work orders whose response is met.",
    "List all the work orders with their delay code and time of delay.",
    "Make a pie chart for the top 5 managers based on their number of work orders.",
    "What are the delay reasons for work orders completed after RDT?",
    "How many work orders have not been completed (no response)?",
    "What was the average delay (in days) for work orders not completed on time?",
]

# Display clickable suggestions in the chat
st.markdown("### Suggested Queries:")
for suggestion in suggestions:
    if st.button(suggestion):
        prompt = suggestion

# Function to reset LIDA cache and reinitialize
def reset_lida_cache():
    """Function to reset LIDA cache and reinitialize it for fresh graph generation."""
    lida = init_lida()  # Reinitialize LIDA Manager to avoid using cached models
    return lida

if prompt and uploaded_file:
    # Add user's input to session history
    st.session_state.history.append({"role": "user", "content": prompt})

    # Handle Graphical Response if the query asks for a graph
    if any(word in prompt.lower() for word in ["graph", "plot", "visualize", "pie"]):
        with st.chat_message("assistant"):
            st.markdown("Generating a graph based on your query...")

        try:
            # Reinitialize LIDA and regenerate graphs for each query, ensuring no caching is done
            lida = reset_lida_cache()

            # Prepare the dataset summary for graphing
            csv_context = f"The user has uploaded a dataset. Here is the summarized data:\n{generate_data_for_model(df)}"
            
            # Request LIDA to generate charts
            summary = lida.summarize(df, summary_method="default", textgen_config=textgen_config)
            charts = lida.visualize(summary=summary, goal=prompt, textgen_config=textgen_config)

            # Display the generated charts dynamically for the current dataset
            for chart in charts:
                image_base64 = chart.raster
                img = base64_to_image(image_base64)
                st.session_state.history.append({"role": "assistant", "type": "chart", "image": img})
        except Exception as e:
            st.error(f"Error generating graph: {str(e)}")

    else:
        # Handle textual analysis with multiple batches if necessary
        all_responses = []
        batch_size = 50

        # Calculate the data for each batch
        for batch_df in batch_data(df, batch_size=batch_size):
            column_data = generate_data_for_model(batch_df, max_rows=batch_size)
            csv_context = f"The user has uploaded a dataset. Here is the data from the dataset:\n{column_data}"

            # Include the data in the prompt
            prompt_template = f"""
            You are an expert data analyst. The user has uploaded a CSV file containing the following data:
            {csv_context}

            Based on the user query, provide the exact figures or points required.

            ### User Query:
            {prompt}
            """

            logging.debug(f"Sending request to OpenAI API with prompt: {prompt_template}")

            try:
                # Use the correct method for creating a chat completion
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt_template}],
                    max_tokens=500
                )

                # Extracting the response content correctly
                bot_response = response.choices[0].message.content
                all_responses.append(bot_response)

            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

        # Combine all batch responses
        combined_response = "\n\n".join(all_responses)
        st.session_state.history.append({"role": "assistant", "content": combined_response})

# Display chat history in the UI
for entry in st.session_state.history:
    if entry["role"] == "user":
        with st.chat_message("user"):
            st.markdown(entry["content"])
    elif entry.get("type") == "chart":
        with st.chat_message("assistant"):
            st.image(entry["image"])
    else:
        with st.chat_message("assistant"):
            st.markdown(entry["content"])









