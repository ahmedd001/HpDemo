import streamlit as st
from lida import Manager, TextGenerationConfig, llm
from dotenv import load_dotenv
import os
import openai
from PIL import Image
from io import BytesIO
import base64
import re
import pandas as pd
import logging
import math

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)

# Load environment variables and OpenAI key
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# Check if the OpenAI API key is loaded
if not openai.api_key:
    st.error("OpenAI API key not found. Please ensure it is set in the .env file.")
    st.stop()  # Stop the app if the API key is missing

# Initialize LIDA Manager and config without using cache to avoid old results
def init_lida():
    return Manager(text_gen=llm("openai"))

textgen_config = TextGenerationConfig(n=1, temperature=0.5, model="gpt-4o", use_cache=False)

# Set up page config for chatbot layout
st.set_page_config(page_title="HPVizionary", layout="wide")

# Custom CSS to style the entire file uploader, send button, and the bot icon
st.markdown(
    """
    <style>
    .stFileUploader {
        background-color:#80AEDF!important;
        color: white !important;
        border-radius: 10px;
        padding: 20px;
    }

    .stFileUploader label {
        background-color: #80AEDF!important;
        color: white !important;
        border: none !important;
    }

    div.stTextInput > div > div > button {
        background-color: #80AEDF !important;
        color: white !important;
    }

    .center-content {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True
)

# Function to convert base64 string to image
def base64_to_image(base64_string):
    byte_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(byte_data))

# Improved Function to extract relevant columns from the query
def extract_columns_from_query(query, columns):
    """
    Extracts relevant columns based on the user query by matching keywords
    to the column names and detects if it's a comparison/statistics-based query.
    """
    # Extract individual keywords from the query
    keywords = re.findall(r'\b\w+\b', query.lower())
    
    # Match columns based on keywords
    matched_columns = [col for col in columns if any(keyword in col.lower() for keyword in keywords)]

    # Handle complex queries involving comparisons/statistics
    if any(word in query.lower() for word in ["compare", "difference", "vs", "between", "correlate"]):
        if len(matched_columns) < 2:  # Ensure at least 2 columns for comparison
            st.warning("Comparison requested but fewer than two relevant columns detected. Selecting the first two columns by default.")
            matched_columns = columns[:2]
    
    # Handle statistics-related queries
    if any(word in query.lower() for word in ["mean", "average", "sum", "median", "min", "max", "statistic"]):
        if len(matched_columns) < 1:  # Default to the first column for stats
            st.warning("Statistics requested but no relevant columns detected. Selecting the first column by default.")
            matched_columns = columns[:1]

    # Default to the first column if nothing is found
    if not matched_columns:
        st.warning("No relevant columns found. Defaulting to the first column.")
        matched_columns = columns[:1]
    
    return matched_columns

# Function to batch data into manageable chunks
def batch_data(df, batch_size=50):
    num_batches = math.ceil(len(df) / batch_size)
    for i in range(num_batches):
        yield df.iloc[i*batch_size: (i+1)*batch_size]

# Function to generate data from relevant columns (for multiple columns)
def generate_column_data(df, relevant_columns, max_rows=50):
    """
    Returns a structured and summarized version of the relevant columns from the DataFrame.
    Limits the number of rows to fit within the token limits while preserving relationships.
    """
    if not relevant_columns:
        return ""

    # Create structured table-like string format for the relevant data
    data_summary = df[relevant_columns].head(max_rows).to_csv(index=False)
    return f"""### Relevant Data (showing up to {max_rows} rows):\n\n{data_summary}\n\n"""

# Track conversation history using session state
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar for file upload and API key entry
with st.sidebar:
    st.title('💬 Dashboard')

    # File uploader for CSV files with blue box and white text
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

    if uploaded_file:
        # Read the uploaded CSV file into a DataFrame
        df = pd.read_csv(uploaded_file)
        st.write("Preview of the dataset:")
        st.dataframe(df)

# Display the title, logo, and "Powered by" text centered on the screen
st.markdown('<div class="center-content">', unsafe_allow_html=True)
st.image("hplogo.png", width=150)  # Adjust the logo width if necessary
st.markdown("<h1 style='font-size: 48px; color: #0096D6;'>HPVizionary</h1>", unsafe_allow_html=True)
st.markdown("<h3>Powered by Avrox</h3>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Input for user query
prompt = st.chat_input("Ask a question or request a visualization")

# Function to reset LIDA cache and reinitialize
def reset_lida_cache():
    """Function to reset LIDA cache and reinitialize it for fresh graph generation."""
    lida = init_lida()  # Reinitialize LIDA Manager to avoid using cached models
    return lida

if prompt and uploaded_file:
    # Add user's input to session history
    st.session_state.history.append({"role": "user", "content": prompt})

    # Check for relevant columns in the dataset based on the query
    relevant_columns = extract_columns_from_query(prompt, df.columns)

    # If no columns are matched or query is ambiguous, show a message
    if len(relevant_columns) == 0:  # Ensure empty column check is accurate
        st.error("No relevant columns found in the dataset for the given query.")
    
    # Handle Graphical Response if the query asks for a graph
    elif any(word in prompt.lower() for word in ["graph", "plot", "visualize"]):
        if relevant_columns:
            column_data = generate_column_data(df, relevant_columns)
            csv_context = f"The user has uploaded a dataset. Here is the data from the relevant columns:\n{column_data}"
        else:
            column_data = df.to_string(index=False)
            csv_context = f"The user has uploaded a dataset. Here is the entire dataset:\n{column_data}"

        with st.chat_message("assistant"):
            st.markdown("Generating a graph based on your query...")

        try:
            # Reinitialize LIDA and regenerate graphs for each query, ensuring no caching is done
            lida = reset_lida_cache()

            # Pass the relevant columns from the DataFrame instead of the entire dataset
            summary = lida.summarize(df[relevant_columns], summary_method="default", textgen_config=textgen_config)
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

        if relevant_columns:
            for batch_df in batch_data(df[relevant_columns], batch_size=batch_size):
                column_data = generate_column_data(batch_df, relevant_columns, max_rows=batch_size)
                csv_context = f"The user has uploaded a dataset. Here is the data from the relevant columns:\n{column_data}"

                prompt_template = f"""
                You are an expert data analyst. The user has uploaded a CSV file containing the following data:
                {csv_context}

                Based on the user query, provide analysis in natural language, interpreting the relevant columns and their data points. Provide exact answers to the user.

                ### User Query:
                {prompt}
                """

                logging.debug(f"Sending request to OpenAI API with prompt: {prompt_template}")

                try:
                    # Use the correct method for creating a chat completion
                    response = openai.chat.completions.create(
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
