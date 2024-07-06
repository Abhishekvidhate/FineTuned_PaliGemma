import streamlit as st
from text_generation import Client  # Assuming you have a module for handling text generation

# Constants for API configuration
API_TOKEN = "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"
API_URL = "https://api-inference.huggingface.co/models/HuggingFaceM4/idefics2-8b-chatty"

# Function to generate text based on input prompt
def generate_text(image_url, user_query):
    client = Client(
        base_url=API_URL,
        headers={"x-use-cache": "0", "Authorization": f"Bearer {API_TOKEN}"},
    )
    # Constructing the prompt for text generation
    SYSTEM_PROMPT = "System: The following is a conversation between Idefics2, a highly knowledgeable and intelligent visual AI assistant created by Hugging Face, referred to as Assistant, and a human user called User. In the following interactions, User and Assistant will converse in natural language, and Assistant will do its best to answer User’s questions. Assistant has the ability to perceive images and reason about them, but it cannot generate images. Assistant was built to be respectful, polite and inclusive. It knows a lot, and always tells the truth. When prompted with an image, it does not make up facts.<end_of_utterance>\nAssistant:"
    QUERY = f"User:![]({image_url}){user_query}<end_of_utterance>\nAssistant:"

    generation_args = {
        "max_new_tokens": 512,
        "repetition_penalty": 1.1,
        "do_sample": False,
    }
    generated_text = client.generate(prompt=SYSTEM_PROMPT + QUERY, **generation_args)
    return generated_text

# Streamlit UI components
st.title("Benchmark testing on Idefics2")
st.markdown("Enter the image URL and your query/question to get a response.")

image_url = st.text_input("Enter Image URL:")
user_query = st.text_area("Enter Your Query/Question:")

if st.button("Generate Response"):
    if image_url and user_query:
        generated_text = generate_text(image_url, user_query)
        st.text_area("Generated Response:", generated_text, height=200)
    else:
        st.warning("Please enter both the image URL and your query/question.")

