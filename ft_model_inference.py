import torch
import streamlit as st
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import requests
from io import BytesIO
import os
from huggingface_hub import login

# Accessing the environment variables
hf_token = st.secrets["HF_TOKEN"]

# Set your Hugging Face token here
# hf_token = "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"
# Login to Hugging Face
login(hf_token)

# Load the model and processor
model_id = "abhishekvidhate/Abhishek-PaliGemma-FT"
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")

# Function to load an image
def load_image(image_path_or_url):
    if os.path.exists(image_path_or_url):
        return Image.open(image_path_or_url)
    elif image_path_or_url.startswith('http'):
        response = requests.get(image_path_or_url)
        image_data = response.content
        return Image.open(BytesIO(image_data))
    else:
        raise ValueError("Unsupported image input. Please provide a valid local file path or image URL.")

# Streamlit UI
st.title("FineTuned PaliGemma Model Inference")
st.markdown("""
PaliGemma is an open vision-language model by Google, inspired by PaLI-3 and built with open components such as the SigLIP vision model and the Gemma language model. PaliGemma is designed as a versatile model for transfer to a wide range of vision-language tasks such as image and short video captioning, visual question answering, text reading, object detection, and object segmentation.

This demo uses a fine-tuned version of PaliGemma.I have Fine tuned Pali-Gemma-3b-pt-224 on small split of DocumentVQA dataset( total unzipped 60+ GB of data) You can upload an image and enter a prompt to see how the model performs on various tasks.


more about [QVA datasets](https://huggingface.co/datasets/HuggingFaceM4/DocumentVQA)
""")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Prompt input
prompt = st.text_input("Enter your prompt:")

# Display the image and run inference if both image and prompt are provided
if uploaded_file is not None and prompt:
    raw_image = Image.open(uploaded_file)
    st.image(raw_image, caption="Uploaded Image", use_column_width=True)

    # Run inference
    inputs = processor(prompt, raw_image.convert("RGB"), return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=20)
    result = processor.decode(output[0], skip_special_tokens=True)[len(prompt):]

    # Display the result
    st.subheader("Output")
    st.write(result)

# Clear button
if st.button("Clear"):
    st.experimental_rerun()
