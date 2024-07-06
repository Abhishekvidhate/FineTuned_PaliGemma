import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import os
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

# Hugging Face API details
api_url = "https://api-inference.huggingface.co/models/abhishekvidhate/Abhishek-PaliGemma-FT"
api_token = "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"  # Replace with your actual read token

# Headers for API request
headers = {
    "Authorization": f"Bearer {api_token}"
}

# Load the processor
# model_id = "abhishekvidhate/Abhishek-PaliGemma-FT"
# model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")

# # Function to load an image
# def load_image(image_path_or_url):
#     if os.path.exists(image_path_or_url):
#         return Image.open(image_path_or_url)
#     elif image_path_or_url.startswith('http'):
#         response = requests.get(image_path_or_url)
#         image_data = response.content
#         return Image.open(BytesIO(image_data))
#     else:
#         raise ValueError("Unsupported image input. Please provide a valid local file path or image URL.")

# # Streamlit UI
# st.title("PaliGemma Model Inference")
# st.write("Upload an image and enter a prompt for inference.")

# # Image upload
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# # Prompt input
# prompt = st.text_input("Enter your prompt:")

# # Display the image and run inference if both image and prompt are provided
# if uploaded_file is not None and prompt:
#     raw_image = Image.open(uploaded_file)
#     st.image(raw_image, caption="Uploaded Image", use_column_width=True)

#     # Convert image to RGB and to bytes
#     image_bytes = BytesIO()
#     raw_image.convert("RGB").save(image_bytes, format="JPEG")
#     image_bytes.seek(0)

#     # Prepare the files for multipart/form-data request
#     files = {
#         "file": ("image.jpg", image_bytes, "image/jpeg"),
#     }
#     data = {
#         "inputs": prompt
#     }

#     # Send request to the API
#     response = requests.post(api_url, headers=headers, files=files, data=data)

#     if response.status_code == 200:
#         result = response.json()
#         output_text = result.get('generated_text', '')[len(prompt):]  # Adjust if needed

#         # Display the result
#         st.subheader("Output")
#         st.write(output_text)
#     else:
#         st.error(f"Error: {response.status_code} - {response.text}")

# # Clear button
# if st.button("Clear"):
#     st.experimental_rerun()

# Streamlit UI
st.title("PaliGemma Model Inference")
st.write("Upload an image and enter a prompt for inference.")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Prompt input
prompt = st.text_input("Enter your prompt:")

# Display the image and run inference if both image and prompt are provided
if uploaded_file is not None and prompt:
    raw_image = Image.open(uploaded_file)
    st.image(raw_image, caption="Uploaded Image", use_column_width=True)

    # Convert image to bytes
    image_bytes = BytesIO()
    raw_image.save(image_bytes, format="PNG")
    image_bytes.seek(0)

    # Prepare files for multipart/form-data request
    files = {
        "file": ("image.png", image_bytes, "image/png"),
    }
    data = {
        "inputs": prompt
    }

    # Send request to the API
    response = requests.post(api_url, headers=headers, files=files, data=data)

    if response.status_code == 200:
        result = response.json()
        output_text = result.get('generated_text', '')

        # Display the result
        st.subheader("Output")
        st.write(output_text)
    else:
        st.error(f"Error: {response.status_code} - {response.text}")

# Clear button
if st.button("Clear"):
    st.experimental_rerun()
