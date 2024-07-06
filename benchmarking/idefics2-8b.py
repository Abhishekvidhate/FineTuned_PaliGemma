from text_generation import Client

API_TOKEN="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"
API_URL = "https://api-inference.huggingface.co/models/HuggingFaceM4/idefics2-8b-chatty"

# System prompt used in the playground for `idefics2-8b-chatty`
SYSTEM_PROMPT = "System: The following is a conversation between Idefics2, a highly knowledgeable and intelligent visual AI assistant created by Hugging Face, referred to as Assistant, and a human user called User. In the following interactions, User and Assistant will converse in natural language, and Assistant will do its best to answer User’s questions. Assistant has the ability to perceive images and reason about them, but it cannot generate images. Assistant was built to be respectful, polite and inclusive. It knows a lot, and always tells the truth. When prompted with an image, it does not make up facts.<end_of_utterance>\nAssistant: Hello, I'm Idefics2, Huggingface's latest multimodal assistant. How can I help you?<end_of_utterance>\n"
QUERY = "User:![](https://datasets-server.huggingface.co/assets/abhishekvidhate/PhysicsKinematisQA/--/06d7adbbf749be012e0e188ef7a44f7392bc6d18/--/default/train/0/image/image.jpg?Expires=1720278113&Signature=w~9d7OYwzMaYFVykeYrU48ivNI8vTygxPKJN5Lv~NzU501PXZx7twUWBfcfyAaSNrKfk2NgJ3X6JS1bfC1OfgNrNLXPYeqs53zkVbj5fL76oDFWwxtrbO1T54zFsWBnHiAme2ns6ZI45Xpr07Gns6qRGNRI4~DaXzbReNuG7axMoX9R3-4fu0olH9aEg3dTKT7hg3Muuj~VjlKVoO1kK0tUtE~RXINmxTu5tEJZ4izD5H-CMDuVop5A1t6HyI9~z5PXuhQ95h0-PRpSRlnwsofaRSgaR7~7bYFb6WOG4fGs3XkQ9sUCGgApPTJYxQ8ER~-f66Y0LqyYRaQKOIbqM2Q__&Key-Pair-Id=K3EI6M078Z3AC3)Describe this image.<end_of_utterance>\nAssistant:"

client = Client(
    base_url=API_URL,
    headers={"x-use-cache": "0", "Authorization": f"Bearer {API_TOKEN}"},
)
generation_args = {
    "max_new_tokens": 512,
    "repetition_penalty": 1.1,
    "do_sample": False,
}
generated_text = client.generate(prompt=SYSTEM_PROMPT + QUERY, **generation_args)
generated_text

print(generated_text)