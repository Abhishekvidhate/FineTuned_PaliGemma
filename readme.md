# Fine Tuning Multi-Modal Language Model ( VLM )

Selected **Pali Gemma** model to work on. Pali Gemma is an advanced open-source vision-language model by Google Research, designed for tasks like image captioning and visual question answering.

### Video Demo : [Google Drive](https://drive.google.com/drive/folders/1i0kcuPQYUWp_8mMMiG6sQZ1issjeJWP6?usp=sharing)

### I selected **Pali Gemma** because of it's **Architecture**, [learn more](https://huggingface.co/blog/paligemma)

**PaliGemma Model Architecture:**

PaliGemma models utilize a transformer-based architecture with attention mechanisms, enabling them to focus on different parts of input text effectively. Pre-trained on diverse datasets, these models are modular and adaptable, making them suitable for a wide range of NLP tasks.

**Ease of Fine-Tuning:**

Pre-trained checkpoints provide a solid foundation for fine-tuning. User-friendly libraries like Hugging Face's Transformers simplify the process. Efficient techniques such as Low-Rank Adaptation (LoRA) and Quantization make fine-tuning feasible even with limited data and computational resources.


![image](https://github.com/Abhishekvidhate/FineTuned_PaliGemma/assets/120262589/8bf94eda-6f61-42d6-8cb3-0626f5c8d275)

### I wanted to Fine tune a model on Different techinque other than LORA and QLORA, so i think **Pali Gemma** would be best fit for this.

# Set Up
I only used **KAGGLE** for all the processing and inference, as even **Google Collab's** free resoucres are not enough for **LORA** fine Tuning

I will recommend to upload my notebooks -
 - [**FINE TUNING**](https://github.com/Abhishekvidhate/FineTuned_PaliGemma/blob/master/finetuning.ipynb)
 - [**INFERENCE**](inference.ipynb)
 - [**BENCHMARKING**](benchmarking/benchmarking-vlm.ipynb)

to your kaggle notebook and run it to replicate my results

# Fine Tuning 

I wanted to try different method other than **LORA** Fine tuning, so first i decided to Perform Fine Tuning use **Transformer Reinforcement Learning**.

### Transformer Reinforcement Learning
The main idea is that instead of training a policy using RL methods, such as fitting a value function, that will tell us what action to take to maximize the return (cumulative reward), we use a sequence modeling algorithm (Transformer) that, given a desired return, past states, and actions, will generate future actions to achieve this desired return.

main issue with this approach was `CUDA out of Memory` , even if i tried it with `LORA & QLoRA` techniques it was not working

![Screenshot 2024-07-05 101343](https://github.com/Abhishekvidhate/FineTuned_PaliGemma/assets/120262589/661d8dda-d2f6-4aa2-af5e-771fe414f778)

so i decided to Perform Normal `LORA config` Fine tuning for **Pali Gemma**, but it was worth it , learnt a lot of new concepts

#### Follow Finetuning notebook for detailed explaination of LORA Config , which lead to fine tuning and resolved **CUDA out of memory** issue. Finally i pushed my Fine tuned Model on huggingface for future quick easy for inference , [model webpage](https://huggingface.co/abhishekvidhate/Abhishek-PaliGemma-FT)

# Dataset 
I used [DocumentVQA dataset](https://huggingface.co/datasets/HuggingFaceM4/DocumentVQA), it is a dataset of document related Question-Answer with Images.
**60+ GB of data when unzipped**, so i decided to use **small split of this dataset**, so later every time i load data i won't have to wait 30 minutes

![Screenshot 2024-07-06 020024](https://github.com/Abhishekvidhate/FineTuned_PaliGemma/assets/120262589/335ffcc1-750a-4050-b742-3c0db902c6d4)

### I have pushed my split of dataset on huggingface for future easy and quick use [dataset](https://huggingface.co/datasets/abhishekvidhate/DocQVA_small)

## Custom Built dataset
I also built custom dataset of Physics quesiton , which i decide to use for fine tuning , but these questions are hard , and later decided to use these for **BENCHMARKING VLMs**
[CUSTOM DATASET](https://huggingface.co/datasets/abhishekvidhate/PhysicsKinematisQA)

# Inference

I learnt new methods to Inference VLMs , as i was only used to deploy and inference **LLMs** and with help of **LANGCHAIN** inference and deployment is easy.

Inference **FINE TUNED MODEL** Locally on my PC is not possible, as first loading model and then using it is not possible for my laptop ( less powerful GPU). So I Inferred using Free resources of **KAGGLE NOTEBOOK**

#### follow my Inference notebook for detailed explaination

Inference with **Streamlit APP**, I was not able to do it using API and even with Streamlit cloud deployment, as even **Streamlit cloud's virtual machine has limited memory, so loading FINE TUNED MODEL, then inferencing was not possible , as CUDA out of memory error**

I tried HUGGINGFACE API INFERENCE, failed to deploy and inference , i need more time to do this ( as i lack some knowledge here)

# BENCHMARK

I was planning to use **VLMEvalKit** for benchmarking my fine tuned VLM against other OpenSource VLMs ,but to use this , my Fine tuned model must be supported by **VLMEvalKit**, which is not supported. so i decided to benchmark using my own **CUSTOM DATASET**.

I used [HuggingFaceM4/idefics2-8b-chatty](https://huggingface.co/HuggingFaceM4/idefics2-8b-chatty) to use for benchmark for my fine tuned VLM
#### Follow BENCHMARK NOTEBOOK for detailed explaination

![image](https://github.com/Abhishekvidhate/FineTuned_PaliGemma/assets/120262589/8de446a9-a623-4040-8b66-b1a47b097260)

# final thoughts on Task

this was really exciting and interesting task, i learnt new things and concepts , learnt new ways to think and understand working of MODELS better, learnt new techniques to deploy and inference VLMS LLMs models. It was good project , learnt alot
