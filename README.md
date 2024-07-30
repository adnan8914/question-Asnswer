# question-Asnswer
# Project Overview
Welcome to the Question-Answering Chatbot Project! This repository contains the code and documentation for building advanced question-answering chatbots using state-of-the-art natural language processing models, including BERT, T5, and GPT. Our approach leverages Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), LangChain prompting, and document embedding using vector databases to deliver highly accurate and contextually relevant responses.

# Table of Contents
1. Project Overview
2. Features
3. Technologies Used
4. Models Implemented
5. Installation
6. Usage
7. Contributing
8. License
9. Acknowledgements

# Features
1. Implementation of BERT, T5, and GPT models for question-answering tasks
2. Integration of Retrieval-Augmented Generation (RAG) for enhanced accuracy
3. Utilization of LangChain for structured and controlled prompting
4. Document embedding using vector databases for efficient retrieval and clustering
5. Scalable and customizable framework for various NLP applications

# Technologies Used
1. Transformers: Hugging Face's library for accessing pre-trained transformer models like BERT, T5, and GPT.
2. PyTorch: An open-source machine learning library used for building and training deep learning models.
3. Hugging Face Datasets: A library for easy-to-use and efficient handling of datasets
# Tried To Apply
CUDA AND CUDNN for gpu accelerated computation. but my pc does not support.

# Tools
1.Jupyter Notebook: An open-source web application used for creating and sharing documents with live code, equations, visualizations, and narrative text.
2.Anaconda: A distribution of Python and R for scientific computing and data science, used for managing dependencies and environments.

# Models Implemented
# BERT
Model: Bidirectional Encoder Representations from Transformers (BERT)
Use Case: Primarily used for understanding and classification tasks
Advantages: Strong contextual understanding, widely used and supported
Limitations: Limited generation capabilities, requires task-specific fine-tuning

# T5
Model: Text-to-Text Transfer Transformer (T5)
Use Case: Versatile for both understanding and generation tasks
Advantages: Strong performance across various NLP benchmarks
Limitations: Resource-intensive, longer training times compared to smaller models

# GPT
Model: Generative Pre-trained Transformer (GPT-2, GPT-3)
Use Case: Exceptional generation capabilities, versatile across numerous tasks
Advantages: State-of-the-art performance in many benchmarks
Limitations: Extremely resource-intensive, high computational and deployment costs

# Usage
Prepare the Dataset:
Place your dataset in the data/ directory. Ensure it contains the required columns for question-answering tasks.
# Dataset
dataset is from Huggingface  Link to database : https://huggingface.co/datasets/toughdata/quora-question-answer-dataset
or simply import it in your local env

import pandas as pd

df = pd.read_json("hf://datasets/toughdata/quora-question-answer-dataset/Quora-QuAD.jsonl", lines=True)


Train the Models:
Run the training scripts for BERT, T5, and GPT models provided in the notebooks/ directory. Each notebook is designed to run independently.

Evaluate the Models:
Evaluate the performance of each model using the evaluation scripts provided.

Deploy the Chatbot:
Integrate the trained models into your application for real-time question-answering.
