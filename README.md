# 🍽️ Recipe Instruction Generator using LSTM Seq2Seq

This project implements a **Sequence-to-Sequence (Seq2Seq)** model using **LSTM** to automatically generate step-by-step cooking instructions from a list of ingredients. It combines Natural Language Processing and Deep Learning to transform a simple ingredient list into human-like recipe directions.

---

## 📌 Overview

Given a list of ingredients, the model generates cooking instructions in natural language. This system can be a foundational module for smart kitchen assistants, recipe apps, or food-related AI applications.

---

## 📊 Dataset

- **Size**: ~20,000 recipes  
- **Source**: Public recipe dataset (e.g., RecipeNLG, scraped collections, etc.)
- **Structure**: Each entry contains:
  - `ingredients`: A list of input tokens
  - `instructions`: Corresponding output instructions

---

## 🧠 Model Architecture

- **Type**: Encoder-Decoder Seq2Seq
- **Layers**:
  - LSTM Encoder
  - LSTM Decoder with teacher forcing during training
- **Vocabulary Size**:
  - Ingredients tokenizer: *`vocab_input_size`*
  - Instructions tokenizer: *`vocab_target_size`*
- **Parameters**: ~2.3 million trainable parameters
- **Embedding Dimension**: 256
- **Latent Dim (LSTM units)**: 512
- **Optimizer**: Adam
- **Loss**: Sparse Categorical Crossentropy

---

## ⚙️ Technologies Used

- Python 3.11  
- TensorFlow / Keras  
- NumPy  
- Jupyter Notebook  
- ipywidgets (for Jupyter-based UI)

---

## 🧪 Training Details

- **Platform**: Trained on Kaggle GPU (Tesla T4)
- **Epochs**: 50  
- **Batch Size**: 64  
- **Validation Split**: 10%

---

## 🎯 Features

- Accepts raw ingredient list input
- Tokenizes and pads input for the encoder
- Uses trained encoder & decoder models to predict output sequence
- Greedy decoding loop for inference
- Interactive Jupyter Notebook UI using `ipywidgets`

---

## 🖥️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/recipe-instruction-generator.git
   cd recipe-instruction-generator
