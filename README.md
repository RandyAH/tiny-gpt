# TinyGPT (From Scratch)

A minimal implementation of a GPT-style transformer language model built from scratch in PyTorch.

This project demonstrates a full understanding of **transformer architecture, training pipelines, and sequence generation** without relying on high-level libraries.

---

## 🚀 Overview

TinyGPT is a character-level language model trained on text data (e.g., Shakespeare) to generate realistic sequences.

It implements the core components behind modern large language models:
- Token + positional embeddings  
- Multi-head self-attention  
- Causal masking  
- Transformer blocks with residual connections  
- Autoregressive text generation  

---

## 🧠 Features

- Built entirely in PyTorch (no prebuilt transformer libraries)  
- Multi-head self-attention with scaled dot-product attention  
- Causal masking for autoregressive prediction  
- Efficient batching and training loop  
- Text generation from learned patterns  

---

## 🏗️ Model Architecture

The model follows a standard GPT-style architecture:

- Token Embedding Layer  
- Positional Embedding Layer  
- Stacked Transformer Blocks:
  - Layer Normalization  
  - Multi-Head Self-Attention  
  - Feedforward Network  
- Final Linear Layer → Vocabulary logits  

---

## 📦 Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/RandyAH/tiny-gpt
cd tiny-gpt
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Train the Model

```bash
python train.py
```

---

## ✨ Generate Text

```bash
python generate.py
```

Example output:

```text
ROMEO:
What light through yonder window breaks?
...
```

---

## 📊 What This Project Demonstrates

- Deep understanding of transformer internals  
- Implementation of attention mechanisms from scratch  
- End-to-end model training and sequence generation  
- Debugging and optimizing neural network workflows  

---

## 📫 Author

Randy Hannah  
- Portfolio: https://randyhannah.com  
- GitHub: https://github.com/RandyAH  
- LinkedIn: https://www.linkedin.com/in/randenn-hannah/  
