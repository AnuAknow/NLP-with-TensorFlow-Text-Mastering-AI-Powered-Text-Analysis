# 🧠 NLP with TensorFlow Text: Mastering AI-Powered Text Analysis

This repository is a comprehensive guide for mastering **Natural Language Processing (NLP)** using TensorFlow and TensorFlow Text. It features hands-on exercises, model-building walkthroughs, and vectorization techniques that equip developers and researchers to build high-performance NLP systems.

## 🚀 What's Inside

Explore a collection of scripts and examples spanning foundational to advanced topics:

### 📚 Text Preprocessing
- `text_tokenization_with_tensorflow.py` – Basic tokenization flow
- `tokenization_and_one-hot_encoding.py` – Intro to one-hot encodings
- `text_standardization_with_non-alphanumeric_characters_removed.py` – Cleaning & normalization
- `text_stemming_and_lemmatization.py` – Lemmatization with NLTK

### 🧩 Feature Engineering
- `bag_of_words_BoW_and_onehot.py` – Manual BOW implementation
- `bag_of_words_BoW_with_scikit-learn.py` – Sklearn-powered BOW
- `creating_embedding_matrix_from_glove_for_textvectorization_layer.py` – GloVe embedding matrix construction
- `tensorflow_custom_embeddings.py` – Custom embeddings workflow

### 🔍 Embeddings & Vectors
- `embeddings_gloVe_into_tensorFlow_text.py`
- `using_glove_word_embeddings_in_keras_model.py`
- `loading_glove_word_embeddings.py`
- `word_embedding_wtensorflow_keras.py`
- `universal_sentence_encoder_use_example.py`

### 🧠 Neural Network Models
- `simple_recurrent_neural_networks_rnn_model_example.py`
- `long_term_short_term_lstm_model_example.py`
- `gru_model_example.py`
- `convolutional_neural_network_cnn_model_example.py`
- `classifying-text-with-cnn_exercise.py`
- `implementing_rnn_for_text_generation_excercise.py`

### 🤖 Transformers Exploration
- `exploring_transformers_for_nlp_tasks_exercise.py`
- `exploring_transformers_for_nlp_tasks_exercise_BAK.py`

## 🛠 Setup

Before running any code, set up your Python environment:

```bash
python3 -m venv nlp_env
source nlp_env/bin/activate  # or nlp_env\Scripts\activate on Windows
pip install -r requirements.txt

🌍 Additional Utility
- ted_hrlr_translate_pt_en_converter – Translation model using TED HRLR dataset (Portuguese → English)
- .gitignore – Ensures a clean working environment
💡 Purpose
This repo is crafted for developers, testers, and AI explorers seeking:
- Hands-on familiarity with NLP architectures and vectorization
- An understanding of embedding ecosystems
- Real-world experimentation with TensorFlow Text workflows
📄 License
Feel free to use, modify, and expand this repository. Attribution appreciated where appropriate.
