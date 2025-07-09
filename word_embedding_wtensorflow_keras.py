import tensorflow as tf
from tensorflow.keras.layers import Embedding # type: ignore
 
# Example text data
text_data = ["This is an example sentence.",
             "Another sentence for illustration purposes.",
             "Yet another example to demonstrate word embeddings."]
 
# Tokenize the text data
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(text_data)
sequences = tokenizer.texts_to_sequences(text_data)
 
# Pad sequences to ensure equal length input
max_length = max(len(seq) for seq in sequences)
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length)
 
# Define and train word embedding layer
vocab_size = len(tokenizer.word_index) + 1  # Adding 1 for padding token
embedding_dim = 100  # Example embedding dimension
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim)
embedded_sequences = embedding_layer(padded_sequences)
 
print(embedded_sequences)