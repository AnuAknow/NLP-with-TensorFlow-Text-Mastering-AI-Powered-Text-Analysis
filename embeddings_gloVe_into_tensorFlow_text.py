import numpy as np
import tensorflow as tf
 
# Sample vocabulary
vocab = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog']
vocab_size = len(vocab)
embedding_dim = 50  # GloVe dimension
 
# Load GloVe embeddings
embeddings_index = {}
glove_path = 'glove.6B.50d.txt'  # Path to GloVe file
with open(glove_path, 'r', encoding='utf-8') as f:
  for line in f:
    values = line.split()
    word = values[0]
    coefs = np.array(values[1:], dtype='float32')
    embeddings_index[word] = coefs
 
# Create a TextVectorization layer to index vocabulary
vectorize_layer = tf.keras.layers.TextVectorization(max_tokens=vocab_size,
                                                    output_mode='int',
                                                    output_sequence_length=8)
vectorize_layer.adapt(vocab)
 
# Create an embedding matrix
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for i, word in enumerate(vocab):
  embedding_vector = embeddings_index.get(word)
  if embedding_vector is not None:
    embedding_matrix[i] = embedding_vector
 
# Print the embedding matrix
print("Embedding Matrix:")
print(embedding_matrix)