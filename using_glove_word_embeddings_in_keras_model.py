import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential #type:ignore
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense #type:ignore
 
# Load GloVe embeddings
embeddings_index = {}
glove_path = 'glove.6B.50d.txt'  # Path to GloVe file
with open(glove_path, 'r', encoding='utf-8') as f:
  for line in f:
    values = line.split()
    word = values[0]
    coefs = np.array(values[1:], dtype='float32')
    embeddings_index[word] = coefs
 
# Sample vocabulary
vocab = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog']
vocab_size = len(vocab)
embedding_dim = 50  # GloVe dimension
 
# Create embedding matrix
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for i, word in enumerate(vocab):
  embedding_vector = embeddings_index.get(word)
  if embedding_vector is not None:
    embedding_matrix[i] = embedding_vector
 
print("Embedding Matrix:")
print(embedding_matrix)
 
# Define the model
model = Sequential([
    Embedding(vocab_size,
              embedding_dim,
              embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
              trainable=False),  # Freeze embedding layer
    GlobalAveragePooling1D(),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
 
# Now the model is ready to be trained on your text data and labels