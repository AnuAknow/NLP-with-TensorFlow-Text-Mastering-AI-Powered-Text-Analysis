import tensorflow as tf
from tensorflow.keras.models import Sequential #type:ignore
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense #type:ignore
 
# Define model parameters
vocab_size = 10000  # Vocabulary size
embedding_dim = 64  # Dimension of the embedding vector
rnn_units = 32  # Number of units in the RNN layer
 
model = Sequential([
    Embedding(vocab_size, embedding_dim),
    SimpleRNN(rnn_units),
    Dense(1, activation='sigmoid')
])
 
# Print the shape after the SimpleRNN layer
print("Shape after SimpleRNN:", model.layers[1].output_shape)
 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])