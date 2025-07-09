##Gated recurrent units (GRUs) are a gating mechanism in recurrent neural networks, introduced in 2014 by Kyunghyun Cho et al. The GRU is like a long short-term memory (LSTM) with a gating mechanism to input or forget certain features, but lacks a context vector or output gate, resulting in fewer parameters than LSTM.

from tensorflow.keras.layers import GRU, Embedding, Dense #type: ignore
from tensorflow.keras.models import Sequential #type: ignore
 
# Define model parameters
vocab_size = 10000
embedding_dim = 64
gru_units = 32
 
model = Sequential([
    Embedding(vocab_size, embedding_dim),
    GRU(gru_units),
    Dense(vocab_size, activation='softmax')
])
 
# Print the shape after the GRU layer
print("Shape after GRU:", model.layers[1].output_shape)
 
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])