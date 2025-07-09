from tensorflow.keras.layers import LSTM, Embedding, Dense #type:ignore
from tensorflow.keras.models import Sequential #type:ignore
 
# Define model parameters
vocab_size = 10000
embedding_dim = 64
lstm_units = 32
 
model = Sequential([
    Embedding(vocab_size, embedding_dim),
    LSTM(lstm_units),
    Dense(1, activation='sigmoid')
])
 
# Print the shape after the LSTM layer
print("Shape after LSTM:", model.layers[1].output_shape)
 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])