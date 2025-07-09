## A convolutional neural network (CNN or ConvNet) is a network architecture for deep learning that learns directly from data. CNNs are particularly useful for finding patterns in images to recognize objects, classes, and categories. They can also be quite effective for classifying audio, time-series, and signal data.

import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense # type: ignore
 
# Define model parameters
vocab_size = 10000  # Vocabulary size
embedding_dim = 128  # Dimension of the embedding vector
num_filters = 128  # Number of filters in the Conv1D layer
kernel_size = 5  # Size of the convolutional kernel
 
model = Sequential([
    Embedding(vocab_size, embedding_dim),
    Conv1D(num_filters, kernel_size, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(10, activation='relu'),
    Dense(1, activation='sigmoid')
])
 
# Print the shape after the Conv1D layer
print("Shape after Conv1D:", model.layers[1].output_shape)
 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])