import tensorflow as tf
import numpy as np
import pandas as pd

# Small text dataset
sentences = [
  'This movie was so poorly written and directed I fell asleep 30 minutes through the movie.',
  'This movie was really bad.',
  'My main comment on this movie is how Zwick was able to get credible actors to work on this movie?',
  'This movie is bufoonery!',
  'This movie fails miserably on every level.'
  'This movie starts out a little slow but kicks into comedic gear quickly.'
]

sentences_size = len(sentences)

# TextVectorization layer
vectorize_layer = tf.keras.layers.TextVectorization(max_tokens=sentences_size,
                                                    output_mode='int',
                                                    output_sequence_length=8)
vectorize_layer.adapt(sentences)

# TensorFlow Text custom embedding
embedding_layer = tf.keras.layers.Embedding(1000, 5)

# simple text classification model with custom embeddings
model = tf.keras.Sequential([
    vectorize_layer,
    embedding_layer,
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Labeled dataset for training
df = pd.read_csv("Test.csv")
x_train, y_train = df['text'], df['label']

# Model for training
x_train = np.array(x_train)
model.fit(x_train, y_train, epochs=10)

# Get the embedding vector for the word "pizza"
pizza_embedding = embedding_layer(np.array([vectorize_layer('pizza')])).numpy()[0]




