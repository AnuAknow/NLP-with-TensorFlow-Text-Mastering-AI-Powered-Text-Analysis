import tensorflow_hub as hub
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
 
# Load a pre-trained Transformer model as a Keras layer
transformer_model = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4", dtype=tf.string, input_shape=[], output_shape=[512])
 
model = Sequential([
    transformer_model,
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])
 
# Print the shape after the USE layer
print("Shape after Universal Sentence Encoder:", model.layers[0].output_shape)
 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])