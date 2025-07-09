
# Classifies movie reviews as positive or negative using the text of the review. 
# This is an example of binary—or two-class—classification, an important and widely applicable kind of machine learning problem.
# The tutorial demonstrates the basic application of transfer learning with TensorFlow Hub and Keras.
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

# Download the IMDB dataset
# Split the training set into 60% and 40% to end up with 15,000 examples
# for training, 10,000 examples for validation and 25,000 examples for testing.
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews", 
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)

# Print first 10 examples
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
train_examples_batch

# Print the first 10 labels.
train_labels_batch

# Create a Keras layer that uses a TensorFlow Hub model to embed the sentences, 
# and try it out on a couple of input examples
embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(embedding, input_shape=[], 
                           dtype=tf.string, trainable=True)
hub_layer(train_examples_batch[:3])

# Build the full model
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))


# Print a useful summary of the model, which includes:
# Name and type of all layers in the model.
# Output shape for each layer.
# Number of weight parameters of each layer.
# If the model has general topology (discussed below), the inputs each layer receives
# The total number of trainable and non-trainable parameters of the model.
model.summary()


# Compile the model
# Configure the model to use an optimizer and a loss function
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])


# Train the model with Epochs
history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=10,
                    validation_data=validation_data.batch(512),
                    verbose=1)


# Evaluate the model
results = model.evaluate(test_data.batch(512), verbose=2)

for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value))
