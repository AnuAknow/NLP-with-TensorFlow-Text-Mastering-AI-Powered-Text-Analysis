import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer # type: ignore

# Example text data
text_data = ["This is an example sentence.",
"Another sentence for illustration purposes.",
"Yet another example to demonstrate word embeddings."]

# Create CountVectorizer instance
vectorizer = CountVectorizer()

# Fit and transform the text data to BoW representation
bow_representation = vectorizer.fit_transform(text_data)


# Tokenize the text data
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(text_data)
sequences = tokenizer.texts_to_sequences(text_data)

# Get the maximum token index
num_tokens = max(tokenizer.word_index.values()) + 1
# Convert sequences to one-hot encoding
one_hot_encoded = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post')
one_hot_encoded = tf.one_hot(one_hot_encoded, depth=num_tokens)

print(f"\n\nbow_representation{bow_representation.toarray()}\n")
print(f"\n\none_hot_encoded: {one_hot_encoded}")