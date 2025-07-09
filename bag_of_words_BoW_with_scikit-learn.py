from sklearn.feature_extraction.text import CountVectorizer # type: ignore
# Example text data
text_data = ["This is an example sentence.",
"Another sentence for illustration purposes.",
"Yet another example to demonstrate word embeddings."]
# Create CountVectorizer instance
vectorizer = CountVectorizer()
# Fit and transform the text data to BoW representation
bow_representation = vectorizer.fit_transform(text_data)
print(bow_representation.toarray())