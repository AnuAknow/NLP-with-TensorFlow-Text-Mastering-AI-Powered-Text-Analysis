import nltk
from nltk.stem import PorterStemmer
import spacy

# Stemming
# Download NLTK resources 
nltk.download('punkt')
 
# Initialize the stemmer
stemmer = PorterStemmer()

# List of words to demonstrate Stemming 
words = ["Hurries", "Worries", "Annoys", "Lays", "Says", "Delays", "Dearing", "Seeing", "Outing", "Ending", "Acting", "friendly", "lovely", "worldly", "sickly", "costly"]
 
# Stemming process
stemmed_words = [stemmer.stem(word) for word in words]
 

# Lemmatization
# Loading the spaCy model
nlp = spacy.load("en_core_web_sm")

# Sentence of words to demonstrate Lemmatizing
sentence = "He had just received several boxes of delicious chocolates, and he couldn't wait to indulge in them all."
 
# Process the sentence of words to demonstrate Lemmatizing
doc = nlp(sentence)
 
# Lemmatization process
lemmatized_words = [token.lemma_ for token in doc]

# Printing Stemming Output
print("\n::::Stemming::::")
print("Original words to demonstrate Stemming:\n", words)
print("\nStemmed words:\n\n", stemmed_words)

# Printing Lemmatizing Output
print("\::::Lemmatizing::::")
print("\nOriginal sentence to demonstrate Lemmatizing:\n", sentence)
print("\nLemmatized words:\n", lemmatized_words)
