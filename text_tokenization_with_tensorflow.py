import tensorflow as tf
import tensorflow_text as tf_text

# Paragraph to be Tokenize 
paragraph = tf.constant(['The Toyota TRD 86 is a high-performance sports car designed for those who enjoy the thrill of driving. Produced by Toyota Racing Development (TRD), this car boasts a 2.0-liter four-cylinder engine that delivers 205 horsepower, ensuring an exciting and smooth driving experience. It also features sport-tuned suspension, Brembo brakes, and a limited-slip differential, giving the driver exceptional control even at high speeds. With its sleek and stylish design, the TRD 86 is an attractive and head-turning sports car that provides the perfect balance of performance and style.'])
 
# Initialize the whitespace tokenizer
tokenizer = tf_text.WhitespaceTokenizer()

# Initialize the Character tokenizer
characterTokenizer = tf_text.UnicodeCharTokenizer()
 
# Create Word Tokens
words = tokenizer.tokenize(paragraph)

# Create Word Tokens
subwords = tokenizer.tokenize(words)

# Create Character tokens
characters = characterTokenizer.tokenize(paragraph)

# Print Word tokens as a list
print(f"Word tokens as a list {words.to_list()}")

# Print Subword tokens as a list
print(f"Subword tokens as a list {subwords.to_list()}")

# Print Character tokens as a list
print(f"Character tokens as a list {characters.to_list()}")