import tensorflow as tf

# Paragraph to be Normalized
paragraph = tf.constant(['The Toyota TRD 86 is a high-performance sports car designed for those who enjoy the thrill of driving. Produced by Toyota Racing Development (TRD), this car boasts a 2.0-liter four-cylinder engine that delivers 205 horsepower, ensuring an exciting and smooth driving experience. It also features sport-tuned suspension, Brembo brakes, and a limited-slip differential, giving the driver exceptional control even at high speeds. With its sleek and stylish design, the TRD 86 is an attractive and head-turning sports car that provides the perfect balance of performance and style.'])

# Lowercasing Text
lower_texts = [tf.strings.lower(paragraph) for text in paragraph]

# Remove all punctuation from the text
standardized_texts = [tf.strings.regex_replace(text, r'[^\w\s]', '') for text in paragraph]
 
# Original texts
print("Original texts:\n", paragraph)

# Lower case texts
print("\nLower case texts:\n", lower_texts)

# Standardized texts with Non-alphanumeric characters removed
print("\nStandardized texts with Non-alphanumeric characters removed:\n", standardized_texts)