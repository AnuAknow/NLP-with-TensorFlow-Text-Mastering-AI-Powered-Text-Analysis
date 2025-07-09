import numpy as np
 
# Assuming you've downloaded GloVe and have the path to it
# https://github.com/stanfordnlp/GloVe/blob/master/README.md
glove_path = 'glove.6B.50d.txt'
embeddings_index = {}
 
with open(glove_path, 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
 
print(f"Found {len(embeddings_index)} word vectors.")