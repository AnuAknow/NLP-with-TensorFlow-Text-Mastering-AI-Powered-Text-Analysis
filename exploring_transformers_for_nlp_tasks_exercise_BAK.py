import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import bibtexparser

(train_data, val_data), info = tfds.load('civil_comments', #version 0.1.0
                                         split=('train', 'test'),
                                         with_info=True, 
                                         as_supervised=True)

print(info)


# Parse the BibTeX entry
bib_database = bibtexparser.loads(info.citation)

# Access the parsed entry
entry = bib_database.entries[0]

# Print parsed entry fields
print("Title:", entry.get("title"))
print("Authors:", entry.get("author"))
print("Journal:", entry.get("journal"))
print("Year:", entry.get("year"))
print("URL:", entry.get("url"))

# Example of formatting the authors
authors = entry.get("author").split(" and ")
formatted_authors = ", ".join(authors)
print("Formatted Authors:", formatted_authors)



# Displaying the classes

# class_names = info.features['text'].name
# num_classes = info.features['threat'].num_classes

# print(f'The news are grouped into {num_classes} classes that are :{class_names}')

# num_train = info.splits['train'].num_examples
# num_val = info.splits['test'].num_examples

# print(f'The number of training samples: {num_train} \nThe number of validation samples: {num_val}')

# news_df = tfds.as_dataframe(train_data.take(10), info)

# news_df.head(10)

# for i in range (0,4):

#   print(f"Sample news {i}\n \
#   Label: {news_df['label'][i]} {(class_names[i])}\n \
#   Description: {news_df['description'][i]}\n----------\n")

#   news_df.columns

# buffer_size = 1000
# batch_size = 32

# train_data = train_data.shuffle(buffer_size)
# train_data = train_data.batch(batch_size).prefetch(1)
# val_data = val_data.batch(batch_size).prefetch(1)

# for news, label in train_data.take(1):

#   print(f'Sample news\n----\n {news.numpy()[:4]} \n----\nCorresponding labels: {label.numpy()[:4]}')

