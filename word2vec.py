import numpy as np
import pandas as pd
from collections import Counter
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from sklearn.decomposition import PCA
from matplotlib import pyplot
from sklearn.metrics import f1_score
from gensim.models import word2vec
from sklearn.decomposition import IncrementalPCA  # inital reduction
from sklearn.manifold import TSNE  # final reduction
from gensim.test.utils import datapath


def plot_with_matplotlib(x_vals, y_vals, labels):
    import matplotlib.pyplot as plt
    import random

    random.seed(0)

    plt.figure(figsize=(12, 12))
    plt.scatter(x_vals, y_vals)

    #
    # Label randomly subsampled 25 data points
    #
    indices = list(range(len(labels)))
    selected_indices = random.sample(indices, 300)
    for i in selected_indices:
        plt.annotate(labels[i], (x_vals[i], y_vals[i]))
    plt.show()


def reduce_dimensions(model):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    # extract the words & their vectors, as numpy arrays
    vectors = np.asarray(model.wv.vectors)
    print(vectors)
    labels = np.asarray(model.wv.index2word)  # fixed-width numpy strings

    # reduce using t-SNE
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels


def get_unmatching_word(words):
    for word in words:
        if word not in model.wv.vocab:
            print("No esta en vocab:", word)
            return None
    return model.wv.doesnt_match(words)


def pad_features(reviews_ints, seq_length, num_features):
    """Retornar valores de review_ints, cada review es truncada o rellenada con 0s dada la longitud
        seq_length."""

    # obtener el shape(row x col)
    features = np.zeros(num_features, dtype=np.float)

    # para cada review, hacer el padding de cada palabra
    # for i, row in enumerate(reviews_ints):
    #    features[i, -len(row):] = np.array(row)[:seq_length]

    return features


data = pd.read_csv('Tweets.csv')
reviews = np.array(data['text'])[:14640]
labels = np.array(data['airline_sentiment'])[:14640]
print(data['text'].loc[14639], "\nSENTIMENT:", data['airline_sentiment'].loc[14639])

# PREPROCESAMIENTO
punctuation = '!"#$%&\'()*+,-./:;<=>?[\\]^_`{|}~’'

# eliminar puntuación
all_reviews = 'separator'.join(reviews)
all_reviews = all_reviews.lower()
all_text = ''.join([c for c in all_reviews if c not in punctuation])

# separar por nuevas líneas y espacios
reviews_split = all_text.split('separator')

# remover direcciones web, digitos y etiquetas(@identificador_twitter)
new_reviews = []
for review in reviews_split:
    review = review.split()
    new_text = []
    for word in review:
        if ('@' not in word) and ('http' not in word) and (not word.isdigit()):
            new_text.append(word)
    new_reviews.append(new_text)


print("Training model...")

model = word2vec.Word2Vec(min_count=5,
                          window=10,
                          size=25,
                          sample=0.1,
                          alpha=0.05,
                          min_alpha=0.0001,
                          negative=50,
                          workers=4)
model.build_vocab(new_reviews, progress_per=10000)
model.train(new_reviews, total_examples=model.corpus_count, epochs=50, report_delay=1)

model.wv.save("word2vec.model")

model_name = "twitter_airline"
model.wv.save(model_name)

# convertir cada una de las evaluaciones a enteros
# almacenar las reviews codificadas en review_ints
reviews_ints = []
for review in new_reviews:
    reviews_ints.append([model.wv[word] for word in review if word in model.wv.vocab])

# estadísticas acerca del vocabulario
print('Palabras únicas: ', len(model.wv.vocab))
print()

# valores de review
# print('Valores review: \n', reviews_ints[2], reviews[2])

words = sorted(model.wv.vocab.keys())
# Save words to file: words.txt
fp = open("words.txt", "w", encoding="utf-8")
for word in words:
    fp.write(word + '\n')
fp.close()

# print(words[1500:1700])

x_vals, y_vals, labels = reduce_dimensions(model)
plot_with_matplotlib(x_vals, y_vals, labels)

analogy_scores = model.wv.evaluate_word_analogies(datapath('questions-words.txt'))
print(analogy_scores[0])
# 1=positive, 1=neutral, 0=negative(etiquetas de conversión)
# encoded_labels = []
# for label in labels:
#    if label == 'neutral':
#        encoded_labels.append(1)
#    elif label == 'negative':
#        encoded_labels.append(0)
#    else:
#        encoded_labels.append(1)
#
# encoded_labels = np.asarray(encoded_labels)

# seq_length = 30

# features = pad_features(reviews_ints, seq_length, num_features)
# print(features.shape)
