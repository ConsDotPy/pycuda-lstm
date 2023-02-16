import numpy as np
import pandas as pd
from collections import Counter
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from gensim.models import word2vec

data = pd.read_csv('Tweets.csv')
reviews = np.array(data['text'])[:14000]
labels = np.array(data['airline_sentiment'])[:14000]
print(data['text'].loc[14639], "\nSENTIMENT:", data['airline_sentiment'].loc[14639])

# PREPROCESAMIENTO
punctuation = '!"#$%&\'()*+,-./:;<=>?[\\]^_`{|}~'

# eliminar puntuación
all_reviews = 'separator'.join(reviews)
all_reviews = all_reviews.lower()
all_text = ''.join([c for c in all_reviews if c not in punctuation])

# separar por nuevas líneas y espacios
reviews_split = all_text.split('separator')
all_text = ' '.join(reviews_split)

# crear lista de palabras
words = all_text.split()

# remover direcciones web, digitos y etiquetas(@identificador_twitter)
new_reviews = []
for review in reviews_split:
    review = review.split()
    new_text = []
    for word in review:
        if (word[0] != '@') and ('http' not in word) and (not word.isdigit()):
            new_text.append(word)
    new_reviews.append(new_text)

# crear diccionario para mapear las palabras en el vocabulario a enteros
counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}
int_to_vocab = {ii: word for word, ii in vocab_to_int.items()}

# convertir cada una de las evaluaciones a enteros
# almacenar las reviews codificadas en review_ints
reviews_ints = []
for review in new_reviews:
    reviews_ints.append([vocab_to_int[word] for word in review if word in vocab_to_int])

model = word2vec.Word2Vec(min_count=20,
                          window=5,
                          size=25,
                          sample=0.1,
                          alpha=0.05,
                          min_alpha=0.0001,
                          negative=20,
                          workers=4)
model.build_vocab(new_reviews, progress_per=10000)
model.train(new_reviews, total_examples=model.corpus_count, epochs=30, report_delay=1)

model.wv.save("word2vec.model")

model_name = "twitter_airline"
model.wv.save(model_name)

# 1=positive, 1=neutral, 0=negative(etiquetas de conversión)
encoded_labels = []
for label in labels:
    if label == 'neutral':
        encoded_labels.append(1)
    elif label == 'negative':
        encoded_labels.append(0)
    else:
        encoded_labels.append(1)

encoded_labels = np.asarray(encoded_labels)


def pad_features(reviews_ints, seq_length):
    """Retornar valores de review_ints, cada review es truncada o rellenada con 0s dada la longitud
        seq_length."""

    # obtener el shape(row x col)
    features = np.zeros((len(reviews_ints), seq_length), dtype=int)

    # para cada review, hacer el padding de cada palabra
    for i, row in enumerate(reviews_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]

    return features


seq_length = 30
dim = 25

features = pad_features(reviews_ints, seq_length=seq_length)
print(features[0:2].flatten())


def emb_Word2Vec(features):
    new_feat = np.array([])
    for i, x in enumerate(features.flatten()):
        if x != 0 and int_to_vocab[x] in model.wv:
            new_feat = np.concatenate((new_feat, model.wv[int_to_vocab[x]]), axis=None)
            print(new_feat.shape)
        else:
            new_feat = np.concatenate((new_feat, np.zeros(dim)), axis=None)
            print(new_feat.shape)
    return new_feat.reshape((2, seq_length, dim))


# dataloaders
batch_size = 100

hola = emb_Word2Vec(features[:2])
print(hola)
# print(hola)
# print(type(new_feat))
split_frac = 0.8

# se debe separar el dataset training, validation y test (features, labels -> x , y)

split_idx = int(len(features) * split_frac)
train_x, remaining_x = features[:split_idx], features[split_idx:]
train_y, remaining_y = encoded_labels[:split_idx], encoded_labels[split_idx:]

test_idx = int(len(remaining_x) * 0.5)
val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]

# forma de datos resultantes
print("\t\t\tShapes de Tr, Val, Test:")
print("Train set: \t\t{0}, {1}".format(train_x.shape, train_y.shape),
      "\nValidation set: \t{0}, {1}".format(val_x.shape, val_y.shape),
      "\nTest set: \t\t{0}, {1}".format(test_x.shape, test_y.shape))

# create Tensor datasets
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

# make sure the SHUFFLE the training data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

# obtain one batch of training data
dataiter = iter(train_loader)
sample_x, sample_y = dataiter.next()

print('Sample input size: ', sample_x.size())  # batch_size, seq_length
print('Sample input: \n', sample_x)
print()
print('Sample label size: ', sample_y.size())  # batch_size
print('Sample label: \n', sample_y)

# First checking if GPU is available
train_on_gpu = False


# torch.cuda.is_available()

# if train_on_gpu:
#    print('Training on GPU.')
# else:
#    print('No GPU available, training on CPU.')


class SentimentRNN(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super(SentimentRNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=drop_prob, batch_first=True)

        # dropout layer
        self.dropout = nn.Dropout(0.3)

        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)

        # embeddings and lstm_out
        print(x)
        x = x.long()
        embeds = self.embedding(x)
        print(embeds.shape)
        lstm_out, hidden = self.lstm(embeds, hidden)

        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        # sigmoid function
        sig_out = self.sig(out)

        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]  # get last batch of labels

        # return last sigmoid output and hidden state
        return sig_out, hidden

    def init_hidden(self, batch_size):
        """ Initializes hidden state """
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if train_on_gpu:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden


# Instantiate the model w/ hyperparams
vocab_size = len(vocab_to_int) + 1  # +1 for the 0 padding + our word tokens
output_size = 1
embedding_dim = 25
hidden_dim = 128
n_layers = 1

net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

# print(net)

# loss and optimization functions
lr = 0.001

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

# training params

epochs = 10

counter = 0
print_every = 10
clip = 5  # gradient clipping

# move model to GPU, if available
if train_on_gpu:
    net.cuda()

net.train()
# train for some number of epochs
for e in range(epochs):
    # initialize hidden state
    h = net.init_hidden(batch_size)

    # batch loop
    for inputs, labels in train_loader:
        counter += 1

        if (train_on_gpu):
            inputs, labels = inputs.cuda(), labels.cuda()

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        # zero accumulated gradients
        net.zero_grad()

        # get the output from the model
        output, h = net(inputs, h)

        # calculate the loss and perform backprop
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        # loss stats
        if counter % print_every == 0:
            # Get validation loss
            val_h = net.init_hidden(batch_size)
            val_losses = []
            net.eval()
            for inputs, labels in valid_loader:

                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                val_h = tuple([each.data for each in val_h])

                if train_on_gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()

                output, val_h = net(inputs, val_h)
                val_loss = criterion(output.squeeze(), labels.float())
                FScore = f1_score(labels.cpu(), output.cpu() > 0.5, average="binary")
                val_losses.append(val_loss.item())

            net.train()
            print("Epoch: {}/{}...".format(e + 1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))

# Get test data loss and accuracy

test_losses = []  # track loss
num_correct = 0
FScore = 0
# init hidden state
h = net.init_hidden(batch_size)

net.eval()
# iterate over test data
for inputs, labels in test_loader:

    # Creating new variables for the hidden state, otherwise
    # we'd backprop through the entire training history
    h = tuple([each.data for each in h])

    if (train_on_gpu):
        inputs, labels = inputs.cuda(), labels.cuda()

    # get predicted outputs
    output, h = net(inputs, h)

    # calculate loss
    test_loss = criterion(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())

    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze())  # rounds to the nearest integer

    # compare predictions to true label
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    FScore += f1_score(labels.cpu(), output.cpu() > 0.5, average="binary")
    num_correct += np.sum(correct)

# -- stats! -- ##
# avg test loss
print("Test loss: {:.3f}".format(np.mean(test_losses)))

# accuracy over all test data
test_acc = num_correct / len(test_loader.dataset)
print("Test accuracy: {:.3f}".format(test_acc))
print("Test F1: {:.3f}".format(FScore / len(test_loader)))
# negative test review
test_review = "@AmericanAir you have my money, you change my flight, and don't answer your phones! Any other suggestions so I can make my commitment??"


def tokenize_review(test_review):
    test_review = test_review.lower()  # lowercase
    # get rid of punctuation
    test_text = ''.join([c for c in test_review if c not in punctuation])

    # splitting by spaces
    test_words = test_text.split()

    # get rid of web address, twitter id, and digit
    new_text = []
    for word in test_words:
        if (word[0] != '@') & ('http' not in word) & (~word.isdigit()):
            new_text.append(word)

    # tokens
    test_ints = []
    test_ints.append([vocab_to_int[word] for word in new_text])

    return test_ints


# test code and generate tokenized review
test_ints = tokenize_review(test_review)
# print(test_ints)

# test sequence padding
seq_length = 30
features = pad_features(test_ints, seq_length)

# print(features)

# test conversion to tensor and pass into your model
feature_tensor = torch.from_numpy(features)


# print(feature_tensor.size())


def predict(net, test_review, sequence_length=30):
    net.eval()

    # tokenize review
    test_ints = tokenize_review(test_review)

    # pad tokenized sequence
    seq_length = sequence_length
    features = pad_features(test_ints, seq_length)

    # convert to tensor to pass into your model
    feature_tensor = torch.from_numpy(features)

    batch_size = feature_tensor.size(0)

    # initialize hidden state
    h = net.init_hidden(batch_size)

    if (train_on_gpu):
        feature_tensor = feature_tensor.cuda()

    # get the output from the model
    output, h = net(feature_tensor, h)

    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze())
    # printing output value, before rounding
    print('Predicción, Valor: {:.6f}'.format(output.item()))

    # print custom response
    if (pred.item() == 1):
        print("No negativo.")
    else:
        print("Negativo.")


seq_length = 30  # good to use the length that was trained on

# call function on negative review
test_review_neg = "@AmericanAir you have my money, you change my flight, and don't answer your phones! Any other suggestions so I can make my commitment??"
predict(net, test_review_neg, seq_length)

# call function on positive review
test_review_pos = "@AmericanAir thank you we got on a different flight to Chicago."
predict(net, test_review_pos, seq_length)

# call function on neutral review
test_review_neu = "@AmericanAir i need someone to help me out"
predict(net, test_review_neu, seq_length)
