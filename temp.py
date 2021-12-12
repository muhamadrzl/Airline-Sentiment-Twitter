import pandas as pd 
import numpy as np
import re
import collections
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
# Packages for data preparation
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
# Packages for modeling
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import(LSTM,
                                    Embedding,
                                    BatchNormalization,
                                    Dense,
                                    TimeDistributed,
                                    Dropout,
                                    Bidirectional,
                                    Flatten,
                                    GlobalMaxPool1D)
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('Tweets.csv.zip')
print(df)
df = df[['text','airline_sentiment']]
import string

def clean_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

import nltk
stemer = nltk.SnowballStemmer('english')
def stem_text(text):
    text = ' '.join(stemer.stem(word) for word in text.split(' '))
    return text

df['message_clean'] = df['text'].apply(stem_text)
le = LabelEncoder()
df['airline_sentiment'] = le.fit_transform(df.airline_sentiment)
stemmer = nltk.SnowballStemmer("english")

def stemm_text(text):
    text = ' '.join(stemmer.stem(word) for word in text.split(' '))
    return text
df['message_clean'] = df['message_clean'].apply(stemm_text)
df.head()
df.dropna(inplace=True)
text = df.message_clean
target = to_categorical(df.airline_sentiment)
word_tokenizer = Tokenizer()
word_tokenizer.fit_on_texts(text)
vocab_length = len(word_tokenizer.word_index)+1

def embed(corpus): 
    return word_tokenizer.texts_to_sequences(corpus) #sentence2vec kayak bow, tfidf

longest_train = max(text, key=lambda sentence: len(word_tokenize(sentence)))
length_long_sentence = len(word_tokenize(longest_train))

train_padded_sentences = pad_sequences(
    embed(text), 
    length_long_sentence, 
    padding='post'
)


embeddings_dictionary = dict()
embedding_dim = 100

# Load GloVe 100D embeddings
with open('glove.6B.100d.txt','r',encoding = 'utf-8') as fp:
    for line in fp.readlines():
        records = line.split()
        word = records[0]
        vector_dimensions = np.asarray(records[1:], dtype='float32')
        embeddings_dictionary [word] = vector_dimensions #dibikin matrix 

embedding_matrix = np.zeros((vocab_length, embedding_dim)) #

for word, index in word_tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word) #
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector #bikin sentence2vector dari yang ada di glove
        
def glove_lstm():
    model = Sequential()
    
    model.add(Embedding(
        input_dim=embedding_matrix.shape[0], 
        output_dim=embedding_matrix.shape[1], 
        #weights = [embedding_matrix], 
        input_length=length_long_sentence
    ))
    
    model.add(Bidirectional(LSTM(
        length_long_sentence, 
        return_sequences = True, 
        recurrent_dropout=0.2
    )))
    
    model.add(GlobalMaxPool1D())
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(length_long_sentence, kernel_regularizer=regularizers.l2(0.001),activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(length_long_sentence, kernel_regularizer=regularizers.l2(0.001),activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation = 'softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

model = glove_lstm()
model.summary()
X_train, X_test, y_train, y_test = train_test_split(
    train_padded_sentences, 
    target, 
    test_size=0.25
)

model = glove_lstm()

checkpoint = ModelCheckpoint(
    'model.h5', 
    monitor = 'val_loss', 
    save_best_only = True
)
reduce_lr = ReduceLROnPlateau(
    monitor = 'val_loss', 
    factor = 0.2, 
    patience = 5,                        
    min_lr = 0.001
)
history = model.fit(
    X_train, 
    y_train, 
    epochs = 20,
    batch_size = 32,
    validation_data = (X_test, y_test),
    callbacks = [reduce_lr, checkpoint]
)
