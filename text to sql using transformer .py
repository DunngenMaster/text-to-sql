# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 14:39:47 2025

@author: ashutosh

https://github.com/DunngenMaster/text-to-sql.git

# additonal libraries installation

pip install datasets
pip install nltk
pip install seaborn
pip install wordcloud
pip install torch
pip install transformer
pip install sentencepiece
pip install tensorflow

"""


# imports
import nltk
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import string
from nltk.corpus import stopwords
from tqdm import tqdm

import nltk
from datasets import load_dataset
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import torch

from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torch.optim import Adam

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, LayerNormalization, Dropout, MultiHeadAttention




# download stopwords and Dictnoray for our NLP task
nltk.download('stopwords')
nltk.download('wordnet')

# Load Dataset
metaData = load_dataset(
    'wikisql', 
    trust_remote_code=True
    )

metaData


def get_name_and_query_dataset(setName: str):
    sampleDataset = []
    
    for each in tqdm(metaData[setName]):
        row = { 
            'question': each['question'], 
            'sql': each['sql']['human_readable']
        }
        sampleDataset.append(row)  

    return pd.DataFrame(sampleDataset) 

trainDF = get_name_and_query_dataset('train')
testDF = get_name_and_query_dataset('test')
validationDF = get_name_and_query_dataset('validation')

df = pd.concat([trainDF,
                validationDF,
                testDF
                ])

#check null
df.isnull().sum()

#convert all text item to lower case
df['question'] = df['question'].str.lower()
df['sql'] = df['sql'].str.lower()
df.head()

#remove punctuation
def Punctuations(sentence):
    return sentence.translate(str.maketrans('', '', string.punctuation))

df['question'] = df['question'].apply(Punctuations)

#remove stopwords from SQL

sql_stopwords = ['SELECT', 'FROM', 'WHERE', 'AND', 'OR', 'NOT', 'AS', 'IN', 'ON', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 'OUTER', 'GROUP', 'BY', 'HAVING', 'ORDER', 'ASC', 'DESC', 'LIMIT', 'OFFSET', 'DISTINCT', 'ALL', 'UNION', 'EXCEPT', 'INTERSECT']
print(f'Number of stop words that are present in sql: {len(sql_stopwords)}')

for i in range(len(sql_stopwords)):
    sql_stopwords[i] = sql_stopwords[i].lower()
    
stopWords = stopwords.words('english')
print(f'Number of stop words that are present in english: {len(sql_stopwords)}')

for i in sql_stopwords:
    if i in stopWords:
        stopWords.remove(i)
        
print(f'Final stopwords list length: {len(stopWords)}')

def stopWordsRemove(sentence):
    finalSentence = ""
    for word in sentence.split():
        if word not in stopWords:
            finalSentence += word + " "
    finalSentence = finalSentence.strip()
    return finalSentence

df['question'] = df['question'].apply(stopWordsRemove)

#Lemmatization
lemma = WordNetLemmatizer()
def lemmatization(text):
    return " ".join([lemma.lemmatize(word, pos = 'n') for word in text.split()])

df['question'] = df['question'].apply(lemmatization)

# Remove extra punctuation from SQL
exclude_punctuations = "=/()"
punctuations = string.punctuation
punctuations = punctuations.translate(str.maketrans('', '', exclude_punctuations))

def deletePunctuationSQL(sentence):
    translator = str.maketrans('', '', punctuations)
    return sentence.translate(translator)

df['sql'] = df['sql'].apply(deletePunctuationSQL)

#Plot question length
df['questionTokens'] = df['question'].apply(lambda x: len(x.split()))
plt.figure(figsize = (10, 6))

sns.histplot(df['questionTokens'], bins = 30, kde = True)
plt.title('1. Distribution Of Question Lengths')
plt.xlabel('Question Length (in words)')
plt.ylabel('Frequency')
plt.show()

#plot word cloud
all_questions = ' '.join(df['question'].tolist())
wordcloud = WordCloud(width=800, height=400, background_color ='white').generate(all_questions)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('2. Word Cloud of Questions')
plt.show()


df['questionType'] = df['question'].apply(lambda x: x.split()[0].lower())
question_type_counts = df['questionType'].value_counts().head(10)
question_type_counts = question_type_counts.reset_index()
question_type_counts.columns = ['Question Type', 'Frequency']
plt.figure(figsize=(10, 6))
sns.barplot(x='Frequency', y='Question Type', data=question_type_counts, palette='coolwarm')
plt.title('4. Top Question Types')
plt.xlabel('Frequency')
plt.ylabel('Question Type')
plt.show()


dv=df.copy()

dv['query_length'] = df['sql'].apply(lambda x: len(x.split()))
dv['sql_complexity'] = df['sql'].apply(lambda x: x.upper().count('SELECT') + x.upper().count('WHERE'))
dv['num_conditions'] = df['sql'].apply(lambda x: x.upper().count('AND') + x.upper().count('OR') + 1 if 'WHERE' in x.upper() else 0)
dv['sql_length'] = df['sql'].apply(lambda x: len(x.split()))
corr_matrix = dv[['query_length', 'sql_complexity', 'num_conditions', 'sql_length']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title('6. Correlation Matrix of SQL Query Features')
plt.show()

sns.pairplot(dv[['query_length', 'sql_complexity', 'num_conditions', 'sql_length']])
plt.suptitle('7. Pairwise Relationships Between Features', verticalalignment='top')
plt.show()

X = df['question'].tolist()
y = df['sql'].tolist()


tokenizer = T5Tokenizer.from_pretrained('t5-small')
def tokenize_texts(text_list, max_length=512):
    return tokenizer(
        text_list,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='np'
    )

X_tokenized = tokenize_texts(X)
y_tokenized = tokenize_texts(y)

X_train, X_temp, y_train, y_temp = train_test_split(X_tokenized['input_ids'], y_tokenized['input_ids'], test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


#training the model 
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout_rate):
    attention = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(inputs, inputs)
    attention = Dropout(dropout_rate)(attention)
    attention = LayerNormalization(epsilon=1e-6)(attention + inputs)

    ff_output = Dense(ff_dim, activation='relu')(attention)
    ff_output = Dense(inputs.shape[-1])(ff_output)
    ff_output = Dropout(dropout_rate)(ff_output)
    encoder_output = LayerNormalization(epsilon=1e-6)(ff_output + attention)
    return encoder_output

def transformer_decoder(inputs, enc_output, head_size, num_heads, ff_dim, dropout_rate):
    self_attention = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(inputs, inputs)
    self_attention = Dropout(dropout_rate)(self_attention)
    self_attention = LayerNormalization(epsilon=1e-6)(self_attention + inputs)

    attention = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(self_attention, enc_output)
    attention = Dropout(dropout_rate)(attention)
    attention = LayerNormalization(epsilon=1e-6)(attention + self_attention)

    ff_output = Dense(ff_dim, activation='relu')(attention)
    ff_output = Dense(inputs.shape[-1])(ff_output)
    ff_output = Dropout(dropout_rate)(ff_output)
    decoder_output = LayerNormalization(epsilon=1e-6)(ff_output + attention)
    return decoder_output


embedding_dim = 256
vocab_size = 32128
head_size = 64
num_heads = 8
ff_dim = 512
dropout_rate = 0.1
max_length = 512

inputs_enc = Input(shape=(None,))
inputs_dec = Input(shape=(None,))

enc_emb = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(inputs_enc)
dec_emb = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(inputs_dec)

enc_output = transformer_encoder(enc_emb, head_size, num_heads, ff_dim, dropout_rate)
dec_output = transformer_decoder(dec_emb, enc_output, head_size, num_heads, ff_dim, dropout_rate)

final_output = Dense(vocab_size, activation='softmax')(dec_output)

model = Model(inputs=[inputs_enc, inputs_dec], outputs=final_output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

model.summary()

y_train_shifted = np.pad(y_train[:, :-1], ((0, 0), (1, 0)), mode='constant', constant_values=tokenizer.pad_token_id)
y_val_shifted = np.pad(y_val[:, :-1], ((0, 0), (1, 0)), mode='constant', constant_values=tokenizer.pad_token_id)

history = model.fit(
    [X_train, y_train_shifted],
    y_train,
    validation_data=([X_val, y_val_shifted], y_val),
    batch_size=5,
    epochs=10
)
model.save('my_transformer_model.h5')