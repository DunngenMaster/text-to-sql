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