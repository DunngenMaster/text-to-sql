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
