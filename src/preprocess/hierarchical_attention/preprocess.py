"""This script runs standard NLP Preprocessing on the given unclean dataset from Params"""

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import pandas as pd
import numpy as np
import string
import tensorflow_datasets as tfds
from utils import pickle_object, load_pickle, Params
from tqdm import tqdm
import en_core_web_sm
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 


def clean_puncts(x):
    x = str(x)
    puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', '#', '—–', 'cnn', 'afp', 'fox']
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')
    return x

def clean_str(string):
    string = re.sub('cnn', '', string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    cleanr = re.compile('<.*?>')
    string = re.sub(cleanr, '', string)
    string = string.replace('_', '')

    return string.strip().lower()

def normalize(text,nlp):
    doc = nlp(text)
    filtered_sentences = []
    for sentence in doc.sents:                    
        sentence = clean_puncts(sentence)
        sentence = clean_str(sentence)                          
        filtered_sentences.append(sentence)
    return filtered_sentences

def clean_doc(params):
    nlp = en_core_web_sm.load()
    df = pd.read_csv(params.unclean_dataset)  
    df = df.sample(frac=1,random_state=232).reset_index(drop=True)
    df['text'] = df['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))

    #NEED TO OPTIMIZE LINE BELOW USING PD.APPLY
    text_tokens = []
    for row in tqdm(df['text']):    
        text_tokens.append(normalize(row,nlp))
    df['text'] = text_tokens
    
    #Removing Stopwords:
    stop = stopwords.words('english')
    df['text']=df['text'].apply(lambda x:" ".join(x for x in x.split() if x not in stop))
    
    #Removing Giveaway words about news sources:
    sources = list(df['site'].unique())
    sources_no_dot = [i.replace('.','') for i in sources]
    sources_no_dotcom = [i.replace('.com','') for i in sources]
    sources_no_news = [i.replace('news','') for i in sources_no_dotcom]
    for i in sources_no_dot:
        sources.append(i)
    for i in sources_no_dotcom:
        sources.append(i)
    for i in sources_no_news:
        sources.append(i)
    sources.append('(cnn)')
    sources.append('washington')
    sources.append('(cnn business)')
    sources.append('new york')
    sources.append('(ap)')
    sources.append('(fox)')
    sources = list(dict.fromkeys(sources))
    df['text']=df['text'].apply(lambda x:" ".join(x for x in x.split() if x not in sources))
    
    #Lemmatization
    lemmatizer = WordNetLemmatizer() 
    df['text'] = df['text'].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]))

    df.to_csv(params.clean_dataset,index=False)


params = Params('params.json')
clean_doc(params)
