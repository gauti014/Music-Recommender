import pandas as pd 
import re
import os
import nltk
from nltk.stem import PorterStemmer


dataset = pd.read_csv("C:/Users/Gautam/Downloads/archive (1)/spotify_millsongdata.csv")
#print(df.head)
dataset.isnull().sum()
dataset = dataset.drop('link', axis=1).reset_index(drop=True)
##print(dataset)
##print(dataset.head(5))

#cant take all the rows/songs, so taking sample
dataset = dataset.sample(3500)
##print(dataset.shape)


# Preprocessing- remove / capital small re etc
show = dataset['text']

# (r'^a-zA-Z0-9','' )
dataset['text'] = dataset['text'].str.lower().replace(r'^\w\s','').replace(r'\n','',regex=True).replace(r'\r','',regex=True)
#print(dataset['text'])

stemmer = PorterStemmer()
def token(txt):
    token=nltk.word_tokenize(txt)
    l = [stemmer.stem(i) for i in token]
    return"".join(l)
token("you are beautiful, bauti")
print(token)

# conversion to vector
dataset['text'].apply(lambda x: token(x))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
    
tfid = TfidfVectorizer(analyzer='word', stop_words='english')

matrix = tfid.fit_transform(dataset['text'])
similiar = cosine_similarity(matrix)

print(dataset[dataset['song'] =='Waiting For The Man'].index[0])



similiar[0]
dataset[dataset['song'] == 'Waiting For The Man']
def recommendation(song_df):
    idx = dataset[dataset['song'] == song_df].index[0]
    distances = sorted(list(enumerate(similiar[idx])),reverse=True,key=lambda x:x[1])
    
    songs = []
    for m_id in distances[1:21]:
        songs.append(dataset.iloc[m_id[0]].song)
        
    return songs
recommendation('Crying Over You')
import pickle
pickle.dump(similiar,open('similarity.pkl','wb'))
pickle.dump(dataset,open('df.pkl','wb'))
