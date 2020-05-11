import numpy as np
import pandas as pd
import nltk
import collections
from sklearn.feature_extraction.text import TfidfVectorizer
import csv

cluster=pd.ExcelFile('insert input file here')
df=cluster.parse('Sheet1')
full_text=[]
full_doc=[]
for i in range (1,len(df)+1):
    tweet=df[0:i].values
    text=nltk.word_tokenize(tweet[(i-1),0])
    full_doc.append(tweet[(i-1),0])
    for j in text:
        if (j.isalpha())==True:
            full_text.append(j)
lower_text=[token.lower() for token in full_text]
counter_var=collections.Counter(lower_text)
top30=counter_var.most_common(30)


vectorizer=TfidfVectorizer(lowercase=False,stop_words='english')
tfidf=vectorizer.fit_transform(full_doc)
tfidf_scores=[]
shape=tfidf.shape
for j in range(shape[0]):
    for i in range (len(vectorizer.get_feature_names())):
        if tfidf[j,i]!=0:
            tfidf_scores.append((j,vectorizer.get_feature_names()[i],tfidf[j,i]))

with open('desired bag of words filename here.csv',mode='w',newline='') as csv_file:
    fieldnames=['word','frequency']
    writer=csv.DictWriter(csv_file,fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(top30)):
        writer.writerow({'word':str(top30[i][0]),'frequency':int(top30[i][1])})

with open('desired tf-idf filename here.csv',mode='w',newline='') as csv_file:
    fieldnames=['tweet_number','word','tf-idf_score']
    writer=csv.DictWriter(csv_file,fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(tfidf_scores)):
        writer.writerow({'tweet_number':str(tfidf_scores[i][0]),'word':str(tfidf_scores[i][1]),'tf-idf_score':tfidf_scores[i][2]})