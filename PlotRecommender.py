#!/usr/bin/env python
# coding: utf-8


# Import necessary libraries
import pandas as pd

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel

#Load the data
file_path = ""

df = pd.read_csv(file_path)


#Filter the American and British movies
movies = df[df['Origin/Ethnicity'].isin(['American', 'British'])]




nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
VERB_CODES = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}


def preprocess_sentences(text):
    text = text.lower()
    temp_sent = []
    words = nltk.word_tokenize(text)
    tags = nltk.pos_tag(words)
    for i, word in enumerate(words):
        if tags[i][1] in VERB_CODES:
            lemmatized = lemmatizer.lemmatize(word, 'v')
        else:
            lemmatized = lemmatizer.lemmatize(word)
        if lemmatized not in stop_words and lemmatized.isalpha():
            temp_sent.append(lemmatized)

    finalsent = ' '.join(temp_sent)
    finalsent = finalsent.replace("n't", " not")
    finalsent = finalsent.replace("'m", " am")
    finalsent = finalsent.replace("'s", " is")
    finalsent = finalsent.replace("'re", " are")
    finalsent = finalsent.replace("'ll", " will")
    finalsent = finalsent.replace("'ve", " have")
    finalsent = finalsent.replace("'d", " would")
    return finalsent

movies["plot_processed"] = movies["Plot"].apply(preprocess_sentences)

   # TF-IDF Vectorization
   tfidf_vectorizer = TfidfVectorizer()
   tfidf_matrix = tfidf_vectorizer.fit_transform(movies['plot_processed'])

   # Compute cosine similarity
   cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

   def get_recommendations(title, cosine_sim=cosine_sim):
    idx = movies[movies['Title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Get top 10 recommendations excluding the input movie
    movie_indices = [i[0] for i in sim_scores]
    return movies['Title'].iloc[movie_indices]




