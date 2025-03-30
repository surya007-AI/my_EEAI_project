# embeddings.py

from sklearn.feature_extraction.text import TfidfVectorizer

def get_tfidf_embd(df):
    """
    Generates TF-IDF embeddings for the 'interaction content' column in the dataframe.
    """
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['interaction content'])
    return tfidf_matrix
