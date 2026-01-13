import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from src.models.base import BaseModel


class TfidfLogisticRegression(BaseModel):
    """TF-IDF + Logistic Regression baseline model."""
    
    def __init__(self, max_features=5000, C=1.0):
        super().__init__("TF-IDF + Logistic Regression")
        self.max_features = max_features
        self.C = C
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
        self.model = OneVsRestClassifier(LogisticRegression(C=C, max_iter=1000, random_state=42))
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_train_tfidf, y_train)
        self.is_trained = True
        return self
    
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        X_tfidf = self.vectorizer.transform(X)
        return self.model.predict_proba(X_tfidf)
    
    def save_model(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump({'vectorizer': self.vectorizer, 'model': self.model}, f)
    
    def load_model(self, filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.vectorizer = data['vectorizer']
            self.model = data['model']
            self.is_trained = True


class TfidfNaiveBayes(BaseModel):
    """TF-IDF + Multinomial Naive Bayes baseline model."""
    
    def __init__(self, max_features=5000, alpha=1.0):
        super().__init__("TF-IDF + Naive Bayes")
        self.max_features = max_features
        self.alpha = alpha
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
        self.model = OneVsRestClassifier(MultinomialNB(alpha=alpha))
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_train_tfidf, y_train)
        self.is_trained = True
        return self
    
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        X_tfidf = self.vectorizer.transform(X)
        return self.model.predict_proba(X_tfidf)
    
    def save_model(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump({'vectorizer': self.vectorizer, 'model': self.model}, f)
    
    def load_model(self, filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.vectorizer = data['vectorizer']
            self.model = data['model']
            self.is_trained = True
