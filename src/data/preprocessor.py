import re
import string
from typing import List, Optional
import nltk
from nltk.corpus import stopwords


class TextPreprocessor:
    """Text preprocessing pipeline for toxic comment classification."""
    
    def __init__(self, remove_stopwords: bool = True, use_lemmatization: bool = False):
        self.remove_stopwords = remove_stopwords
        self.use_lemmatization = use_lemmatization
        
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))
        
        self.contraction_map = {
            "ain't": "is not", "aren't": "are not", "can't": "cannot",
            "can't've": "cannot have", "could've": "could have",
            "couldn't": "could not", "didn't": "did not",
            "doesn't": "does not", "don't": "do not",
            "hadn't": "had not", "hasn't": "has not",
            "haven't": "have not", "he'd": "he would",
            "he'll": "he will", "he's": "he is",
            "i'd": "i would", "i'll": "i will",
            "i'm": "i am", "i've": "i have",
            "isn't": "is not", "it'd": "it would",
            "it'll": "it will", "it's": "it is",
            "let's": "let us", "shouldn't": "should not",
            "that's": "that is", "there's": "there is",
            "they'd": "they would", "they'll": "they will",
            "they're": "they are", "they've": "they have",
            "wasn't": "was not", "we'd": "we would",
            "we'll": "we will", "we're": "we are",
            "we've": "we have", "weren't": "were not",
            "what'll": "what will", "what're": "what are",
            "what's": "what is", "what've": "what have",
            "where's": "where is", "who'd": "who would",
            "who'll": "who will", "who's": "who is",
            "won't": "will not", "wouldn't": "would not",
            "you'd": "you would", "you'll": "you will",
            "you're": "you are", "you've": "you have"
        }
    
    def expand_contractions(self, text: str) -> str:
        """Expand contractions in text."""
        for contraction, expansion in self.contraction_map.items():
            text = text.replace(contraction, expansion)
        return text
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess a single text string."""
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        text = self.expand_contractions(text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        tokens = text.split()
        
        if self.remove_stopwords:
            tokens = [word for word in tokens if word not in self.stop_words and len(word) > 1]
        
        return ' '.join(tokens)
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Preprocess a batch of texts."""
        return [self.clean_text(text) for text in texts]
