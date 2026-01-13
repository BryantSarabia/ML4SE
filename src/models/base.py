from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional


class BaseModel(ABC):
    """Abstract base class for all toxicity classification models."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_trained = False
        self.label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model on the provided data."""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Predict probabilities for the provided data."""
        pass
    
    @abstractmethod
    def save_model(self, filepath: str):
        """Save the model to disk."""
        pass
    
    @abstractmethod
    def load_model(self, filepath: str):
        """Load the model from disk."""
        pass
    
    def get_model_name(self) -> str:
        """Return the model name."""
        return self.model_name
    
    def is_model_trained(self) -> bool:
        """Check if the model has been trained."""
        return self.is_trained
