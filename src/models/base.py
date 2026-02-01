from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Abstract base class for all toxicity classification models."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_trained = False
        self.label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass
    
    @abstractmethod
    def save_model(self, filepath: str):
        pass
    
    @abstractmethod
    def load_model(self, filepath: str):
        pass
    
    def get_model_name(self) -> str:
        return self.model_name
    
    def is_model_trained(self) -> bool:
        return self.is_trained
