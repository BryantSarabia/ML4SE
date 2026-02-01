import numpy as np
import pickle
from pathlib import Path


class ToxicityPredictor:
    """Model predictor for toxic comment classification."""
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.model_path = model_path
        self.label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
                self.model = data
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def predict(self, comment: str) -> dict:
        if self.model is None:
            return self._mock_prediction(comment)
        
        try:
            if hasattr(self.model, 'predict'):
                predictions = self.model.predict([comment])[0]
            else:
                predictions = self._mock_prediction(comment)['predictions']
                return {
                    'comment': comment,
                    'predictions': predictions,
                    'risk_level': self._calculate_risk_level(predictions),
                    'highest_risk': self._get_highest_risk(predictions)
                }
        except Exception as e:
            print(f"Prediction error: {e}")
            return self._mock_prediction(comment)
        
        predictions_dict = {label: float(pred) for label, pred in zip(self.label_columns, predictions)}
        
        return {
            'comment': comment,
            'predictions': predictions_dict,
            'risk_level': self._calculate_risk_level(predictions_dict),
            'highest_risk': self._get_highest_risk(predictions_dict)
        }
    
    def _mock_prediction(self, comment: str) -> dict:
        toxic_words = ['stupid', 'idiot', 'hate', 'kill', 'die', 'fuck', 'shit']
        
        toxicity_score = sum(1 for word in toxic_words if word in comment.lower()) * 0.15
        toxicity_score = min(toxicity_score, 0.95)
        
        predictions = {
            'toxic': toxicity_score,
            'severe_toxic': toxicity_score * 0.3,
            'obscene': toxicity_score * 0.7,
            'threat': toxicity_score * 0.2,
            'insult': toxicity_score * 0.6,
            'identity_hate': toxicity_score * 0.25
        }
        
        return {
            'comment': comment,
            'predictions': predictions,
            'risk_level': self._calculate_risk_level(predictions),
            'highest_risk': self._get_highest_risk(predictions)
        }
    
    def _calculate_risk_level(self, predictions: dict) -> str:
        max_prob = max(predictions.values())
        
        if max_prob >= 0.7:
            return 'high'
        elif max_prob >= 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _get_highest_risk(self, predictions: dict) -> str:
        return max(predictions.items(), key=lambda x: x[1])[0]
