import numpy as np
import pickle
import json
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences


class ToxicityPredictor:
    """Model predictor for toxic comment classification with BiLSTM."""
    
    def __init__(self, model_dir: str = None):
        """
        Initialize predictor.
        
        Args:
            model_dir: Directory containing model files (default: ../models)
        """
        self.model = None
        self.tokenizer = None
        self.config = None
        self.max_len = None
        self.label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        
        if model_dir is None:
            model_dir = '../models'
        
        try:
            self.load_model(model_dir)
        except Exception as e:
            print(f"Warning: Could not load model from {model_dir}: {e}")
            print("Server will use mock predictions")
    
    def load_model(self, model_dir: str):
        """
        Load Keras model, tokenizer, and configuration.
        
        Args:
            model_dir: Path to model directory or base filename
        """
        try:
            model_path = Path(model_dir)
            
            # If it's a directory, append the default model name
            if model_path.is_dir():
                base_path = model_path / "bilstm_toxic_classifier"
            # If it's a file path (with extension), remove extension
            elif model_path.suffix in ['.h5', '.keras']:
                base_path = model_path.with_suffix('')
            # If it's a base path without extension, use as-is
            else:
                base_path = model_path
            
            weights_path = str(base_path) + "_weights.h5"
            tokenizer_path = str(base_path) + "_tokenizer.pkl"
            config_path = str(base_path) + "_config.json"
            
            # Load config first to get architecture parameters
            print(f"Loading config from: {config_path}")
            with open(config_path, 'r') as f:
                self.config = json.load(f)
                self.max_len = self.config['max_len']
            
            # Reconstruct model architecture from config
            print("Reconstructing model architecture...")
            max_features = self.config.get('max_features', 3000)
            max_len = self.config.get('max_len', 100)
            embedding_dim = self.config.get('embedding_dim', 32)
            lstm_units = self.config.get('lstm_units', 32)
            
            self.model = Sequential([
                Embedding(max_features + 1, embedding_dim, input_length=max_len, mask_zero=True),
                Bidirectional(LSTM(lstm_units, activation='tanh', return_sequences=False)),
                Dense(64, activation='relu'),
                Dropout(0.3),
                Dense(64, activation='relu'),
                Dropout(0.3),
                Dense(6, activation='sigmoid')
            ])
            
            self.model.compile(
                loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy']
            )
            
            # Load weights into the reconstructed model
            print(f"Loading model weights from: {weights_path}")
            self.model.load_weights(weights_path)
            
            print(f"Loading tokenizer from: {tokenizer_path}")
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            
            print("Model loaded successfully!")
            print(f"  Model type: {self.config.get('model_name', 'BiLSTM')}")
            print(f"  Max sequence length: {self.max_len}")
            print(f"  Vocabulary size: {max_features}")
            
        except FileNotFoundError as e:
            print(f"Error: Model files not found - {e}")
            print("  Falling back to mock predictions")
            self.model = None
        except Exception as e:
            print(f"Error loading model: {e}")
            print("  Falling back to mock predictions")
            self.model = None
    
    def predict(self, comment: str) -> dict:
        """
        Predict toxicity for a comment.
        
        Args:
            comment: Text comment to analyze
            
        Returns:
            dict with predictions, risk_level, and highest_risk
        """
        if self.model is None or self.tokenizer is None:
            return self._mock_prediction(comment)
        
        try:
            sequence = self.tokenizer.texts_to_sequences([comment])
            padded = pad_sequences(sequence, maxlen=self.max_len)
            
            predictions = self.model.predict(padded, verbose=0)[0]
            
            predictions_dict = {
                label: float(pred) 
                for label, pred in zip(self.label_columns, predictions)
            }
            
            return {
                'comment': comment,
                'predictions': predictions_dict,
                'risk_level': self._calculate_risk_level(predictions_dict),
                'highest_risk': self._get_highest_risk(predictions_dict)
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return self._mock_prediction(comment)
    
    def _mock_prediction(self, comment: str) -> dict:
        """Fallback mock prediction based on keyword matching."""
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
            'highest_risk': self._get_highest_risk(predictions),
            'note': 'Using mock predictions - model not loaded'
        }
    
    def _calculate_risk_level(self, predictions: dict) -> str:
        """Calculate overall risk level from predictions."""
        max_prob = max(predictions.values())
        
        if max_prob >= 0.7:
            return 'high'
        elif max_prob >= 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _get_highest_risk(self, predictions: dict) -> str:
        """Get the category with highest toxicity probability."""
        return max(predictions.items(), key=lambda x: x[1])[0]
