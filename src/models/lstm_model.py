import numpy as np
import pickle
import json
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.models.base import BaseModel


class BiLSTMModel(BaseModel):
    """Bidirectional LSTM model for toxic comment classification."""
    
    def __init__(self, max_features=5000, max_len=100, embedding_dim=32, lstm_units=32):
        super().__init__("BiLSTM")
        self.max_features = max_features
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.tokenizer = Tokenizer(num_words=max_features)
        self.model = None
        self._build_model()
    
    def _build_model(self):
        self.model = Sequential([
            Embedding(self.max_features + 1, self.embedding_dim, input_length=self.max_len, mask_zero=True),
            Bidirectional(LSTM(self.lstm_units, activation='tanh', return_sequences=False)),
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
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=5, batch_size=64):
        self.tokenizer.fit_on_texts(X_train)
        
        X_train_seq = self.tokenizer.texts_to_sequences(X_train)
        X_train_pad = pad_sequences(X_train_seq, maxlen=self.max_len)
        
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_seq = self.tokenizer.texts_to_sequences(X_val)
            X_val_pad = pad_sequences(X_val_seq, maxlen=self.max_len)
            validation_data = (X_val_pad, y_val)
        
        history = self.model.fit(
            X_train_pad, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        self.is_trained = True
        return history
    
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_seq = self.tokenizer.texts_to_sequences(X)
        X_pad = pad_sequences(X_seq, maxlen=self.max_len)
        return self.model.predict(X_pad)
    
    def save_model(self, filepath: str):
        """Save model, tokenizer, and configuration for server deployment."""
        if filepath.endswith('.keras'):
            model_path = filepath.replace('.keras', '.h5')
        elif filepath.endswith('.h5'):
            model_path = filepath
        else:
            model_path = filepath + '.h5'
        
        self.model.save(model_path)
        
        tokenizer_path = model_path.replace('.h5', '_tokenizer.pkl')
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        config = {
            'max_features': self.max_features,
            'max_len': self.max_len,
            'embedding_dim': self.embedding_dim,
            'lstm_units': self.lstm_units,
            'model_name': self.get_model_name()
        }
        config_path = model_path.replace('.h5', '_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Model saved to: {model_path}")
        print(f"Tokenizer saved to: {tokenizer_path}")
        print(f"Config saved to: {config_path}")
    
    def load_model(self, filepath: str):
        if filepath.endswith('.keras'):
            filepath = filepath.replace('.keras', '.h5')
        elif not filepath.endswith('.h5'):
            filepath = filepath + '.h5'
        
        self.model = keras.models.load_model(filepath)
        
        tokenizer_path = filepath.replace('.h5', '_tokenizer.pkl')
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        self.is_trained = True
