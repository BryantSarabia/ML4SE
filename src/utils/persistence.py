import os
import pickle
import json
from typing import Any, Dict
from datetime import datetime


class ModelPersistence:
    """Utility class for saving and loading models with metadata."""
    
    @staticmethod
    def save_model_with_metadata(model, filepath: str, metadata: Dict[str, Any] = None):
        """Save model with metadata (training date, parameters, metrics, etc.)."""
        model.save_model(filepath)
        
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'model_name': model.get_model_name(),
            'save_timestamp': datetime.now().isoformat(),
            'is_trained': model.is_model_trained()
        })
        
        metadata_path = filepath.replace('.pkl', '_metadata.json').replace('.keras', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved to {filepath}")
        print(f"Metadata saved to {metadata_path}")
    
    @staticmethod
    def load_model_with_metadata(model, filepath: str) -> Dict[str, Any]:
        """Load model and return metadata."""
        model.load_model(filepath)
        
        metadata_path = filepath.replace('.pkl', '_metadata.json').replace('.keras', '_metadata.json')
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"Model loaded from {filepath}")
            print(f"Metadata: {metadata}")
            return metadata
        else:
            print(f"Model loaded from {filepath} (no metadata found)")
            return {}
    
    @staticmethod
    def create_model_directory(base_dir: str = "models"):
        """Create directory structure for saving models."""
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        
        subdirs = ['baseline', 'deep_learning', 'ensemble']
        for subdir in subdirs:
            path = os.path.join(base_dir, subdir)
            if not os.path.exists(path):
                os.makedirs(path)
        
        return base_dir
