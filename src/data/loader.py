from abc import ABC, abstractmethod
import pandas as pd
from typing import Tuple, Optional


class DataLoader(ABC):
    """Abstract base class for data loading with file/memory persistence abstraction."""
    
    @abstractmethod
    def load_train_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pass
    
    @abstractmethod
    def load_test_data(self) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def save_predictions(self, predictions: pd.DataFrame, filename: str) -> None:
        pass


class FileBasedDataLoader(DataLoader):
    """File-based data loader that reads from CSV files."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    def load_train_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.read_csv(f"{self.data_dir}/train.csv")
        X = df[['id', 'comment_text']]
        y = df[self.label_columns]
        return X, y
    
    def load_test_data(self) -> pd.DataFrame:
        df = pd.read_csv(f"{self.data_dir}/test.csv")
        return df[['id', 'comment_text']]
    
    def save_predictions(self, predictions: pd.DataFrame, filename: str) -> None:
        predictions.to_csv(filename, index=False)


class MemoryBasedDataLoader(DataLoader):
    """In-memory data loader for testing or when data is already loaded."""
    
    def __init__(self, train_df: Optional[pd.DataFrame] = None, 
                 test_df: Optional[pd.DataFrame] = None):
        self.train_df = train_df
        self.test_df = test_df
        self.label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        self.predictions_cache = {}
    
    def load_train_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.train_df is None:
            raise ValueError("Training data not loaded in memory")
        X = self.train_df[['id', 'comment_text']]
        y = self.train_df[self.label_columns]
        return X, y
    
    def load_test_data(self) -> pd.DataFrame:
        if self.test_df is None:
            raise ValueError("Test data not loaded in memory")
        return self.test_df[['id', 'comment_text']]
    
    def save_predictions(self, predictions: pd.DataFrame, filename: str) -> None:
        self.predictions_cache[filename] = predictions.copy()


def get_data_loader(mode: str = "file", **kwargs) -> DataLoader:
    """Factory function to get appropriate data loader based on mode."""
    if mode == "file":
        return FileBasedDataLoader(**kwargs)
    elif mode == "memory":
        return MemoryBasedDataLoader(**kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'file' or 'memory'")
