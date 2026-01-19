import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    hamming_loss, roc_auc_score, confusion_matrix
)
from typing import Dict, List, Tuple


class MetricsCalculator:
    """Calculate evaluation metrics for multi-label classification."""
    
    def __init__(self, label_columns: List[str] = None):
        if label_columns is None:
            self.label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        else:
            self.label_columns = label_columns
    
    def calculate_metrics(self, y_true, y_pred, threshold: float = 0.5) -> Dict[str, float]:
        """Calculate comprehensive metrics for multi-label classification."""
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred_binary),
            'precision': precision_score(y_true, y_pred_binary, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred_binary, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred_binary, average='weighted', zero_division=0),
            'hamming_loss': hamming_loss(y_true, y_pred_binary)
        }
        
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred, average='weighted')
        except ValueError:
            metrics['roc_auc'] = 0.0
        
        return metrics
    
    def calculate_per_label_metrics(self, y_true, y_pred, threshold: float = 0.5) -> Dict[str, Dict[str, float]]:
        """Calculate metrics for each label individually."""
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        per_label_metrics = {}
        
        for i, label in enumerate(self.label_columns):
            per_label_metrics[label] = {
                'precision': precision_score(y_true[:, i], y_pred_binary[:, i], zero_division=0),
                'recall': recall_score(y_true[:, i], y_pred_binary[:, i], zero_division=0),
                'f1_score': f1_score(y_true[:, i], y_pred_binary[:, i], zero_division=0),
                'support': int(y_true[:, i].sum())
            }
            
            try:
                per_label_metrics[label]['roc_auc'] = roc_auc_score(y_true[:, i], y_pred[:, i])
            except ValueError:
                per_label_metrics[label]['roc_auc'] = 0.0
        
        return per_label_metrics
    
    def get_confusion_matrices(self, y_true, y_pred, threshold: float = 0.5) -> Dict[str, np.ndarray]:
        """Get confusion matrix for each label."""
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        confusion_matrices = {}
        for i, label in enumerate(self.label_columns):
            confusion_matrices[label] = confusion_matrix(y_true[:, i], y_pred_binary[:, i])
        
        return confusion_matrices
    
    def print_metrics_summary(self, metrics: Dict[str, float]) -> None:
        """Print formatted metrics summary."""
        print("\n" + "="*50)
        print("OVERALL METRICS")
        print("="*50)
        for metric, value in metrics.items():
            print(f"{metric:20s}: {value:.4f}")
        print("="*50 + "\n")
    
    def print_per_label_metrics(self, per_label_metrics: Dict[str, Dict[str, float]]) -> None:
        """Print formatted per-label metrics."""
        print("\n" + "="*70)
        print("PER-LABEL METRICS")
        print("="*70)
        print(f"{'Label':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<12} {'Support':<10}")
        print("-"*70)
        
        for label, metrics in per_label_metrics.items():
            print(f"{label:<15} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
                  f"{metrics['f1_score']:<12.4f} {metrics['roc_auc']:<12.4f} {metrics['support']:<10}")
        
        print("="*70 + "\n")
