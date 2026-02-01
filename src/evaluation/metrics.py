import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    hamming_loss, roc_auc_score, confusion_matrix
)
from sklearn.model_selection import KFold
from typing import Dict, List, Tuple, Any
import copy


class MetricsCalculator:
    """Calculate evaluation metrics for multi-label classification."""
    
    def __init__(self, label_columns: List[str] = None):
        if label_columns is None:
            self.label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        else:
            self.label_columns = label_columns
    
    def calculate_metrics(self, y_true, y_pred, threshold: float = 0.5) -> Dict[str, float]:
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
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        per_label_metrics = {}
        
        for i, label in enumerate(self.label_columns):
            support = int(y_true[:, i].sum())
            
            if support == 0 or y_pred_binary[:, i].sum() == 0:
                per_label_metrics[label] = {
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'support': support
                }
            else:
                per_label_metrics[label] = {
                    'precision': precision_score(y_true[:, i], y_pred_binary[:, i], zero_division=0),
                    'recall': recall_score(y_true[:, i], y_pred_binary[:, i], zero_division=0),
                    'f1_score': f1_score(y_true[:, i], y_pred_binary[:, i], zero_division=0),
                    'support': support
                }
            
            try:
                per_label_metrics[label]['roc_auc'] = roc_auc_score(y_true[:, i], y_pred[:, i])
            except ValueError:
                per_label_metrics[label]['roc_auc'] = 0.0
        
        return per_label_metrics
    
    def get_confusion_matrices(self, y_true, y_pred, threshold: float = 0.5) -> Dict[str, np.ndarray]:
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        confusion_matrices = {}
        for i, label in enumerate(self.label_columns):
            confusion_matrices[label] = confusion_matrix(y_true[:, i], y_pred_binary[:, i])
        
        return confusion_matrices
    
    def print_metrics_summary(self, metrics: Dict[str, float]) -> None:
        print("\n" + "="*50)
        print("OVERALL METRICS")
        print("="*50)
        for metric, value in metrics.items():
            print(f"{metric:20s}: {value:.4f}")
        print("="*50 + "\n")
    
    def print_per_label_metrics(self, per_label_metrics: Dict[str, Dict[str, float]]) -> None:
        print("\n" + "="*70)
        print("PER-LABEL METRICS")
        print("="*70)
        print(f"{'Label':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<12} {'Support':<10}")
        print("-"*70)
        
        for label, metrics in per_label_metrics.items():
            print(f"{label:<15} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
                  f"{metrics['f1_score']:<12.4f} {metrics['roc_auc']:<12.4f} {metrics['support']:<10}")
        
        print("="*70 + "\n")


class KFoldValidator:
    """K-Fold cross-validation for multi-label classification."""
    
    def __init__(self, n_splits: int = 10, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    def cross_validate(self, model, X, y, metrics_calculator: MetricsCalculator) -> Dict[str, Any]:
        """Perform k-fold cross-validation and return aggregated metrics."""
        fold_metrics = []
        fold_predictions = []
        
        X_array = np.array(X) if not isinstance(X, np.ndarray) else X
        y_array = np.array(y) if not isinstance(y, np.ndarray) else y
        
        for fold, (train_idx, val_idx) in enumerate(self.kfold.split(X_array), 1):
            print(f"\nFold {fold}/{self.n_splits}")
            print("-" * 50)
            
            X_train_fold = X_array[train_idx]
            X_val_fold = X_array[val_idx]
            y_train_fold = y_array[train_idx]
            y_val_fold = y_array[val_idx]
            
            model_copy = copy.deepcopy(model)
            model_copy.fit(X_train_fold, y_train_fold)
            
            y_pred_fold = model_copy.predict(X_val_fold)
            
            metrics = metrics_calculator.calculate_metrics(y_val_fold, y_pred_fold)
            fold_metrics.append(metrics)
            
            print(f"Fold {fold} Metrics:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        aggregated_metrics = self._aggregate_metrics(fold_metrics)
        
        return {
            'fold_metrics': fold_metrics,
            'mean_metrics': aggregated_metrics['mean'],
            'std_metrics': aggregated_metrics['std']
        }
    
    def _aggregate_metrics(self, fold_metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        metric_names = fold_metrics[0].keys()
        
        mean_metrics = {}
        std_metrics = {}
        
        for metric in metric_names:
            values = [fold[metric] for fold in fold_metrics]
            mean_metrics[metric] = np.mean(values)
            std_metrics[metric] = np.std(values)
        
        return {'mean': mean_metrics, 'std': std_metrics}
    
    def print_cv_summary(self, cv_results: Dict[str, Any]) -> None:
        print("\n" + "="*60)
        print(f"K-FOLD CROSS-VALIDATION SUMMARY (k={self.n_splits})")
        print("="*60)
        print(f"{'Metric':<20} {'Mean':<15} {'Std':<15}")
        print("-"*60)
        
        for metric in cv_results['mean_metrics'].keys():
            mean = cv_results['mean_metrics'][metric]
            std = cv_results['std_metrics'][metric]
            print(f"{metric:<20} {mean:<15.4f} {std:<15.4f}")
        
        print("="*60 + "\n")
