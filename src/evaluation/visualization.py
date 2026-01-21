import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc
from typing import Dict, List


class Visualizer:
    """Visualization utilities for model evaluation."""
    
    def __init__(self, label_columns: List[str] = None):
        if label_columns is None:
            self.label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        else:
            self.label_columns = label_columns
        
        sns.set_style('whitegrid')
        self.colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    def plot_confusion_matrices(self, confusion_matrices: Dict[str, np.ndarray], 
                                save_path: str = None) -> None:
        """Plot confusion matrices for all labels."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (label, cm) in enumerate(confusion_matrices.items()):
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'])
            axes[idx].set_title(f'{label.capitalize()} - Confusion Matrix')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_metrics_comparison(self, model_metrics: Dict[str, Dict[str, float]], 
                                save_path: str = None) -> None:
        """Compare metrics across multiple models."""
        metrics_to_plot = ['precision', 'recall', 'f1_score', 'roc_auc']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_to_plot):
            models = list(model_metrics.keys())
            values = [model_metrics[model][metric] for model in models]
            
            axes[idx].bar(models, values, color=self.colors[:len(models)])
            axes[idx].set_title(f'{metric.replace("_", " ").title()}')
            axes[idx].set_ylabel('Score')
            axes[idx].set_ylim(0, 1.0)
            axes[idx].tick_params(axis='x', rotation=45)
            
            for i, v in enumerate(values):
                axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_per_label_metrics(self, per_label_metrics: Dict[str, Dict[str, float]],
                               save_path: str = None) -> None:
        """Plot metrics for each label."""
        labels = list(per_label_metrics.keys())
        metrics = ['precision', 'recall', 'f1_score']
        
        x = np.arange(len(labels))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, metric in enumerate(metrics):
            values = [per_label_metrics[label][metric] for label in labels]
            ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title(),
                   color=self.colors[i])
        
        ax.set_xlabel('Labels')
        ax.set_ylabel('Score')
        ax.set_title('Per-Label Metrics Comparison')
        ax.set_xticks(x + width)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_training_history(self, history, save_path: str = None) -> None:
        """Plot training history for deep learning models."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(history.history['loss'], label='Training Loss', color='#e74c3c')
        if 'val_loss' in history.history:
            axes[0].plot(history.history['val_loss'], label='Validation Loss', color='#3498db')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        axes[1].plot(history.history['accuracy'], label='Training Accuracy', color='#2ecc71')
        if 'val_accuracy' in history.history:
            axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', color='#f39c12')
        axes[1].set_title('Model Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_roc_curves(self, y_true, y_pred, save_path: str = None) -> None:
        """Plot ROC curves for all labels."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, label in enumerate(self.label_columns):
            fpr, tpr, _ = roc_curve(y_true[:, idx], y_pred[:, idx])
            roc_auc = auc(fpr, tpr)
            
            axes[idx].plot(fpr, tpr, color=self.colors[idx], lw=2,
                          label=f'ROC curve (AUC = {roc_auc:.2f})')
            axes[idx].plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
            axes[idx].set_xlim([0.0, 1.0])
            axes[idx].set_ylim([0.0, 1.05])
            axes[idx].set_xlabel('False Positive Rate')
            axes[idx].set_ylabel('True Positive Rate')
            axes[idx].set_title(f'{label.capitalize()} - ROC Curve')
            axes[idx].legend(loc="lower right")
            axes[idx].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
