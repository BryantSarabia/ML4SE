"""
Toxic Comment Classification - BiLSTM with K-Fold Cross-Validation
Training script following deep learning best practices for multi-label classification.
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold, train_test_split

from src.data.loader import FileBasedDataLoader
from src.models.lstm_model import BiLSTMModel
from src.evaluation.metrics import MetricsCalculator
from src.utils.persistence import ModelPersistence

# Configuration (matching example project)
MAX_FEATURES = 3000      # vocabulary size
MAX_SEQUENCE = 100       # sequence length
EMBEDDING_DIM = 32       # embedding dimension
LSTM_UNITS = 32          # LSTM units per direction
NUM_EPOCHS = 5           # epochs per fold
BATCH_SIZE = 32          # batch size
NUM_FOLDS = 10           # k-fold splits

print("=" * 80)
print("Toxic Comment Classification - BiLSTM with 10-Fold Cross-Validation")
print("=" * 80)

# Check GPU availability
gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
if gpu_available:
    gpu_name = tf.config.list_physical_devices('GPU')[0].name
    print(f"\nGPU Available: {gpu_name}")
else:
    print("\nGPU Available: False (training will use CPU)")

# Load full dataset
print("\nLoading dataset...")
loader = FileBasedDataLoader(data_dir='data')
X_full, y_full = loader.load_train_data()

print(f"Dataset loaded: {len(X_full):,} samples")
print(f"Labels: {list(y_full.columns)}")

# Display configuration
print(f"\nModel Configuration:")
print(f"  Architecture: Bidirectional LSTM")
print(f"  Vocabulary size: {MAX_FEATURES:,} tokens")
print(f"  Sequence length: {MAX_SEQUENCE}")
print(f"  Embedding dimension: {EMBEDDING_DIM}")
print(f"  LSTM units: {LSTM_UNITS} per direction ({LSTM_UNITS*2} total)")
print(f"  Dense layers: 2 × 64 units with ReLU + Dropout(0.3)")
print(f"  Output layer: 6 sigmoid units (multi-label)")

print(f"\nTraining Configuration:")
print(f"  K-Fold splits: {NUM_FOLDS}")
print(f"  Epochs per fold: {NUM_EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Loss: Binary cross-entropy")
print(f"  Optimizer: Adam")

# K-Fold Cross-Validation
print("\n" + "=" * 80)
print("Starting 10-Fold Cross-Validation")
print("=" * 80)

kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
evaluator = MetricsCalculator()
fold_metrics = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_full), 1):
    print(f"\n{'='*80}")
    print(f"FOLD {fold}/{NUM_FOLDS}")
    print(f"{'='*80}")
    
    # Split data for this fold
    X_train_fold = X_full.iloc[train_idx]['comment_text']
    y_train_fold = y_full.iloc[train_idx]
    X_val_fold = X_full.iloc[val_idx]['comment_text']
    y_val_fold = y_full.iloc[val_idx]
    
    print(f"Training samples: {len(X_train_fold):,}")
    print(f"Validation samples: {len(X_val_fold):,}")
    
    # Create new model for this fold
    model = BiLSTMModel(
        max_features=MAX_FEATURES,
        max_len=MAX_SEQUENCE,
        embedding_dim=EMBEDDING_DIM,
        lstm_units=LSTM_UNITS
    )
    
    # Train on fold
    print(f"\nTraining fold {fold}...")
    model.fit(
        X_train_fold, 
        y_train_fold,
        X_val=X_val_fold,
        y_val=y_val_fold,
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE
    )
    
    # Evaluate on validation set
    print(f"\nEvaluating fold {fold}...")
    y_pred = model.predict(X_val_fold)
    metrics = evaluator.calculate_metrics(y_val_fold.values, y_pred)
    fold_metrics.append(metrics)
    
    print(f"\nFold {fold} Results:")
    print(f"  Hamming Loss: {metrics['hamming_loss']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")

# Aggregate results across folds
print("\n" + "=" * 80)
print("FINAL RESULTS - 10-Fold Cross-Validation")
print("=" * 80)

mean_metrics = {}
std_metrics = {}
for metric in fold_metrics[0].keys():
    values = [fold[metric] for fold in fold_metrics]
    mean_metrics[metric] = np.mean(values)
    std_metrics[metric] = np.std(values)

print("\nMean ± Standard Deviation:")
for metric in mean_metrics.keys():
    print(f"  {metric:20s}: {mean_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")

# Train final model on full dataset for deployment
print("\n" + "=" * 80)
print("Training Final Model on Full Dataset")
print("=" * 80)

final_model = BiLSTMModel(
    max_features=MAX_FEATURES,
    max_len=MAX_SEQUENCE,
    embedding_dim=EMBEDDING_DIM,
    lstm_units=LSTM_UNITS
)

# Use 80/20 split for final model
X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
    X_full['comment_text'], y_full, test_size=0.2, random_state=42
)

print(f"Training final model on {len(X_train_final):,} samples...")
print(f"Validation on {len(X_val_final):,} samples...")

final_model.fit(
    X_train_final,
    y_train_final,
    X_val=X_val_final,
    y_val=y_val_final,
    epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE
)

# Save final model
print("\nSaving final model...")
ModelPersistence.create_model_directory('models')
ModelPersistence.save_model_with_metadata(final_model, 'models/bilstm_toxic_classifier.keras')
print("  Model saved to: models/bilstm_toxic_classifier.keras")

# Save results summary
results_summary = f"""# Toxic Comment Classification - BiLSTM Results

## Configuration
- Dataset: {len(X_full):,} samples (full Jigsaw Toxic Comment dataset)
- Model: Bidirectional LSTM
- Vocabulary: {MAX_FEATURES:,} tokens
- Sequence length: {MAX_SEQUENCE}
- Architecture: Embedding({EMBEDDING_DIM}) → BiLSTM({LSTM_UNITS}) → Dense(64×2) → Output(6)
- Validation: 10-fold cross-validation
- Epochs per fold: {NUM_EPOCHS}
- Batch size: {BATCH_SIZE}
- Hardware: {'GPU' if gpu_available else 'CPU'}

## Cross-Validation Results (Mean ± Std)

- Hamming Loss: {mean_metrics['hamming_loss']:.4f} ± {std_metrics['hamming_loss']:.4f}
- F1 Score: {mean_metrics['f1_score']:.4f} ± {std_metrics['f1_score']:.4f}
- Precision: {mean_metrics['precision']:.4f} ± {std_metrics['precision']:.4f}
- Recall: {mean_metrics['recall']:.4f} ± {std_metrics['recall']:.4f}
- Accuracy: {mean_metrics['accuracy']:.4f} ± {std_metrics['accuracy']:.4f}

## Per-Fold Results

"""

for i, fold in enumerate(fold_metrics, 1):
    results_summary += f"Fold {i:2d}: F1={fold['f1_score']:.4f}, Hamming Loss={fold['hamming_loss']:.4f}, Accuracy={fold['accuracy']:.4f}\n"

with open('report/results_summary.txt', 'w') as f:
    f.write(results_summary)

print("\nResults summary saved to: report/results_summary.txt")

print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
print(f"\nFinal Performance (10-Fold CV):")
print(f"  F1 Score: {mean_metrics['f1_score']:.4f} ± {std_metrics['f1_score']:.4f}")
print(f"  Hamming Loss: {mean_metrics['hamming_loss']:.4f} ± {std_metrics['hamming_loss']:.4f}")
print("\nNext steps:")
print("  1. Check report/results_summary.txt for detailed results")
print("  2. Update report/main.tex with the metrics above")
print("  3. Run 'cd report && make' to generate PDF")
