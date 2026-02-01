"""
Standalone script to train models and generate results for the report.
This script trains all three models and saves results.
"""
import sys
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("Toxic Comment Classification - Model Training and Evaluation")
print("=" * 80)

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.loader import FileBasedDataLoader
from src.data.preprocessor import TextPreprocessor
from src.models.ml_baselines import TfidfLogisticRegression, TfidfNaiveBayes
from src.models.lstm_model import BiLSTMModel
from src.evaluation.metrics import MetricsCalculator
from src.utils.persistence import ModelPersistence

sns.set_style('whitegrid')

# 1. Load Data
print("\n1. Loading Data...")
loader = FileBasedDataLoader(data_dir='data')
X_train_full, y_train_full = loader.load_train_data()
print(f"Total training samples: {len(X_train_full)}")

# Use a reasonable subset for training (to save time while still getting good results)
TRAIN_SIZE = 80000
TEST_SIZE = 10000

print(f"Using {TRAIN_SIZE} samples for training and {TEST_SIZE} for testing")

X_train = X_train_full['comment_text'].head(TRAIN_SIZE)
y_train = y_train_full.head(TRAIN_SIZE)
X_test = X_train_full['comment_text'].iloc[TRAIN_SIZE:TRAIN_SIZE+TEST_SIZE]
y_test = y_train_full.iloc[TRAIN_SIZE:TRAIN_SIZE+TEST_SIZE]

# 2. Train TF-IDF + Logistic Regression
print("\n2. Training TF-IDF + Logistic Regression...")
lr_model = TfidfLogisticRegression(max_features=10000)
lr_model.fit(X_train, y_train)
print("   Model trained!")

# Evaluate
print("   Evaluating...")
lr_pred = lr_model.predict(X_test)
evaluator = MetricsCalculator()
lr_metrics = evaluator.calculate_metrics(y_test.values, lr_pred)

print(f"   Hamming Loss: {lr_metrics['hamming_loss']:.4f}")
print(f"   F1 Score: {lr_metrics['f1_score']:.4f}")
print(f"   Precision: {lr_metrics['precision']:.4f}")
print(f"   Recall: {lr_metrics['recall']:.4f}")

# 3. Train TF-IDF + Naive Bayes
print("\n3. Training TF-IDF + Naive Bayes...")
nb_model = TfidfNaiveBayes(max_features=10000)
nb_model.fit(X_train, y_train)
print("   Model trained!")

# Evaluate
print("   Evaluating...")
nb_pred = nb_model.predict(X_test)
nb_metrics = evaluator.calculate_metrics(y_test.values, nb_pred)

print(f"   Hamming Loss: {nb_metrics['hamming_loss']:.4f}")
print(f"   F1 Score: {nb_metrics['f1_score']:.4f}")
print(f"   Precision: {nb_metrics['precision']:.4f}")
print(f"   Recall: {nb_metrics['recall']:.4f}")

# 4. Train BiLSTM (smaller subset due to computational cost)
print("\n4. Training BiLSTM Neural Network...")
print("   Note: Using smaller dataset for demo purposes")

LSTM_TRAIN_SIZE = 20000
LSTM_TEST_SIZE = 2000

X_lstm_train = X_train_full['comment_text'].head(LSTM_TRAIN_SIZE)
y_lstm_train = y_train_full.head(LSTM_TRAIN_SIZE)
X_lstm_test = X_train_full['comment_text'].iloc[LSTM_TRAIN_SIZE:LSTM_TRAIN_SIZE+LSTM_TEST_SIZE]
y_lstm_test = y_train_full.iloc[LSTM_TRAIN_SIZE:LSTM_TRAIN_SIZE+LSTM_TEST_SIZE]

lstm_model = BiLSTMModel(
    max_features=20000,
    max_len=150,
    embedding_dim=64,
    lstm_units=64
)

lstm_model.fit(X_lstm_train, y_lstm_train, epochs=3, batch_size=128)
print("   Model trained!")

# Evaluate
print("   Evaluating...")
lstm_pred = lstm_model.predict(X_lstm_test)
lstm_metrics = evaluator.calculate_metrics(y_lstm_test.values, lstm_pred)

print(f"   Hamming Loss: {lstm_metrics['hamming_loss']:.4f}")
print(f"   F1 Score: {lstm_metrics['f1_score']:.4f}")
print(f"   Precision: {lstm_metrics['precision']:.4f}")
print(f"   Recall: {lstm_metrics['recall']:.4f}")

# 5. Create comparison visualization
print("\n5. Creating Model Comparison Visualization...")

comparison_data = {
    'Model': ['Logistic Regression', 'Naive Bayes', 'BiLSTM'],
    'Hamming Loss': [lr_metrics['hamming_loss'], nb_metrics['hamming_loss'], lstm_metrics['hamming_loss']],
    'F1 Score': [lr_metrics['f1_score'], nb_metrics['f1_score'], lstm_metrics['f1_score']],
    'Precision': [lr_metrics['precision'], nb_metrics['precision'], lstm_metrics['precision']],
    'Recall': [lr_metrics['recall'], nb_metrics['recall'], lstm_metrics['recall']]
}

comparison_df = pd.DataFrame(comparison_data)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# F1 Scores
comparison_df.plot(x='Model', y=['F1 Score', 'Precision', 'Recall'], kind='bar', ax=axes[0], rot=0)
axes[0].set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Score')
axes[0].set_ylim([0, 1])
axes[0].legend(loc='lower right')
axes[0].grid(axis='y', alpha=0.3)

# Hamming Loss
comparison_df.plot(x='Model', y='Hamming Loss', kind='bar', ax=axes[1], rot=0, color='coral', legend=False)
axes[1].set_title('Hamming Loss Comparison (lower is better)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Hamming Loss')
axes[1].set_ylim([0, max(comparison_df['Hamming Loss']) * 1.2])
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('report/figures/model_comparison.png', dpi=300, bbox_inches='tight')
print("   Saved to report/figures/model_comparison.png")

# 6. Save models
print("\n6. Saving Models...")

# Create models directory
ModelPersistence.create_model_directory('models')

# Save Logistic Regression
ModelPersistence.save_model_with_metadata(lr_model, 'models/tfidf_logistic_regression.pkl')
print("   Saved Logistic Regression model")

# Save Naive Bayes
ModelPersistence.save_model_with_metadata(nb_model, 'models/tfidf_naive_bayes.pkl')
print("   Saved Naive Bayes model")

# Save BiLSTM
try:
    ModelPersistence.save_model_with_metadata(lstm_model, 'models/bilstm_model.keras')
    print("   Saved BiLSTM model")
except Exception as e:
    print(f"   Warning: Could not save BiLSTM model: {e}")

# 7. Save results to file for report
print("\n7. Saving Results Summary...")
results_summary = f"""# Model Training Results

## Dataset
- Total samples: {len(X_train_full)}
- Training samples (classical ML): {TRAIN_SIZE}
- Test samples (classical ML): {TEST_SIZE}
- Training samples (BiLSTM): {LSTM_TRAIN_SIZE}
- Test samples (BiLSTM): {LSTM_TEST_SIZE}

## Model Performance

### TF-IDF + Logistic Regression
- Hamming Loss: {lr_metrics['hamming_loss']:.4f}
- F1 Score: {lr_metrics['f1_score']:.4f}
- Precision: {lr_metrics['precision']:.4f}
- Recall: {lr_metrics['recall']:.4f}

### TF-IDF + Naive Bayes
- Hamming Loss: {nb_metrics['hamming_loss']:.4f}
- F1 Score: {nb_metrics['f1_score']:.4f}
- Precision: {nb_metrics['precision']:.4f}
- Recall: {nb_metrics['recall']:.4f}

### BiLSTM Neural Network
- Hamming Loss: {lstm_metrics['hamming_loss']:.4f}
- F1 Score: {lstm_metrics['f1_score']:.4f}
- Precision: {lstm_metrics['precision']:.4f}
- Recall: {lstm_metrics['recall']:.4f}

## Comparison Table

{comparison_df.to_string(index=False)}
"""

with open('report/results_summary.txt', 'w') as f:
    f.write(results_summary)

print("   Saved to report/results_summary.txt")

print("\n" + "=" * 80)
print("Training Complete!")
print("=" * 80)
print("\nAll models trained, evaluated, and saved successfully.")
print("Results and visualizations are available in the 'report' folder.")
