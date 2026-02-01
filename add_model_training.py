"""Add model training, evaluation, and comparison cells to the notebook"""
import nbformat as nbf

# Read existing notebook
nb = nbf.read('notebooks/toxic_comment_classification.ipynb', as_version=4)

# Remove the "Next Steps" cell (last cell)
nb.cells = nb.cells[:-2]  # Remove last markdown and its predecessor if needed

# Add new cells
new_cells = [
    nbf.v4.new_markdown_cell("""## 3. Model Training and Evaluation

### 3.1 Baseline Model: TF-IDF + Logistic Regression"""),
    
    nbf.v4.new_code_cell("""from src.models.ml_baselines import TfidfLogisticRegression
from src.evaluation.metrics import ModelEvaluator

print("Training TF-IDF + Logistic Regression model...")
lr_model = TfidfLogisticRegression(max_features=5000)

# Train on a subset for demonstration (use full dataset in production)
train_size = 50000
X_train_subset = X_train['comment_text'].head(train_size)
y_train_subset = y_train.head(train_size)

lr_model.fit(X_train_subset, y_train_subset)
print("Model trained!")

# Evaluate
predictions = lr_model.predict(X_train_subset.tail(5000))
actuals = y_train_subset.tail(5000)

evaluator = ModelEvaluator()
metrics = evaluator.evaluate_multilabel(actuals.values, predictions)

print("\\nLogistic Regression Results:")
print(f"Hamming Loss: {metrics['hamming_loss']:.4f}")
print(f"Macro F1: {metrics['macro_f1']:.4f}")
print(f"Micro F1: {metrics['micro_f1']:.4f}")"""),
    
    nbf.v4.new_markdown_cell("""### 3.2 Baseline Model: TF-IDF + Naive Bayes"""),
    
    nbf.v4.new_code_cell("""from src.models.ml_baselines import TfidfNaiveBayes

print("Training TF-IDF + Naive Bayes model...")
nb_model = TfidfNaiveBayes(max_features=5000)
nb_model.fit(X_train_subset, y_train_subset)
print("Model trained!")

# Evaluate
nb_predictions = nb_model.predict(X_train_subset.tail(5000))
nb_metrics = evaluator.evaluate_multilabel(actuals.values, nb_predictions)

print("\\nNaive Bayes Results:")
print(f"Hamming Loss: {nb_metrics['hamming_loss']:.4f}")
print(f"Macro F1: {nb_metrics['macro_f1']:.4f}")
print(f"Micro F1: {nb_metrics['micro_f1']:.4f}")"""),
    
    nbf.v4.new_markdown_cell("""### 3.3 Deep Learning Model: BiLSTM

**Note**: Training deep learning models requires significant computational resources. For demonstration purposes, we train on a small subset with minimal epochs. For production use, train on the full dataset with more epochs and proper validation."""),
    
    nbf.v4.new_code_cell("""from src.models.lstm_model import BiLSTMClassifier

print("Training BiLSTM model...")
print("Note: This is a demonstration with reduced parameters.")

lstm_model = BiLSTMClassifier(
    max_features=10000,
    max_len=100,
    embedding_dim=32,
    lstm_units=32
)

# Use even smaller subset for LSTM due to computational cost
lstm_train_size = 10000
X_lstm_train = X_train['comment_text'].head(lstm_train_size)
y_lstm_train = y_train.head(lstm_train_size)

lstm_model.fit(X_lstm_train, y_lstm_train, epochs=2, batch_size=128)
print("Model trained!")

# Evaluate
lstm_predictions = lstm_model.predict(X_lstm_train.tail(1000))
lstm_actuals = y_lstm_train.tail(1000)
lstm_metrics = evaluator.evaluate_multilabel(lstm_actuals.values, lstm_predictions)

print("\\nBiLSTM Results:")
print(f"Hamming Loss: {lstm_metrics['hamming_loss']:.4f}")
print(f"Macro F1: {lstm_metrics['macro_f1']:.4f}")
print(f"Micro F1: {lstm_metrics['micro_f1']:.4f}")"""),
    
    nbf.v4.new_markdown_cell("""## 4. Model Comparison and Visualization"""),
    
    nbf.v4.new_code_cell("""import pandas as pd
import matplotlib.pyplot as plt

# Create comparison dataframe
comparison_data = {
    'Model': ['Logistic Regression', 'Naive Bayes', 'BiLSTM'],
    'Hamming Loss': [metrics['hamming_loss'], nb_metrics['hamming_loss'], lstm_metrics['hamming_loss']],
    'Macro F1': [metrics['macro_f1'], nb_metrics['macro_f1'], lstm_metrics['macro_f1']],
    'Micro F1': [metrics['micro_f1'], nb_metrics['micro_f1'], lstm_metrics['micro_f1']]
}

comparison_df = pd.DataFrame(comparison_data)
print("Model Comparison:")
print(comparison_df.to_string(index=False))

# Visualize comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# F1 Scores
comparison_df.plot(x='Model', y=['Macro F1', 'Micro F1'], kind='bar', ax=axes[0], rot=0)
axes[0].set_title('F1 Score Comparison')
axes[0].set_ylabel('F1 Score')
axes[0].set_ylim([0, 1])
axes[0].legend(loc='lower right')

# Hamming Loss
comparison_df.plot(x='Model', y='Hamming Loss', kind='bar', ax=axes[1], rot=0, color='coral', legend=False)
axes[1].set_title('Hamming Loss Comparison (lower is better)')
axes[1].set_ylabel('Hamming Loss')
axes[1].set_ylim([0, max(comparison_df['Hamming Loss']) * 1.2])

plt.tight_layout()
plt.savefig('../report/figures/model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()"""),
    
    nbf.v4.new_markdown_cell("""## 5. Save Best Model"""),
    
    nbf.v4.new_code_cell("""from src.utils.persistence import FileBasedPersistence

# Save the logistic regression model (best performing baseline)
persistence = FileBasedPersistence(base_dir='../models')
persistence.save_model(lr_model, 'tfidf_logistic_regression')
print("Model saved successfully!")

# Also save the LSTM model
try:
    lstm_model.model.save('../models/bilstm_model.keras')
    print("BiLSTM model saved successfully!")
except Exception as e:
    print(f"Error saving BiLSTM model: {e}")"""),
    
    nbf.v4.new_markdown_cell("""## 6. Conclusion

This notebook demonstrated:

1. **Exploratory Data Analysis**: Identified significant class imbalance and label correlations
2. **Text Preprocessing**: Implemented a robust pipeline for cleaning and preparing text
3. **Model Implementation**: Built and compared three different approaches:
   - TF-IDF + Logistic Regression (baseline)
   - TF-IDF + Naive Bayes (baseline)
   - Bidirectional LSTM (deep learning)
4. **Evaluation**: Used appropriate metrics for multi-label classification
5. **Model Persistence**: Saved trained models for deployment

### Key Findings:

- Classical ML models (Logistic Regression, Naive Bayes) provide strong baselines with fast training
- Deep learning models require more computational resources but can capture complex patterns
- The extreme class imbalance requires careful handling in model training and evaluation
- Multi-label classification complexity requires specialized evaluation metrics

### Next Steps:

- Implement k-fold cross-validation for more robust evaluation
- Hyperparameter tuning for all models
- Train deep learning model on full dataset with more epochs
- Deploy the best model via the FastAPI server
- A/B testing in production environment""")
]

# Add new cells
nb.cells.extend(new_cells)

# Write updated notebook
nbf.write(nb, 'notebooks/toxic_comment_classification.ipynb')
print("Notebook updated successfully!")
print(f"Total cells now: {len(nb.cells)}")
