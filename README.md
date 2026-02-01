# Toxic Comment Classification with BiLSTM

A multi-label text classification system for identifying toxic comments in online discussions using deep learning. This project implements a Bidirectional LSTM neural network with robust 10-fold cross-validation for the Jigsaw Toxic Comment Classification Challenge.

## Project Overview

This is an implementation of the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge). The system predicts probabilities for six toxicity categories:

- **toxic**: General negativity or hostility
- **severe_toxic**: Highly aggressive or harmful content
- **obscene**: Offensive or vulgar language
- **threat**: Explicit threats or intentions of harm
- **insult**: Disrespectful or demeaning language
- **identity_hate**: Discrimination based on identity factors

## Key Features

- **BiLSTM Deep Learning Model**: Bidirectional LSTM with embedding layer for contextual text understanding
- **10-Fold Cross-Validation**: Stratified K-fold CV ensuring robust evaluation on imbalanced data
- **Advanced Text Preprocessing**: Contraction expansion, lemmatization, stopword removal with spaCy
- **Comprehensive Metrics**: F1-Score (0.6877), Precision (0.7783), Recall (0.6282), ROC-AUC (0.9658)
- **Design Patterns**: Abstract Factory and Strategy patterns for flexible data persistence
- **Web API**: FastAPI backend with live prediction capabilities
- **Academic Report**: Complete LaTeX documentation with visualizations

## Architecture Highlights

The project follows software engineering best practices:

- **Abstract Factory Pattern**: Seamless switching between file-based and in-memory data storage
- **Strategy Pattern**: Uniform model interface allowing different implementations
- **Separation of Concerns**: Modular structure with clear boundaries between layers
- **Dependency Injection**: Decoupled components for easy testing and extension

## Project Structure

```
ML4SE/
├── notebooks/              # Jupyter notebook for training and analysis
├── src/                    # Source code modules
│   ├── data/               # Data loading (file/memory) and preprocessing
│   ├── models/             # BiLSTM model implementation
│   ├── evaluation/         # Metrics and visualization
│   └── utils/              # Utility functions and persistence
├── server/                 # FastAPI web server
├── web/                    # Frontend HTML/CSS/JS
├── report/                 # LaTeX report and figures
├── models/                 # Trained BiLSTM model and tokenizer
└── data/                   # Dataset files (159,571 training samples)
```

## Installation

### Prerequisites

- Python 3.9+ (recommended) or Python 3.11
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download NLP Models

After installing dependencies, download required spaCy model:

```bash
python -m spacy download en_core_web_sm
```

### Download NLTK Data

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

## Usage

### Training the Model

Open and run the Jupyter notebook for full training pipeline:

```bash
jupyter lab notebooks/toxic_comment_classification.ipynb
```

The notebook performs:
1. Data loading with factory pattern
2. Text preprocessing and tokenization
3. 10-fold stratified cross-validation
4. Model training on full dataset
5. Performance evaluation and visualization

### Running the Web Server

Start the FastAPI server:

```bash
uvicorn server.main:app --reload
```

Access the API:
- Interactive docs: http://localhost:8000/docs
- Web interface: Open `web/index.html` in your browser

### API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Classify a comment
- `GET /examples` - Get example comments

## Model Performance

Results from 10-fold cross-validation on 159,571 samples:

| Metric | Mean | Std Dev |
|--------|------|---------|
| F1-Score (Macro) | 0.6877 | 0.0143 |
| Precision (Macro) | 0.7783 | 0.0236 |
| Recall (Macro) | 0.6282 | 0.0246 |
| ROC-AUC (Macro) | 0.9658 | 0.0016 |
| Accuracy | 0.9172 | 0.0030 |
| Hamming Loss | 0.0190 | 0.0007 |

The model demonstrates excellent consistency across folds with very low standard deviation, indicating robust generalization.

## Dataset

- **Training set**: 159,571 Wikipedia comments
- **Test set**: 153,164 comments
- **Class distribution**: Highly imbalanced (~90% clean comments)
- **Labels**: Multi-label binary classification (6 categories)

## Technical Stack

- **Deep Learning**: TensorFlow 2.13.1, Keras
- **NLP**: spaCy, NLTK
- **Data Processing**: pandas, NumPy, scikit-learn
- **Visualization**: matplotlib, seaborn
- **Web**: FastAPI, uvicorn
- **Documentation**: LaTeX (IEEE format)

## Design Patterns Implementation

### Factory Pattern for Data Loading

```python
from src.data.loader import get_data_loader

# File-based loading
loader = get_data_loader(mode='file', data_dir='../data')

# In-memory loading
loader = get_data_loader(mode='memory', train_df=df, test_df=test)
```

Both implementations share the same interface, allowing seamless switching without modifying business logic.

## Repository Structure

- `notebooks/toxic_comment_classification.ipynb` - Main training pipeline
- `src/data/loader.py` - Abstract Factory for data persistence
- `src/models/lstm_model.py` - BiLSTM implementation
- `server/main.py` - FastAPI application
- `report/main.tex` - Academic report (LaTeX)

## Author

**Bryant Michelle Sarabia Ortega**  
University of L'Aquila - ML4SE Course  
Email: bryantmichelle.sarabiaortega@student.univaq.it

## Academic Context

This project fulfills the requirements of the ML4SE (Machine Learning for Software Engineering) course, demonstrating:
- Application of design patterns for flexible software architecture
- Implementation of deep learning for NLP tasks
- Rigorous evaluation methodology with cross-validation
- Professional documentation and reporting

## License

This project is for educational purposes as part of university coursework.
