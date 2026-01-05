# Toxic Comment Classification Challenge

A multi-label text classification system for identifying toxic comments in online discussions. This project implements both traditional machine learning and deep learning approaches to classify comments into six toxicity categories.

## Project Overview

This is an implementation of the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) from Kaggle. The goal is to predict probabilities for the following toxicity types:

- **toxic**: General negativity or hostility
- **severe_toxic**: Highly aggressive or harmful content
- **obscene**: Offensive or vulgar language
- **threat**: Explicit threats or intentions of harm
- **insult**: Disrespectful or demeaning language
- **identity_hate**: Discrimination based on identity factors

## Features

- **Multiple Model Approaches**: TF-IDF + Logistic Regression, TF-IDF + Naive Bayes, BiLSTM Neural Network
- **Advanced Text Preprocessing**: Contraction expansion, lemmatization, stopword removal
- **K-Fold Cross-Validation**: Robust model evaluation with stratified 10-fold CV
- **Comprehensive Metrics**: Precision, Recall, F1-Score, Hamming Loss, ROC-AUC
- **Rich Visualizations**: Confusion matrices, ROC curves, learning curves
- **Web Interface**: FastAPI backend with vanilla HTML/CSS/JS frontend for live predictions
- **Modular Architecture**: Separation of concerns with abstract interfaces for persistence

## Project Structure

```
ML4SE/
├── notebooks/              # Jupyter notebooks for analysis
├── src/                    # Source code modules
│   ├── data/               # Data loading and preprocessing
│   ├── models/             # Model implementations
│   ├── evaluation/         # Metrics and visualization
│   └── utils/              # Utility functions
├── server/                 # FastAPI web server
├── web/                    # Frontend HTML/CSS/JS
├── report/                 # LaTeX report and figures
├── models/                 # Saved trained models
└── data/                   # Dataset files
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training Models

Open and run the Jupyter notebook:

```bash
jupyter lab notebooks/toxic_comment_classification.ipynb
```

### Running the Web Server

```bash
uvicorn server.main:app --reload
```

Then open `web/index.html` in your browser.

## Dataset

The dataset consists of Wikipedia comments labeled by human raters:
- Training set: 159,571 comments
- Test set: 153,164 comments
- Highly imbalanced: ~90% clean comments

## Requirements

- Python 3.9+
- See `requirements.txt` for full dependency list

## Author

University Project - ML4SE Course

## License

This project is for educational purposes.
