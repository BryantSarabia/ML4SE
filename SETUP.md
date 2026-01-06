# Toxic Comment Classification - Setup Instructions

## Installing Dependencies

```bash
pip install -r requirements.txt
```

## Download Required NLP Models

After installing dependencies, download the spacy English model:

```bash
python -m spacy download en_core_web_sm
```

## Download NLTK Data

Run Python and execute:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```
