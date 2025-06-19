# Semantic Similarity CLI

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A command-line tool for detecting semantic similarity between sentences.

## Features
- Single sentence pair comparison
- Batch CSV processing
- Confidence scoring
- Docker support

## Installation
```bash
git clone https://github.com/YOUR-USERNAME/semantic-similarity-cli.git
cd semantic-similarity-cli
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

## Usage
### Single Pair
```bash
python semantic_similarity.py -s1 "First sentence" -s2 "Second sentence"
```

### Batch Processing
```bash
python semantic_similarity.py -f input.csv -o results.csv
```

## Docker Support
```bash
docker build -t similarity-cli .
docker run similarity-cli -s1 "Hello" -s2 "Hi"
```

## Model Information
- Sentence Transformer: `all-MiniLM-L6-v2`
- Classifier: XGBoost
- Trained on Quora Question Pairs dataset