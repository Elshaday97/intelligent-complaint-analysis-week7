# Intelligent Complaint Analysis — Week 7

## Project overview

An NLP pipeline to ingest, preprocess, analyze, and classify customer complaints. Goals: clean and label complaint data, train models to detect categories/urgency, evaluate performance, and produce reproducible reports and artifacts.

## Project structure — detailed

- README.md — this file
- data/
  - raw/ — original input files (CSV, JSON). Do not edit.
  - processed/ — cleaned, split datasets used for modeling
- notebooks/
  - complaints_eda.ipynb — initial EDA on customer complaints
  - rag_evaluation.ipynb — Evaluation of RAG model on user input
  - vectorization.ipynb — Vectorize customer complaints
- src/
  - **init**.py
  - data/
    - loader.py — dataset loading and saving
    - preprocess.py — text cleaning, normalization, tokenization
  - rag_system.py — complete RAG system implementation from user input to llm response
  - text_preprocessor.py — chunk texts and prepare metadata for vectorization
  - vector_manager.py — creates and stores vector embeddings using FAISS
- scripts/
  - constants.py — shared constants (e.g., Column names)
  - prepare_parquet.py - load parquet to data frame then vectorize
  - utils.py — utility functions to clean and normalize text data
- test/
  - test_data_loader.py - unit tests for data loading/saving
- vector_store/ — saved vector embeddings and indexes
- requirements.txt / environment.yml — dependency manifests
- .github/
  - workflows/ci.yml — CI for tests and linting
- .gitignore
- app.py - Streamlit app to interact with RAG system

## Setup guide

Prerequisites

- Python 3.8+ or Conda
- Git

Clone

```
git clone git@github.com:Elshaday97/intelligent-complaint-analysis-week7.git
cd intelligent-complaint-analysis-week7
```

Virtual environment (pip)

```
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Common commands

- Run notebooks:

```
jupyter lab
```

- Run streamlit app to interact with RAG system:

```
streamlit run app.py
```

Notes

- Place raw data in data/raw/ and update paths in config files before running pipeline.
- Use version control for models and config files; store large artifacts outside Git or in LFS.

Notes and best practices

- Keep raw data immutable in data/raw/.
- Version models and large artifacts with DVC or Git LFS if needed.
- Add new experiments as notebooks or scripts and record config changes for reproducibility.
