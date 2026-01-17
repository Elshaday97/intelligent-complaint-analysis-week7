# Intelligent Complaint Analysis — Week 7

## Project overview

An NLP pipeline to ingest, preprocess, analyze, and classify customer complaints. Goals: clean and label complaint data, train models to detect categories/urgency, evaluate performance, and produce reproducible reports and artifacts.

## Project structure

- README.md — this file
- data/ — raw and processed datasets (raw/, processed/)
- notebooks/ — exploratory analysis and experiments (.ipynb)
- src/ — main code (data preprocessing, loading)
- test/ — Unit tests for src/
- requirements.txt / environment.yml — dependency manifests
- scripts/ — runnable scripts (train, evaluate, predict)
- .gitignore

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

Notes

- Place raw data in data/raw/ and update paths in config files before running pipeline.
- Use version control for models and config files; store large artifacts outside Git or in LFS.
