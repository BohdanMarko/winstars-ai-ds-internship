# Task 2 - Animal NER + Image Classification Pipeline

ML pipeline that combines Named Entity Recognition (NER) and image classification to verify whether a text description matches an animal in an image. Takes two inputs - a text message and an image - and returns a boolean: `True` if the text mentions the same animal shown in the image, `False` otherwise.

## Models

| Component | Model | Purpose |
|-----------|-------|---------|
| NER | `dslim/bert-base-NER` (fine-tuned) | Extract animal names from text |
| Image Classifier | ResNet-50 (fine-tuned) | Classify animal in image |

## Dataset

**Animals-10** (Kaggle) - ~28k images across 10 classes:
butterfly, cat, chicken, cow, dog, elephant, horse, sheep, spider, squirrel

## Project Structure

## Requirements

- Python 3.10
- Dependencies listed in `requirements.txt`

## Setup

The virtual environment is shared across both tasks and lives at the repo root.

**Activate:**
```bash
venv\Scripts\activate
```

**Install dependencies:**
```bash
pip install -r task2/requirements.txt
```

**Register the venv as a Jupyter kernel (ignore if done during task 1):**
```bash
pip install ipykernel
python -m ipykernel install --user --name=venv --display-name "Python 3.10 (venv)"
```
