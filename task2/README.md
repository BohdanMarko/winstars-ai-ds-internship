# Task 2 - Animal NER + Image Classification Pipeline

ML pipeline that combines Named Entity Recognition (NER) and image classification to verify whether a text description matches an animal in an image. Takes two inputs - a text message and an image - and returns a boolean: `True` if the text mentions the same animal shown in the image, `False` otherwise.

## Models

| Component | Model | Type | Purpose |
|-----------|-------|------|---------|
| NER | `dslim/bert-base-NER` (fine-tuned) | Transformer Encoder (BERT) | Extract animal names from text |
| Image Classifier | ResNet-50 (fine-tuned) | CNN | Classify animal in image |

Both models are **not generative** — they classify, not generate content.

## Dataset

**Animals-10** (Kaggle) - ~28k images across 10 classes:
butterfly, cat, chicken, cow, dog, elephant, horse, sheep, spider, squirrel

## How It Works

1. `ner/dataset.py` - generates synthetic training data
2. `ner/train.py` - fine-tunes BERT on that data
3. `ner/inference.py` - extracts animal name from user text
4. `image_classifier/train.py` - fine-tunes ResNet-50 on Animals-10
5. `image_classifier/inference.py` - classifies animal in the image
6. `pipeline.py` - compares NER output with classifier output -> `True` / `False`

### NER Dataset (`ner/dataset.py`)

Generates a synthetic BIO-tagged dataset for training the NER model. Since no
ready-made dataset of animal-annotated sentences exists, we create one from
sentence templates combined with animal names and their synonyms.

Each word in a sentence gets a BIO tag:
- **B-ANIMAL** - first word of an animal entity
- **I-ANIMAL** - continuation of the entity (for multi-word names)
- **O** - not an entity

Example: `"I can see a cat."` → `[O, O, O, O, B-ANIMAL]`

The dataset is generated in memory during training (not saved to disk). A fixed
random seed ensures reproducibility. The data is split into two parts:
- **Train** (~85%) - used to teach the model
- **Val** (validation, ~15%) - held-out data the model never trains on, used to
  measure how well it generalizes to unseen examples

### Image Classifier (`image_classifier/`)

Fine-tunes a pre-trained ResNet-50 (ImageNet) on Animals-10 using transfer learning
with a two-phase approach:

- **Phase 1 (epochs 1-5)**: Backbone frozen, only the classification head trains.
  The head learns to map ResNet's visual features to our 10 animal classes.
- **Phase 2 (epochs 6-20)**: Full fine-tuning with differential learning rates —
  backbone gets 10x lower lr (`1e-4`) to gently adjust, head continues at `1e-3`.

The dataset is downloaded automatically via `kagglehub`. Italian folder names are
translated to English. Images are split into train/val (85%/15%) with stratified
sampling. Training augmentations (random crop, flip, color jitter) help prevent
overfitting.

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

## Known Limitations

**NER false positives on rare words.** The NER model is trained on synthetic
templates, so it may occasionally tag non-animal words as entities in unusual
contexts ("savanna" in *"A majestic elephant roaming the savanna"*). The
pipeline filters these out via `KNOWN_CLASSES`, so they don't affect the final
result, but NER precision could be improved.
*Fix:* Add more diverse negative examples to the synthetic training data -
sentences containing location names, plant names, and other nouns that appear
near animals in natural text. Alternatively, fine-tune on a real-world NER
dataset with animal annotations.

**Plural synonyms are not resolved.** The NER model resolves singular synonyms
correctly ("puppy" -> "dog", "kitten" -> "cat"), and `normalize_animal_name()`
strips plurals of the 10 canonical class names ("dogs" -> "dog"). However,
**plural synonyms** like "puppies" or "kittens" fall through - "puppy" is not
in `KNOWN_CLASSES`, so the plural strip has no effect.
*Fix:* Add a `SYNONYM_MAP` dict in `pipeline.py` that maps synonyms to
canonical names, and apply it inside `normalize_animal_name()` after plural
stripping. This would catch "puppies" -> strip "ies" -> "puppy" -> synonym lookup
-> "dog". Alternatively, include plural forms of synonyms in the NER training
data so the model learns to resolve them directly.

**Only 10 animal classes.** The image classifier is trained on Animals-10, so
any animal outside those 10 classes (kangaroo, turtle, penguin) will be
misclassified as one of the known classes as shown in some examples in demo. The pipeline handles this gracefully for unknown animals in *text* (NER won't recognize them or they'll be filtered by `KNOWN_CLASSES`), but for unknown animals in *images* the classifier will confidently predict the wrong class.
*Fix:* Add an "unknown" / "other" rejection class, or use a confidence
threshold combined with entropy - if the classifier's top prediction has low
confidence or the probability distribution is spread across multiple classes,
treat it as "unknown" instead of trusting the top-1 prediction. Training on a
larger dataset with more species would help as well.
