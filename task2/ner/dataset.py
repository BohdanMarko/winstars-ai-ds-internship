""" Synthetic NER dataset generation for animal entity extraction.
    
    Generates BIO-tagged sentences from diverse templates, producing HuggingFace datasets ready for the Trainer.
"""

import random
from collections import Counter
from datasets import Dataset
from ner.config import NERConfig


ANIMALS: list[str] = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"]

SYNONYMS: dict[str, str] = {
    "kitty": "cat", 
    "kitten": "cat",
    "pup": "dog", 
    "puppy": "dog", 
    "hound": "dog",
    "hen": "chicken", 
    "rooster": "chicken",
    "lamb": "sheep", 
    "ewe": "sheep", 
    "ram": "sheep",
    "mare": "horse", 
    "stallion": "horse", 
    "pony": "horse", 
    "foal": "horse",
    "calf": "cow", 
    "bull": "cow"
}

ALL_ANIMAL_NAMES: list[str] = ANIMALS + list(SYNONYMS.keys())

LABEL_TO_ID: dict[str, int] = {"O": 0, "B-ANIMAL": 1, "I-ANIMAL": 2}
ID_TO_LABEL: dict[int, str] = {v: k for k, v in LABEL_TO_ID.items()}

POSITIVE_TEMPLATES = [
    "There is a {animal} in the picture.",
    "I can see a {animal}.",
    "The photo shows a {animal}.",
    "A {animal} is standing in the field.",
    "The image contains a {animal}.",
    "That is definitely a {animal}.",
    "Is that a {animal}?",
    "Can you see the {animal}?",
    "Is there a {animal} in this image?",
    "Look at this {animal}!",
    "Wow, what a beautiful {animal}!",
    "Check out the {animal} in the image.",
    "Pretty sure that's a {animal}.",
    "Looks like a {animal} to me.",
    "Yep, definitely a {animal}.",
    "The {animal} appears to be resting.",
    "A young {animal} is playing in the grass.",
    "The {animal} is looking directly at the camera.",
]

_NEGATIVE = [
    "There is nothing in the picture.",
    "This is a beautiful landscape.",
    "I don't see anything interesting here.",
    "The photo appears to be blurry.",
    "There are no animals in this photo.",
]


def _tag_sentence(sentence: str, animal: str | None) -> tuple[list[str], list[int]]:
    """Split sentence into word tokens and assign BIO tags.

    Returns (tokens, ner_tags) where ner_tags are integer label IDs.
    """
    tokens = sentence.split()
    tags = [LABEL_TO_ID["O"]] * len(tokens)

    if animal is None:
        return tokens, tags

    animal_parts = animal.split()  # we believe that all current names are single-word
    n = len(animal_parts)

    for i in range(len(tokens) - n + 1):
        # Strip punctuation for comparison
        window = [t.strip(".,!?;:'\"()[]") for t in tokens[i:i + n]]
        if [w.lower() for w in window] == [a.lower() for a in animal_parts]:
            tags[i] = LABEL_TO_ID["B-ANIMAL"]
            for j in range(1, n):
                tags[i + j] = LABEL_TO_ID["I-ANIMAL"]
            break  # tag first occurrence only

    return tokens, tags


def generate_ner_dataset(num_samples_per_animal: int = 500, seed: int = 67, val_split: float = 0.15) -> tuple[Dataset, Dataset]:
    """Generate a synthetic BIO-tagged NER dataset for animal extraction.

    Args:
        num_samples_per_animal: Number of sentences to generate per animal name (including synonyms).
        seed: Random seed for reproducibility.
        val_split: Fraction of data reserved for validation.

    Returns:
        (train_dataset, val_dataset) as HuggingFace Dataset objects with columns ["tokens", "ner_tags"].
    """
    rng = random.Random(seed)
    all_tokens: list[list[str]] = []
    all_tags: list[list[int]] = []

    # Positive examples: one set per animal name (canonical + synonyms)
    for animal in ALL_ANIMAL_NAMES:
        templates = rng.choices(POSITIVE_TEMPLATES, k=num_samples_per_animal)
        for tpl in templates:
            sentence = tpl.replace("{animal}", animal)
            tokens, tags = _tag_sentence(sentence, animal)
            all_tokens.append(tokens)
            all_tags.append(tags)

    # Negative examples: +- 12% of total positive count
    num_positive = len(all_tokens)
    num_negative = max(1, int(num_positive * 0.12))
    neg_templates = rng.choices(_NEGATIVE, k=num_negative)
    for tpl in neg_templates:
        tokens, tags = _tag_sentence(tpl, animal=None)
        all_tokens.append(tokens)
        all_tags.append(tags)

    # Shuffle
    combined = list(zip(all_tokens, all_tags))
    rng.shuffle(combined)
    all_tokens, all_tags = zip(*combined)

    # Split into train / val
    split_idx = int(len(all_tokens) * (1 - val_split))
    train_ds = Dataset.from_dict({
        "tokens": list(all_tokens[:split_idx]),
        "ner_tags": list(all_tags[:split_idx]),
    })
    val_ds = Dataset.from_dict({
        "tokens": list(all_tokens[split_idx:]),
        "ner_tags": list(all_tags[split_idx:]),
    })

    return train_ds, val_ds


def tokenize_and_align_labels(examples: dict, tokenizer, max_length: int = 128) -> dict:
    """Tokenize word-level tokens and align BIO labels to sub-word tokens.

    First sub-token of each word keeps the word's label; subsequent
    sub-tokens and special tokens ([CLS], [SEP]) get -100 (ignored by loss).
    """
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    all_labels = []
    for i, word_tags in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        labels = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)
            elif word_id != prev_word_id:
                labels.append(word_tags[word_id])
            else:
                labels.append(-100)
            prev_word_id = word_id
        all_labels.append(labels)

    tokenized["labels"] = all_labels
    return tokenized


def create_ner_datasets(tokenizer, config: NERConfig) -> tuple[Dataset, Dataset]:
    """Generate and tokenize NER datasets ready for HuggingFace Trainer."""
    train_ds, val_ds = generate_ner_dataset(
        num_samples_per_animal=config.num_samples_per_animal,
        seed=config.seed,
    )

    def _tokenize(examples):
        return tokenize_and_align_labels(
            examples, tokenizer, max_length=config.max_seq_length,
        )

    train_ds = train_ds.map(_tokenize, batched=True, remove_columns=["tokens", "ner_tags"])
    val_ds = val_ds.map(_tokenize, batched=True, remove_columns=["tokens", "ner_tags"])

    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    val_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    return train_ds, val_ds


if __name__ == "__main__":
    train_ds, val_ds = generate_ner_dataset()

    total = len(train_ds) + len(val_ds)
    print(f"Total examples: {total}")
    print(f"  Train: {len(train_ds)}")
    print(f"  Val:   {len(val_ds)}")

    # Count positive / negative
    neg_train = sum(1 for tags in train_ds["ner_tags"] if all(t == 0 for t in tags))
    neg_val = sum(1 for tags in val_ds["ner_tags"] if all(t == 0 for t in tags))
    neg_total = neg_train + neg_val
    print(f"\nNegative examples: {neg_total} ({neg_total / total * 100:.1f}%)")
    print(f"Positive examples: {total - neg_total}")

    # Count per animal (from positive examples in train set)
    animal_counter: Counter[str] = Counter()
    for tokens, tags in zip(train_ds["tokens"], train_ds["ner_tags"]):
        entity_parts = []
        for tok, tag in zip(tokens, tags):
            if tag in (LABEL_TO_ID["B-ANIMAL"], LABEL_TO_ID["I-ANIMAL"]):
                entity_parts.append(tok.strip(".,!?;:'\"()[]").lower())
            else:
                if entity_parts:
                    animal_counter[" ".join(entity_parts)] += 1
                    entity_parts = []
        if entity_parts:
            animal_counter[" ".join(entity_parts)] += 1

    print(f"\nAnimal distribution (train set, top 30):")
    for name, count in animal_counter.most_common(30):
        print(f"  {name}: {count}")

    # Sample sentences
    print("\n--- Sample sentences ---")
    for i in range(5):
        tokens = train_ds[i]["tokens"]
        tags = train_ds[i]["ner_tags"]
        tagged = " ".join(
            f"[{t}|{ID_TO_LABEL[tag]}]" if tag != 0 else t
            for t, tag in zip(tokens, tags)
        )
        print(f"  {tagged}")
