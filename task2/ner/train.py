"""Fine-tune dslim/bert-base-NER on synthetic animal NER dataset.

Reinitializes the classification head from 9 labels (PER/ORG/LOC/MISC)
to 3 labels (O, B-ANIMAL, I-ANIMAL) while keeping the BERT encoder
weights that already understand NER patterns.
"""

from pathlib import Path
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
from ner.dataset import create_ner_datasets, ID_TO_LABEL, LABEL_TO_ID
from ner.config import NERConfig


def compute_metrics(eval_preds) -> dict:
    """Compute entity-level precision, recall, F1 using seqeval."""
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)

    true_labels = []
    true_preds = []

    for pred_seq, label_seq in zip(preds, labels):
        seq_labels = []
        seq_preds = []
        for p, l in zip(pred_seq, label_seq):
            if l == -100:
                continue
            seq_labels.append(ID_TO_LABEL[l])
            seq_preds.append(ID_TO_LABEL[p])
        true_labels.append(seq_labels)
        true_preds.append(seq_preds)

    return {
        "precision": precision_score(true_labels, true_preds),
        "recall": recall_score(true_labels, true_preds),
        "f1": f1_score(true_labels, true_preds),
    }


def train(config: NERConfig) -> None:
    """Run full NER training pipeline."""
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)

    model = AutoModelForTokenClassification.from_pretrained(
        config.model_name_or_path,
        num_labels=len(config.label_list),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        ignore_mismatched_sizes=True,
    )

    train_ds, val_ds = create_ner_datasets(tokenizer, config)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        seed=config.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    metrics = trainer.evaluate()
    print(f"\nFinal evaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    config = NERConfig()
    model_path = Path(config.output_dir) / "model.safetensors"
    if model_path.exists():
        print(f"Trained model already exists at {config.output_dir}, skipping training.")
    else:
        train(config)
