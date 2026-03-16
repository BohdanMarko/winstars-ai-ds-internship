from dataclasses import dataclass, field


@dataclass
class NERConfig:
    """Default hyperparameters for NER model training.

    Attributes:
        model_name_or_path: Base model from HuggingFace Hub. BERT pre-trained 
            on NER (PER/ORG/LOC), we fine-tune it for our 3 animal labels.
        
        output_dir: Directory to save the trained model and tokenizer.
        
        num_train_epochs: Number of full passes through the training dataset.
        
        per_device_train_batch_size: Sentences processed per training step.
            Larger = faster but more GPU memory.
        
        per_device_eval_batch_size: Same for validation. Can be larger since
            no gradients are stored during evaluation.
        
        learning_rate: How much the model adjusts weights per step. Small
            value (2e-5) is typical for fine-tuning to preserve pre-trained knowledge.
        
        weight_decay: Regularization that penalizes large weights to prevent
            overfitting on training data.
        
        warmup_steps: Learning rate gradually ramps from 0 to target over these
            steps. Prevents destabilizing the model with a sudden start.
        
        max_seq_length: Maximum tokens per sentence (longer ones are truncated).
        
        seed: Fixed random seed for reproducibility.
        
        num_samples_per_animal: Sentences generated per animal name in the
            synthetic dataset (26 names x 500 = 13,000 positive examples).
        
        label_list: All possible BIO tags. Determines the size of the
            classification head (3 output neurons).
    """

    model_name_or_path: str = "dslim/bert-base-NER"
    output_dir: str = "models/ner"
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 200
    max_seq_length: int = 128
    seed: int = 67
    num_samples_per_animal: int = 500
    label_list: list[str] = field(default_factory=lambda: ["O", "B-ANIMAL", "I-ANIMAL"])
