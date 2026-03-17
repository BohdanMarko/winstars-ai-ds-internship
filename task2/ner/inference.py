"""Extract animal entities from text.

python -m ner.inference
"""

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from ner.dataset import SYNONYMS


class NERPredictor:
    """Loads a fine-tuned NER model and extracts animal entities from text."""

    def __init__(self, model_path: str = "models/ner", device: str | None = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self.model.to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def predict(self, text: str, threshold: float = 0.85) -> list[str]:
        """Extract animal entities from text.

        Returns canonical animal names, synonyms resolved - "kitty" -> "cat".
        """
        if not text.strip():
            raise ValueError("Input text must not be empty.")

        inputs = self.tokenizer(
            text,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
        )
        offset_mapping = inputs.pop("offset_mapping")[0]
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=-1)[0]
        pred_ids = torch.argmax(probs, dim=-1)

        # Group consecutive B-ANIMAL / I-ANIMAL tokens into entity spans
        entities = []
        current_start = None
        current_end = None

        for i, (pred_id, prob, offset) in enumerate(zip(pred_ids, probs, offset_mapping)):
            # Skip special tokens ([CLS], [SEP]) — offset (0,0)
            if offset[0] == 0 and offset[1] == 0:
                continue

            label = self.id2label[pred_id.item()]
            confidence = prob[pred_id].item()

            is_entity = label in ("B-ANIMAL", "I-ANIMAL") and confidence >= threshold

            if is_entity and current_start is not None and offset[0].item() <= current_end:
                # Continuation of current entity (sub-word or I-ANIMAL)
                current_end = offset[1].item()
            elif label == "B-ANIMAL" and confidence >= threshold:
                if current_start is not None:
                    entities.append((current_start, current_end))
                current_start = offset[0].item()
                current_end = offset[1].item()
            else:
                if current_start is not None:
                    entities.append((current_start, current_end))
                    current_start = None
                    current_end = None

        if current_start is not None:
            entities.append((current_start, current_end))

        result = []
        for start, end in entities:
            name = text[start:end].lower().strip()
            canonical = SYNONYMS.get(name, name)
            result.append(canonical)

        return result


if __name__ == "__main__":
    predictor = NERPredictor()

    test_cases = [
        "There is a dog in the picture.",
        "I can see a kitty and a horse.",
        "Trying to get into winstars ai internship.",
        "Look at this beautiful stallion!!!",
    ]

    for text in test_cases:
        entities = predictor.predict(text)
        print(f"{text!r:50s} -> {entities}")
