"""Animal text-image matching pipeline.

python pipeline.py --text "There is a big elephant on the picture" --image_path <elephant.jpg>
"""

import argparse
from pathlib import Path
from common import ANIMALS
from ner.inference import NERPredictor
from image_classifier.inference import ImageClassifier


KNOWN_CLASSES: set[str] = set(ANIMALS)


def normalize_animal_name(name: str) -> str:
    name = name.lower().strip()

    if name.endswith("ies") and name[:-3] + "y" in KNOWN_CLASSES:
        name = name[:-3] + "y"
    elif name.endswith("s") and name[:-1] in KNOWN_CLASSES:
        name = name[:-1]

    return name


class AnimalPipeline:
    """Combines NER + Image Classification to verify text-image animal match."""

    def __init__(
        self,
        ner_model_path: str,
        clf_model_path: str,
        class_map_path: str,
        confidence_threshold: float = 0.6,
        device: str | None = None,
    ):
        self.ner = NERPredictor(ner_model_path, device=device)
        self.classifier = ImageClassifier(clf_model_path, class_map_path, device=device)
        self.confidence_threshold = confidence_threshold

    def run(self, text: str, image_path: str, verbose: bool = False) -> dict:
        if not text.strip():
            raise ValueError("Text input cannot be empty")
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # NER
        raw_entities = self.ner.predict(text)
        normalized = [normalize_animal_name(e) for e in raw_entities]
        known = [a for a in normalized if a in KNOWN_CLASSES]

        if not raw_entities:
            result = {
                "match": False,
                "image_prediction": None,
                "image_confidence": None,
                "reason": "No animal entity found in text",
            }
            if verbose:
                print("\nNo animal mentioned in the text. Skipping image analysis.\n")
            return result

        if not known:
            result = {
                "match": False,
                "image_prediction": None,
                "image_confidence": None,
                "reason": f"Animal '{normalized[0]}' not in known classes",
            }
            if verbose:
                print(f"\nFound '{normalized[0]}' in text, but it's not one of the known animal classes.\n")
            return result

        # Image Classification
        predictions = self.classifier.predict(image_path, top_k=1)
        predicted_class, confidence = predictions[0]

        if confidence < self.confidence_threshold:
            result = {
                "match": False,
                "image_prediction": predicted_class,
                "image_confidence": confidence,
                "reason": f"Low confidence: {predicted_class} ({confidence:.2f} < {self.confidence_threshold})",
            }
            if verbose:
                print(f"\nConfidence too low ({confidence:.0%} < {self.confidence_threshold:.0%}). Cannot confirm.\n")
            return result

        match = predicted_class in known
        text_animal = known[0]

        if verbose:
            if match:
                print(f"\nCorrect! Text mentions '{text_animal}' and the image is indeed a '{predicted_class}'.\n")
            else:
                print(f"\nWrong! Text mentions '{text_animal}', but the image shows a '{predicted_class}'.\n")

        return {
            "match": match,
            "image_prediction": predicted_class,
            "image_confidence": confidence,
            "reason": f"Text mentions '{text_animal}', image classified as '{predicted_class}' ({confidence:.2f})",
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animal text-image matching pipeline")
    parser.add_argument("--text", required=True, help="Input text")
    parser.add_argument("--image_path", required=True, help="Path to image")
    args = parser.parse_args()

    pipeline = AnimalPipeline(
        ner_model_path="models/ner", 
        clf_model_path="models/image_classifier/best_model.pt",
        class_map_path="models/image_classifier/class_to_idx.json", 
        confidence_threshold=0.6)
    
    result = pipeline.run(args.text, args.image_path, verbose=True)
