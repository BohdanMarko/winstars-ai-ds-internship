"""Classify animals in images.

Loads the trained ResNet-50 model and returns predicted animal class with confidence score.

python -m image_classifier.inference
"""

import json
from pathlib import Path
import torch
from PIL import Image
from image_classifier.dataset import get_val_transforms
from image_classifier.train import create_model


class ImageClassifier:
    """Loads a trained image classification model and predicts animal classes."""

    def __init__(
        self,
        model_path: str = "models/image_classifier/best_model.pt",
        class_map_path: str = "models/image_classifier/class_to_idx.json",
        device: str | None = None
    ):
        with open(class_map_path) as f:
            class_to_idx = json.load(f)
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}

        model = create_model(num_classes=len(class_to_idx), pretrained=False)

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        self.model = model
        self.transforms = get_val_transforms()

    def predict(self, image_path: str, top_k: int = 1) -> list[tuple[str, float]]:
        """Classify an image file. Returns list of (class_name, confidence)."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(path).convert("RGB")
        return self.predict_from_pil(image, top_k=top_k)

    def predict_from_pil(self, image: Image.Image, top_k: int = 1) -> list[tuple[str, float]]:
        """Classify a PIL image. Returns list of (class_name, confidence)."""
        image = image.convert("RGB")
        tensor = self.transforms(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)

        probs = torch.softmax(logits, dim=-1)[0]
        top_probs, top_indices = torch.topk(probs, k=top_k)

        return [
            (self.idx_to_class[idx.item()], prob.item())
            for prob, idx in zip(top_probs, top_indices)
        ]


if __name__ == "__main__":
    classifier = ImageClassifier()

    test_images = list(Path("test_data").glob("*"))[:3]
    
    for img_path in test_images:
            path = Path(str(img_path))
            image = Image.open(path).convert("RGB")
            preds = classifier.predict_from_pil(image, top_k=10)
            print(f"{img_path.name}:")
            for cls, conf in preds:
                print(f"  {cls}: {conf:.4f}")
