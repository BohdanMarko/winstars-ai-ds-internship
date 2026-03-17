import os
import subprocess
from pathlib import Path
import kagglehub
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from common import ANIMALS as CLASS_NAMES

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRANSLATE = {
    "cane": "dog",
    "cavallo": "horse",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "ragno": "spider",
    "scoiattolo": "squirrel"
}


class _ToRGB:
    """Convert PIL image to RGB (picklable, unlike lambda)."""
    def __call__(self, img):
        return img.convert("RGB")


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        _ToRGB(),
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        _ToRGB(),
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _download_and_prepare(data_dir: str) -> Path:
    """Download Animals-10 via kagglehub and rename Italian folders to English.

    Returns the path to the prepared directory with English folder names.
    """
    data_path = Path(data_dir)
    
    if data_path.exists() and (data_path / "cat").is_dir():
        return data_path

    raw_path = Path(kagglehub.dataset_download("alessiocorrado99/animals10")) / "raw-img"

    data_path.mkdir(parents=True, exist_ok=True)
    for italian, english in TRANSLATE.items():
        src = raw_path / italian
        dst = data_path / english
        if src.exists() and not dst.exists():
            # Junction works on Windows without admin privileges
            if os.name == "nt":
                subprocess.run(
                    ["cmd", "/c", "mklink", "/J", str(dst), str(src)],
                    check=True, capture_output=True,
                )
            else:
                os.symlink(src, dst, target_is_directory=True)

    return data_path


def prepare_dataset(
    data_dir: str = "data/animals10",
    batch_size: int = 32,
    val_split: float = 0.15,
    seed: int = 67,
    num_workers: int = 4,
    image_size: int = 224,) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    """Load Animals-10, split into train/val, return DataLoaders.

    Returns: (train_loader, val_loader, class_to_idx)
    """
    data_path = _download_and_prepare(data_dir)

    # Load full dataset to get file list and targets
    full_dataset = ImageFolder(str(data_path))

    # Stratified split on indices
    targets = [s[1] for s in full_dataset.samples]
    train_idx, val_idx = train_test_split(
        range(len(full_dataset)),
        test_size=val_split,
        stratify=targets,
        random_state=seed,
    )

    # Build separate datasets with appropriate transforms
    train_dataset = ImageFolder(str(data_path), transform=get_train_transforms(image_size))
    val_dataset = ImageFolder(str(data_path), transform=get_val_transforms(image_size))

    # Filter samples to their respective split
    train_samples = [full_dataset.samples[i] for i in train_idx]
    val_samples = [full_dataset.samples[i] for i in val_idx]
    train_dataset.samples = train_samples
    train_dataset.targets = [s[1] for s in train_samples]
    val_dataset.samples = val_samples
    val_dataset.targets = [s[1] for s in val_samples]

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
    
    return train_loader, val_loader, class_to_idx
