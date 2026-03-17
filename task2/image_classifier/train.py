"""Two-phase fine-tuning of ResNet-50 on Animals-10.

Phase 1: Frozen backbone - only train the classification head.
Phase 2: Full fine-tuning - unfreeze backbone with lower learning rate.

python -m image_classifier.train
"""

import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import resnet50, ResNet50_Weights
from common import get_device
from image_classifier.config import ImageClassifierConfig
from image_classifier.dataset import prepare_dataset, CLASS_NAMES


def create_model(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """Load ResNet-50 with a new classification head for num_classes."""
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def run_epoch(model, loader, criterion, optimizer, device, is_train=True):
    """Run one epoch of training or validation. Returns (avg_loss, accuracy)."""
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def train(config: ImageClassifierConfig) -> None:
    """Run full two-phase training pipeline."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = get_device()

    model = create_model(num_classes=len(CLASS_NAMES), pretrained=config.pretrained)
    model.to(device)

    train_loader, val_loader, class_to_idx = prepare_dataset(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        seed=config.seed,
        num_workers=config.num_workers,
        image_size=config.image_size,
    )

    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Phase 1: Frozen backbone")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    optimizer = Adam(model.fc.parameters(), lr=config.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.freeze_backbone_epochs)

    for epoch in range(1, config.freeze_backbone_epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device, is_train=False)
        scheduler.step()

        print(f"  Epoch {epoch}/{config.num_epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / "best_model.pt")

    print("Phase 2: Full fine-tuning")
    for param in model.parameters():
        param.requires_grad = True

    backbone_params = [p for n, p in model.named_parameters() if "fc" not in n]
    head_params = model.fc.parameters()
    optimizer = Adam([
        {"params": backbone_params, "lr": config.lr_backbone},
        {"params": head_params, "lr": config.lr},
    ])

    remaining_epochs = config.num_epochs - config.freeze_backbone_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=remaining_epochs)

    for epoch in range(config.freeze_backbone_epochs + 1, config.num_epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device, is_train=False)
        scheduler.step()

        print(f"  Epoch {epoch}/{config.num_epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / "best_model.pt")

    # Save class mapping
    with open(output_dir / "class_to_idx.json", "w") as f:
        json.dump(class_to_idx, f, indent=2)

    print(f"\nTraining complete. Best val_acc: {best_val_acc:.4f}")
    print(f"Model saved to {output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    config = ImageClassifierConfig()
    model_path = Path(config.output_dir) / "best_model.pt"
    if model_path.exists():
        print(f"Trained model already exists at {config.output_dir}, skipping training.")
    else:
        train(config)
