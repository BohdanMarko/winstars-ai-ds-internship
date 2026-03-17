from dataclasses import dataclass


@dataclass
class ImageClassifierConfig:
    """Default hyperparameters for image classifier training.

    Attributes:
        data_dir: Path to Animals-10 dataset with English folder names.
        
        output_dir: Directory to save best_model.pt and class_to_idx.json.
        
        model_arch: Backbone architecture. ResNet-50 is a 50-layer deep CNN, well suited for transfer learning.
        
        pretrained: Whether to load ImageNet pre-trained weights. The model
            already understands basic visual patterns (edges, textures, shapes),
            we just fine-tune it for our 10 animal classes.
        
        num_epochs: Total training epochs (Phase 1 + Phase 2).
        
        batch_size: Images processed per training step. Balance between speed and GPU memory.
        
        lr: Learning rate for the classification head. Relatively high (1e-3) because the head is randomly initialized and needs fast learning.
        
        lr_backbone: Learning rate for the backbone in Phase 2. 10x lower than lr to gently adjust pre-trained weights without destroying them.
        
        freeze_backbone_epochs: Epochs with frozen backbone (Phase 1). Gives the head time to catch up before joint fine-tuning begins.
        
        scheduler: LR scheduler type. "cosine" smoothly decreases the learning rate from initial value to 0 following a cosine curve.
        
        seed: Fixed random seed for reproducibility.
        
        num_workers: Parallel processes for loading images from disk. They prepare next batches while the GPU processes the current one.
        
        image_size: Input image size in pixels (224x224). Standard for ResNet, matching the ImageNet pre-training resolution.
    """

    data_dir: str = "data/animals10"
    output_dir: str = "models/image_classifier"
    model_arch: str = "resnet50"
    pretrained: bool = True
    num_epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-3
    lr_backbone: float = 1e-4
    freeze_backbone_epochs: int = 5
    scheduler: str = "cosine"
    seed: int = 67
    num_workers: int = 4
    image_size: int = 224
