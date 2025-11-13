from torch.utils.data import random_split, DataLoader
from dataset import EllipseDataset
from torchvision import transforms
import torch
from typing import Optional



def create_dataloaders(dataset: EllipseDataset, batch_size: int = 64, num_workers: int = 4, device: Optional[torch.device] = None):
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Decide pin_memory based on provided device (if any) or CUDA availability
    if device is None:
        use_cuda = torch.cuda.is_available()
    else:
        use_cuda = (device.type == "cuda")

    pin_memory = bool(use_cuda)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    images_dir = "/home/hritik/Desktop/Hritik/Project/Dataset/Ellipses"
    annotations_path = "/home/hritik/Desktop/Hritik/Project/Dataset/annotations.json"

    transform = transforms.Compose([
        transforms.Resize((20, 20)),     #  Resize to 20x20
        transforms.Grayscale(num_output_channels=1),  #  Make it single channel
        transforms.ToTensor()
    ])


    dataset = EllipseDataset(images_dir=images_dir, annotations_path=annotations_path, transform=transform)
    print(f"Dataset created successfully with {len(dataset)} samples.")

    # select device safely (use CUDA only when available and compatible)
    def select_device(min_arch: int = 70) -> torch.device:
        """Return a torch.device: 'cuda' if available and meets compute capability, else 'cpu'.

        min_arch: e.g. 70 means sm_70 (compute capability 7.0). If your installed PyTorch
        expects sm_70+ and GPU is older, this will choose CPU.
        """
        if not torch.cuda.is_available():
            return torch.device("cpu")
        try:
            prop = torch.cuda.get_device_properties(0)
            compute = prop.major * 10 + prop.minor
            if compute >= min_arch:
                return torch.device("cuda")
            print(f"GPU {prop.name} compute capability {prop.major}.{prop.minor} < required {min_arch/10:.1f}; using CPU.")
            return torch.device("cpu")
        except Exception:
            # If anything goes wrong querying properties, fallback to CPU
            return torch.device("cpu")

    def move_batch_to_device(batch: dict, device: torch.device) -> dict:
        """Move all tensor values in a batch dict to the given device."""
        return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

    device = select_device()
    print("Using device:", device)

    train_loader, val_loader, test_loader = create_dataloaders(dataset, batch_size=64, num_workers=4, device=device)
    print(f"Train samples: {len(train_loader.dataset)}, Validation: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # Quick check: load a batch from train_loader and move to device for smoke-test
    train_batch = next(iter(train_loader))
    train_batch = move_batch_to_device(train_batch, device)
    print("Train batch - images:", train_batch["image"].shape, "params:", train_batch["params"].shape)
