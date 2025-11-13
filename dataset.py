import os
from torch.utils.data import Dataset, DataLoader
import json
from typing import Optional, Dict, Any

from PIL import Image
import torch
from torchvision import transforms
import numpy as np


class EllipseDataset(Dataset):
    """Dataset for ellipse images and their parameters.

    Expected annotation JSON format (example):
    {
      "images": [ {"id": 0, "file_name": "0.png"}, ... ],
      "annotations": [ {"image_id": 0, "cx": ..., "cy": ..., "covariance_matrix": [[..., ...],[..., ...]]}, ... ]
    }
    """

    def __init__(self, images_dir: str, annotations_path: str, transform: Optional[transforms.Compose] = None, inference: bool = False):
        self.images_dir = images_dir
        self.transform = transform if transform is not None else transforms.Compose([transforms.ToTensor()])
        self.inference = inference

        # Validate paths
        if not os.path.isdir(self.images_dir):
            raise ValueError(f"images_dir does not exist or is not a directory: {self.images_dir}")
        if not os.path.isfile(annotations_path):
            raise ValueError(f"annotations_path does not exist: {annotations_path}")

        # Load annotations JSON file
        with open(annotations_path, "r") as f:
            data = json.load(f)

        self.images = data.get("images", [])
        self.annotations = data.get("annotations", [])

        # Build maps
        # map image id -> image info dict
        self.id_to_image: Dict[int, Dict[str, Any]] = {img["id"]: img for img in self.images}
        # map image id -> annotation dict
        self.id_to_ann: Dict[int, Dict[str, Any]] = {ann["image_id"]: ann for ann in self.annotations}

        if len(self.images) == 0:
            raise ValueError("No images found in annotations file (data['images'] is empty or missing)")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_info = self.images[idx]
        image_id = image_info.get("id")

        # Determine image path 
        img_path = os.path.join(self.images_dir, f"id_{image_id}.png")

        # Check that the file actually exists
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")    

        # Load image
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        if self.inference:
            return {"image": image, "image_id": image_id}

        ann = self.id_to_ann.get(image_id)
        cx = ann.get("cx")
        cy = ann.get("cy")
        cov = np.array(ann.get("covariance_matrix"), dtype=np.float32)

        l1 = cov[0, 0] #/ 400.0
        l2 = cov[1, 1] #/ 400.0
        l3 = cov[0, 1] #/ 400.0

        params = np.array([cx, cy, l1, l2, l3], dtype=np.float32)
        return {"image": image, "params": torch.tensor(params), "image_id": image_id}


if __name__ == "__main__":
    images_dir = "/home/hritik/Desktop/Hritik/Project/Dataset/Ellipses"
    annotations_path = "/home/hritik/Desktop/Hritik/Project/Dataset/annotations.json"

    # Optional: simple transforms
    transform = transforms.Compose([
    transforms.Resize((20, 20)),     #  Resize to 20x20
    transforms.Grayscale(num_output_channels=1),  #  Make it single channel
    transforms.ToTensor()
])


    print("Creating dataset")
    dataset = EllipseDataset(images_dir=images_dir, annotations_path=annotations_path, transform=transform)

    print(f"Dataset created successfully with {len(dataset)} samples.")

   