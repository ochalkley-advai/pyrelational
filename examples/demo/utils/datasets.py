"""
Simple datasets in PyTorch to use in examples
"""
import json
import os

import torch
from PIL import Image
from sklearn.datasets import load_breast_cancer, load_diabetes
from torch.utils.data import Dataset


class COCODataset(Dataset):
    """
    Dataset class for COCO-style object detection.
    """

    def __init__(self, image_dir, annotation_file, transform=None):
        """
        Args:
            image_dir (str): Directory with all the images.
            annotation_file (str): Path to the JSON file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.transform = transform
        with open(annotation_file) as f:
            self.annotations = json.load(f)
        self.images = self.annotations["images"]
        self.categories = {category["id"]: category["name"] for category in self.annotations["categories"]}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_id = self.images[idx]["id"]
        img_path = os.path.join(self.image_dir, self.images[idx]["file_name"])
        image = Image.open(img_path).convert("RGB")
        annotations = [anno for anno in self.annotations["annotations"] if anno["image_id"] == img_id]
        boxes = []
        labels = []
        for anno in annotations:
            boxes.append(anno["bbox"])
            labels.append(anno["category_id"])

        sample = {
            "image": image,
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class DiabetesDataset(Dataset):
    """A small regression dataset for examples"""

    def __init__(self):
        # Load the diabetes dataset
        diabetes_X, diabetes_y = load_diabetes(return_X_y=True)
        self.x = torch.FloatTensor(diabetes_X)
        self.y = torch.FloatTensor(diabetes_y)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class BreastCancerDataset(Dataset):
    """A small classification dataset for examples"""

    def __init__(self):
        super(BreastCancerDataset, self).__init__()
        sk_x, sk_y = load_breast_cancer(return_X_y=True)
        self.x = torch.FloatTensor(sk_x)
        self.y = torch.LongTensor(sk_y)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
