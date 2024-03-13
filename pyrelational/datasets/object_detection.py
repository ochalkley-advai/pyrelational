from typing import Any, Dict, Tuple

import torch
from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection
from torchvision.transforms import Compose, ToTensor


class VOCDetectionDataset(Dataset[Tuple[torch.Tensor, Dict[str, Any]]]):
    """
    Dataset class for Pascal VOC that includes functionality for creating
    random train/test splits. Inherits from PyTorch's Dataset class.

    Attributes:
        root (str): Directory where the VOC dataset is located or will be downloaded.
        year (str): The dataset's year. Defaults to '2012'.
        image_set (str): The type of image set to use ('train', 'val', 'trainval', 'test').
        transform (Compose, optional): A function/transform that takes in an image and returns a transformed version.
        download (bool): If true, downloads the dataset from the internet and puts it in the root directory.
        n_splits (int): The number of random splits to create.
        test_size (float): The proportion of the dataset to include in the test split.
    """

    def __init__(
        self,
        root: str = "./data/voc",
        year: str = "2012",
        image_set: str = "val",
        transform: Compose = None,
        download: bool = True,
        n_splits: int = 5,
        test_size: float = 0.2,
    ) -> None:
        super().__init__()
        self.dataset = VOCDetection(
            root, year=year, image_set=image_set, download=download, transform=transform or ToTensor()
        )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Retrieve an item from the dataset.

        Parameters:
            idx (int): The index of the item.

        Returns:
            Tuple containing the image (as a torch.Tensor) and its annotations (as a dictionary).
        """
        image, annotation = self.dataset[idx]
        return image, annotation

    def __len__(self) -> int:
        """
        Determine the total number of samples in the dataset.

        Returns:
            The number of samples.
        """
        return len(self.dataset)
