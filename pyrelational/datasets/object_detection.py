import os
from typing import Any, Callable, Dict, Optional, Tuple

import torchvision.datasets as datasets
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose


class VOCDetectionDataset(Dataset):  # type: ignore
    """
    Pascal VOC dataset for object detection.

    This dataset includes images along with their corresponding annotations
    (object bounding boxes and labels). It's widely used for object detection tasks.
    This class automatically downloads the dataset and prepares it for use in a
    PyTorch model, ensuring no manual download is required.

    Parameters:
    - root (str): The directory where the dataset will be stored.
      Defaults to './data/voc'.
    - year (str): The year of the VOC dataset edition. Defaults to '2012'.
    - image_set (str): Specify 'train', 'trainval', or 'val' to use the respective
      dataset. Default is 'val'.
    - download (bool): If true, downloads the dataset from the internet and
      puts it in the root directory. If the dataset is already downloaded, it is not
      downloaded again.
    - transform (Callable[[Image.Image], Any], optional): Optional transform to be applied on a sample.
    """

    def __init__(
        self,
        root: str = "./data/voc",
        year: str = "2012",
        image_set: str = "val",
        download: bool = True,
        transform: Optional[Callable[[Image.Image], Any]] = None,
    ):
        super().__init__()
        self.root = root
        self.transform = transform or Compose([])
        self.dataset = datasets.VOCDetection(
            root=root, year=year, image_set=image_set, download=download, transform=transform
        )

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Get item method for dataset indexing.

        Parameters:
        - idx (int): Index of the item.

        Returns:
        - Tuple[Image.Image, Dict[str, Any]]: Tuple containing the image and its annotations.
        """
        image, target = self.dataset[idx]
        return image, target

    def __len__(self) -> int:
        """
        Get the total number of samples in the dataset.

        Returns:
        - int: Total number of samples.
        """
        return len(self.dataset)
