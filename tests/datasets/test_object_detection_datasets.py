import unittest

import torchvision.transforms as transforms

from pyrelational.datasets.object_detection import VOCDetectionDataset


class TestVOCDetectionDataset(unittest.TestCase):
    """Unit tests for the VOCDetectionDataset class."""

    @classmethod
    def setUpClass(cls):
        """Set up the dataset once for all tests."""
        cls.root = "./data/voc"
        cls.year = "2012"
        cls.image_set = "val"
        cls.transform = transforms.Compose([transforms.ToTensor()])
        cls.dataset = VOCDetectionDataset(
            root=cls.root, year=cls.year, image_set=cls.image_set, transform=cls.transform, download=True
        )

    def test_dataset_initialization(self):
        """Test initialization of the dataset."""
        self.assertIsInstance(self.dataset, VOCDetectionDataset)

    def test_dataset_len(self):
        """Test the length of the dataset."""
        # This will depend on the actual size of the dataset, so it's more of a sanity check
        self.assertTrue(len(self.dataset) > 0)

    def test_get_item(self):
        """Test retrieving an item from the dataset."""
        image, target = self.dataset[0]
        self.assertIsNotNone(image)
        self.assertIsInstance(target, dict)
        self.assertIn("annotation", target)

    def test_transform(self):
        """Test if the transform is applied correctly."""
        image, _ = self.dataset[0]
        # Assuming default transform is ToTensor, which changes PIL images to Tensors
        self.assertEqual(image.dim(), 3)  # Check if image is transformed to a tensor


if __name__ == "__main__":
    unittest.main()
