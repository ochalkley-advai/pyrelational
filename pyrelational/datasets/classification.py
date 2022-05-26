"""Classification datasets that can be used for benchmarking AL strategies
"""

import os
import urllib.request
from os import path

import numpy as np
import pyreadr
import scipy.io
import torch
import torch.distributions as distributions
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from .uci_datasets import UCIDatasets


class SynthClass1(Dataset):
    """
    Synth1 dataset as described in Yang and Loog

    Consists of a binary classification task of positive
    and negative class samples being generated by a multivariate
    gaussian distribution centered at [1,1] and [-1,-1]
    respectively.

    :param n_splits: an int describing the number of class stratified
            splits to compute
    :param size: an int describing the number of observations the dataset
            is to have
    :param random_seed: random seed for reproducibility on splits
    """

    def __init__(self, n_splits=5, size=500, random_seed=1234):
        super(SynthClass1, self).__init__()
        self.n_splits = n_splits
        pos_distribution = distributions.MultivariateNormal(torch.FloatTensor([3, 3]), torch.eye(2))
        neg_distribution = distributions.MultivariateNormal(torch.FloatTensor([0, 0]), torch.eye(2))

        num_pos = int(size / 2.0)
        num_neg = size - num_pos
        pos_samples = torch.vstack([pos_distribution.sample() for _ in range(num_pos)])
        neg_samples = torch.vstack([neg_distribution.sample() for _ in range(num_neg)])
        pos_targets = torch.ones(num_pos, dtype=torch.long)
        neg_targets = torch.ones(num_neg, dtype=torch.long) * 0

        self.x = torch.cat([pos_samples, neg_samples])
        self.y = torch.cat([pos_targets, neg_targets])

        skf = StratifiedKFold(n_splits=n_splits)
        self.data_splits = skf.split(self.x, self.y)
        self.data_splits = [(idx[0], idx[1]) for idx in self.data_splits]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class SynthClass2(Dataset):
    """
    Synth2 dataset as described in Yang and Loog

    Originally proposed by Huang et al in:
    Active learning by querying informative
    and representative examples

    :param n_splits: an int describing the number of class stratified
            splits to compute
    :param size: an int describing the number of observations the dataset
            is to have
    :param random_seed: random seed for reproducibility on splits
    """

    def __init__(self, n_splits=5, size=500, random_seed=1234):
        super(SynthClass2, self).__init__()
        self.n_splits = n_splits

        pos_dist_1 = distributions.MultivariateNormal(torch.FloatTensor([0, 5]), torch.eye(2))
        neg_dist_1 = distributions.MultivariateNormal(torch.FloatTensor([0, -5]), torch.eye(2))

        pos_dist_2 = distributions.MultivariateNormal(torch.FloatTensor([-5, 10]), torch.eye(2))
        pos_dist_3 = distributions.MultivariateNormal(torch.FloatTensor([-5, -10]), torch.eye(2))

        neg_dist_2 = distributions.MultivariateNormal(torch.FloatTensor([5, 10]), torch.eye(2))
        neg_dist_3 = distributions.MultivariateNormal(torch.FloatTensor([5, -10]), torch.eye(2))

        num_pos = int(size / 2.0)
        num_neg = size - num_pos

        # find number of samples to generate from the positives and negative blobs constrained
        # to input size
        num_pos1, num_pos2, num_pos3 = [len(x) for x in self._split(range(num_pos), 3)]
        num_neg1, num_neg2, num_neg3 = [len(x) for x in self._split(range(num_neg), 3)]

        pos_samples_1 = torch.vstack([pos_dist_1.sample() for _ in range(num_pos1)])
        pos_samples_2 = torch.vstack([pos_dist_2.sample() for _ in range(num_pos2)])
        pos_samples_3 = torch.vstack([pos_dist_3.sample() for _ in range(num_pos3)])

        neg_samples_1 = torch.vstack([neg_dist_1.sample() for _ in range(num_neg1)])
        neg_samples_2 = torch.vstack([neg_dist_2.sample() for _ in range(num_neg2)])
        neg_samples_3 = torch.vstack([neg_dist_3.sample() for _ in range(num_neg3)])

        pos_targets = torch.ones(num_pos, dtype=torch.long)
        neg_targets = torch.ones(num_neg, dtype=torch.long) * 0

        self.x = torch.cat([pos_samples_1, pos_samples_2, pos_samples_3, neg_samples_1, neg_samples_2, neg_samples_3])
        self.y = torch.cat([pos_targets, neg_targets])

        skf = StratifiedKFold(n_splits=n_splits)
        self.data_splits = skf.split(self.x, self.y)
        self.data_splits = [(idx[0], idx[1]) for idx in self.data_splits]

    def _split(self, iterable, n):
        # split the iterable into n approximately same size parts
        k, m = divmod(len(iterable), n)
        return (iterable[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class SynthClass3(Dataset):
    """SynthClass3 dataset as described in Yang and Loog

    :param n_splits: an int describing the number of class stratified
            splits to compute
    :param size: an int describing the number of observations the dataset
            is to have
    :param random_seed: random seed for reproducibility on splits
    """

    def __init__(self, n_splits=5, size=500, random_seed=1234):
        super(SynthClass3, self).__init__()
        self.size = size
        self.random_seed = random_seed
        self.n_splits = n_splits

        cov = torch.FloatTensor([[0.60834549, -0.63667341], [-0.40887718, 0.85253229]])
        cov = torch.matmul(cov, cov.T)

        pos_dist_1 = distributions.MultivariateNormal(torch.FloatTensor([0, 0]), cov)
        pos_dist_2 = distributions.MultivariateNormal(torch.FloatTensor([3, 10]), cov)
        neg_dist_1 = distributions.MultivariateNormal(torch.FloatTensor([3, 3]), torch.FloatTensor([[1, 2], [2, 7]]))

        num_pos = int(size / 2.0)
        num_neg1 = size - num_pos
        num_pos1, num_pos2 = [len(x) for x in self._split(range(num_pos), 2)]

        pos_samples_1 = torch.vstack([pos_dist_1.sample() for _ in range(num_pos1)])
        pos_samples_2 = torch.vstack([pos_dist_2.sample() for _ in range(num_pos2)])
        neg_samples_1 = torch.vstack([neg_dist_1.sample() for _ in range(num_neg1)])

        pos_targets = torch.ones(num_pos, dtype=torch.long)
        neg_targets = torch.ones(num_neg1, dtype=torch.long) * 0

        self.x = torch.cat([pos_samples_1, pos_samples_2, neg_samples_1])
        self.y = torch.cat([pos_targets, neg_targets])

        skf = StratifiedKFold(n_splits=n_splits)
        self.data_splits = skf.split(self.x, self.y)
        self.data_splits = [(idx[0], idx[1]) for idx in self.data_splits]

    def _split(self, iterable, n):
        # split the iterable into n approximately same size parts
        k, m = divmod(len(iterable), n)
        return (iterable[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class BreastCancerDataset(Dataset):
    """UCI ML Breast Cancer Wisconsin (Diagnostic) dataset

    :param n_splits: an int describing the number of class stratified
            splits to compute
    """

    def __init__(self, n_splits=5):
        super(BreastCancerDataset, self).__init__()
        sk_x, sk_y = load_breast_cancer(return_X_y=True)
        self.x = torch.FloatTensor(sk_x)
        self.y = torch.LongTensor(sk_y)

        skf = StratifiedKFold(n_splits=n_splits)
        self.data_splits = skf.split(self.x, self.y)
        self.data_splits = [(idx[0], idx[1]) for idx in self.data_splits]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class DigitDataset(Dataset):
    """UCI ML hand-written digits datasets

    From C. Kaynak (1995) Methods of Combining Multiple Classifiers and
    Their Applications to Handwritten Digit Recognition, MSc Thesis,
    Institute of Graduate Studies in Science and Engineering, Bogazici
    University.

    :param n_splits: an int describing the number of class stratified
            splits to compute
    """

    def __init__(self, n_splits=5):
        super(DigitDataset, self).__init__()
        sk_x, sk_y = load_digits(return_X_y=True)
        self.x = torch.FloatTensor(sk_x)  # data
        self.y = torch.LongTensor(sk_y)  # target

        skf = StratifiedKFold(n_splits=n_splits)
        self.data_splits = skf.split(self.x, self.y)
        self.data_splits = [(idx[0], idx[1]) for idx in self.data_splits]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class FashionMNIST(Dataset):
    """Fashion MNIST Dataset

    From Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning
    Algorithms. Han Xiao, Kashif Rasul, Roland Vollgraf. arXiv:1708.07747

    :param n_splits: an int describing the number of class stratified
            splits to compute
    """

    def __init__(self, data_dir="/tmp/", n_splits=5):
        super(FashionMNIST, self).__init__()
        train_dataset = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transforms.ToTensor())
        test_dataset = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transforms.ToTensor())
        dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        self.x = torch.stack([(dataset[i][0]).flatten() for i in range(len(dataset))])
        self.y = torch.stack([torch.tensor(dataset[i][1]) for i in range(len(dataset))])

        skf = StratifiedKFold(n_splits=n_splits)
        self.data_splits = skf.split(self.x, self.y)
        self.data_splits = [(idx[0], idx[1]) for idx in self.data_splits]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class UCIClassification(Dataset):
    """UCI classification abstract class

    :param name: string denotation for dataset to download
        as specified in uci_datasets.UCIDatasets
    :param n_splits: an int describing the number of class stratified
            splits to compute
    """

    def __init__(self, name, data_dir="/tmp/", n_splits=5):
        super(UCIClassification, self).__init__()
        dataset = UCIDatasets(name=name, data_dir=data_dir, n_splits=n_splits)
        torch_dataset = dataset.get_simple_dataset()

        self.data_dir = dataset.data_dir
        self.name = dataset.name
        self.data_splits = dataset.data_splits

        self.len_dataset = len(torch_dataset)
        self.x = torch_dataset[:][0]
        self.y = torch_dataset[:][1].squeeze()

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def remap_to_int(torch_class_array):
    """Remaps the values in the torch_class_array to integers from 0
    to n for n unique values in the torch_class_array

    :param torch_class_array: class array whose elements are to be
        mapped to contiguous ints
    """
    remapped_array = []
    tca2idx = {}
    mapping_value = 0
    for val in torch_class_array:
        val = int(val)
        if val in tca2idx.keys():
            remapped_array.append(tca2idx[val])
        else:
            tca2idx[val] = mapping_value
            mapping_value += 1
            remapped_array.append(tca2idx[val])
    return torch.Tensor(remapped_array)


class UCIGlass(UCIClassification):
    """UCI Glass dataset

    :param n_splits: an int describing the number of class stratified
            splits to compute
    """

    def __init__(self, data_dir="/tmp/", n_splits=5):
        super(UCIGlass, self).__init__(name="glass", data_dir=data_dir, n_splits=n_splits)
        self.y -= 1  # for 0 - k-1 class relabelling
        self.y = remap_to_int(self.y).long()  # UCIGlass has mislabelling


class UCIParkinsons(UCIClassification):
    """UCI Parkinsons dataset

    :param n_splits: an int describing the number of class stratified
            splits to compute
    """

    def __init__(self, data_dir="/tmp/", n_splits=5):
        super(UCIParkinsons, self).__init__(name="parkinsons", data_dir=data_dir, n_splits=n_splits)


class UCISeeds(UCIClassification):
    """UCI Seeds dataset

    :param n_splits: an int describing the number of class stratified
            splits to compute
    """

    def __init__(self, data_dir="/tmp/", n_splits=5):
        super(UCISeeds, self).__init__(name="seeds", data_dir=data_dir, n_splits=n_splits)
        self.y -= 1  # for 0 - k-1 class relabeling


class StriatumDataset(Dataset):
    """Striatum dataset as used in Konyushkova et al. 2017

    From Ksenia Konyushkova, Raphael Sznitman, Pascal Fua 'Learning Active
    Learning from Data', NIPS 2017

    :param data_dir: path where to save the raw data default to /tmp/
    :param n_splits: an int describing the number of class stratified
            splits to compute
    """

    def __init__(self, data_dir="/tmp/", n_splits=5):
        super(StriatumDataset, self).__init__()
        self.data_dir = data_dir
        self.n_splits = n_splits
        self.train_feat_url = (
            "https://github.com/ksenia-konyushkova/LAL/raw/master/data/striatum_train_features_mini.mat"
        )
        self.test_feat_url = "https://github.com/ksenia-konyushkova/LAL/raw/master/data/striatum_test_features_mini.mat"
        self.train_label_url = (
            "https://github.com/ksenia-konyushkova/LAL/raw/master/data/striatum_train_labels_mini.mat"
        )
        self.test_label_url = "https://github.com/ksenia-konyushkova/LAL/raw/master/data/striatum_test_labels_mini.mat"

        self._load_dataset()

    def _download_dataset(self, url):
        if not path.exists(self.data_dir):
            os.mkdir(self.data_dir)

        file_name = url.split("/")[-1]
        if not path.exists(self.data_dir + file_name):
            urllib.request.urlretrieve(url, self.data_dir + file_name)

    def _load_dataset(self):
        """Download, process, and get stratified splits"""

        # download
        self._download_dataset(self.train_feat_url)
        self._download_dataset(self.test_feat_url)
        self._download_dataset(self.train_label_url)
        self._download_dataset(self.test_label_url)

        # process
        train_feat = (scipy.io.loadmat(self.data_dir + "striatum_train_features_mini.mat"))["features"]
        test_feat = scipy.io.loadmat(self.data_dir + "striatum_test_features_mini.mat")["features"]
        train_label = scipy.io.loadmat(self.data_dir + "striatum_train_labels_mini.mat")["labels"]
        test_label = scipy.io.loadmat(self.data_dir + "striatum_test_labels_mini.mat")["labels"]

        self.x = np.vstack([train_feat, test_feat])
        self.y = np.vstack([train_label, test_label])

        skf = StratifiedKFold(n_splits=self.n_splits)
        self.in_dim = self.x.shape[1]
        self.out_dim = 1
        self.data_splits = skf.split(self.x, self.y)
        self.data_splits = [(idx[0], idx[1]) for idx in self.data_splits]

        self.x = torch.from_numpy(self.x).float()
        self.y = torch.from_numpy(self.y).long().squeeze()
        self.y = remap_to_int(self.y).long()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class GaussianCloudsDataset(Dataset):
    """GaussianClouds from Konyushkova et al. 2017 basically a imbalanced
    binary classification task created from multivariate gaussian blobs

    From Ksenia Konyushkova, Raphael Sznitman, Pascal Fua 'Learning Active
    Learning from Data', NIPS 2017

    :param data_dir: path where to save the raw data default to /tmp/
    :param n_splits: an int describing the number of class stratified
            splits to compute
    """

    def __init__(self, data_dir="/tmp/", n_splits=5):
        self.data_dir = data_dir
        self.n_splits = n_splits
        self._load_dataset()

    def _load_dataset(self, size=1000, n_dim=2, random_balance=False, n_splits=10):
        if random_balance:
            # proportion of class 1 to vary from 10% to 90%
            cl1_prop = np.random.rand()
            cl1_prop = (cl1_prop - 0.5) * 0.8 + 0.5
        else:
            cl1_prop = 0.8

        trainSize1 = int(size * cl1_prop)
        trainSize2 = size - trainSize1
        testSize1 = trainSize1 * 10
        testSize2 = trainSize2 * 10

        # Generate parameters of datasets
        mean1 = scipy.random.rand(n_dim)
        cov1 = scipy.random.rand(n_dim, n_dim) - 0.5
        cov1 = np.dot(cov1, cov1.transpose())
        mean2 = scipy.random.rand(n_dim)
        cov2 = scipy.random.rand(n_dim, n_dim) - 0.5
        cov2 = np.dot(cov2, cov2.transpose())

        # Training data generation
        trainX1 = np.random.multivariate_normal(mean1, cov1, trainSize1)
        trainY1 = np.ones((trainSize1, 1))
        trainX2 = np.random.multivariate_normal(mean2, cov2, trainSize2)
        trainY2 = np.zeros((trainSize2, 1))

        # Testing data generation
        testX1 = np.random.multivariate_normal(mean1, cov1, testSize1)
        testY1 = np.ones((testSize1, 1))
        testX2 = np.random.multivariate_normal(mean2, cov2, testSize2)
        testY2 = np.zeros((testSize2, 1))

        train_data = np.concatenate((trainX1, trainX2), axis=0)
        train_labels = np.concatenate((trainY1, trainY2))
        test_data = np.concatenate((testX1, testX2), axis=0)
        test_labels = np.concatenate((testY1, testY2))

        self.x = np.vstack([train_data, test_data])
        self.y = np.vstack([train_labels, test_labels]).squeeze()

        skf = StratifiedKFold(n_splits=self.n_splits)  # change to Stratified later
        self.in_dim = self.x.shape[1]
        self.out_dim = 1
        self.data_splits = skf.split(self.x, self.y)
        self.data_splits = [(idx[0], idx[1]) for idx in self.data_splits]

        self.x = torch.from_numpy(self.x).float()
        self.y = torch.from_numpy(self.y).long().squeeze()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class Checkerboard2x2Dataset(Dataset):
    """Checkerboard2x2 dataset from Konyushkova et al. 2017

    From Ksenia Konyushkova, Raphael Sznitman, Pascal Fua 'Learning Active
    Learning from Data', NIPS 2017

    :param data_dir: path where to save the raw data default to /tmp/
    :param n_splits: an int describing the number of class stratified
            splits to compute

    """

    def __init__(self, data_dir="/tmp/", n_splits=5):
        super(Checkerboard2x2Dataset, self).__init__()
        self.data_dir = data_dir
        self.n_splits = n_splits

        self.raw_train_url = "https://github.com/ksenia-konyushkova/LAL/raw/master/data/checkerboard2x2_train.npz"
        self.raw_test_url = "https://github.com/ksenia-konyushkova/LAL/raw/master/data/checkerboard2x2_test.npz"

        self._load_dataset()

    def _download_dataset(self, url):
        if not path.exists(self.data_dir):
            os.mkdir(self.data_dir)

        file_name = url.split("/")[-1]
        if not path.exists(self.data_dir + file_name):
            urllib.request.urlretrieve(url, self.data_dir + file_name)

    def _load_dataset(self):
        """Download, process, and get stratified splits"""

        # download
        self._download_dataset(self.raw_train_url)
        self._download_dataset(self.raw_test_url)

        # process
        train = np.load(self.data_dir + "checkerboard2x2_train.npz")
        test = np.load(self.data_dir + "checkerboard2x2_test.npz")

        train_feat, train_label = train["x"], train["y"]
        test_feat, test_label = test["x"], test["y"]

        self.x = np.vstack([train_feat, test_feat])
        self.y = np.vstack([train_label, test_label])

        skf = StratifiedKFold(n_splits=self.n_splits)  # change to Stratified later
        self.in_dim = self.x.shape[1]
        self.out_dim = 1
        self.data_splits = skf.split(self.x, self.y)
        self.data_splits = [(idx[0], idx[1]) for idx in self.data_splits]

        self.x = torch.from_numpy(self.x).float()
        self.y = torch.from_numpy(self.y).long().squeeze()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class Checkerboard4x4Dataset(Dataset):
    """Checkerboard 4x4 dataset from Konyushkova et al. 2017

    From Ksenia Konyushkova, Raphael Sznitman, Pascal Fua 'Learning Active
    Learning from Data', NIPS 2017

    :param data_dir: path where to save the raw data default to /tmp/
    :param n_splits: an int describing the number of class stratified
            splits to compute

    """

    def __init__(self, data_dir="/tmp/", n_splits=5):
        super(Checkerboard4x4Dataset, self).__init__()
        self.data_dir = data_dir
        self.n_splits = n_splits

        self.train_url = "https://github.com/ksenia-konyushkova/LAL/raw/master/data/checkerboard4x4_train.npz"
        self.test_url = "https://github.com/ksenia-konyushkova/LAL/raw/master/data/checkerboard4x4_test.npz"

        self._load_dataset()

    def _download_dataset(self, url):
        if not path.exists(self.data_dir):
            os.mkdir(self.data_dir)

        file_name = url.split("/")[-1]
        if not path.exists(self.data_dir + file_name):
            urllib.request.urlretrieve(url, self.data_dir + file_name)

    def _load_dataset(self):
        """Download, process, and get stratified splits"""

        # download
        self._download_dataset(self.train_url)
        self._download_dataset(self.test_url)

        # process
        train = np.load(self.data_dir + "checkerboard4x4_train.npz")
        test = np.load(self.data_dir + "checkerboard4x4_test.npz")

        train_feat, train_label = train["x"], train["y"]
        test_feat, test_label = test["x"], test["y"]

        self.x = np.vstack([train_feat, test_feat])
        self.y = np.vstack([train_label, test_label])

        skf = StratifiedKFold(n_splits=self.n_splits)  # change to Stratified later
        self.in_dim = self.x.shape[1]
        self.out_dim = 1
        self.data_splits = skf.split(self.x, self.y)
        self.data_splits = [(idx[0], idx[1]) for idx in self.data_splits]

        self.x = torch.from_numpy(self.x).float()
        self.y = torch.from_numpy(self.y).long().squeeze()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class CreditCardDataset(Dataset):
    """Credit card fraud dataset, highly unbalanced and challenging.

    From Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, and Gianluca Bontempi.
    Calibrating probability with undersampling for unbalanced classification. In 2015
    IEEE Symposium Series on Computational Intelligence, pages 159–166, 2015.

    We use the original data from http://www.ulb.ac.be/di/map/adalpozz/data/creditcard.Rdata
    processed using pyreadr

    :param data_dir: path where to save the raw data default to /tmp/
    :param n_splits: an int describing the number of class stratified
            splits to compute

    """

    def __init__(self, data_dir="/tmp/", n_splits=5):
        super(CreditCardDataset, self).__init__()
        self.raw_url = "http://www.ulb.ac.be/di/map/adalpozz/data/creditcard.Rdata"
        self.data_dir = data_dir
        self.n_splits = n_splits

        self._load_dataset()

    def _load_dataset(self):
        if not path.exists(self.data_dir):
            os.mkdir(self.data_dir)

        url = self.raw_url
        file_name = url.split("/")[-1]
        if not path.exists(self.data_dir + file_name):
            urllib.request.urlretrieve(self.raw_url, self.data_dir + file_name)

        data = pyreadr.read_r(self.data_dir + file_name)
        data = data["creditcard"]
        data.reset_index(inplace=True)
        self.df = data
        cols = data.columns
        xcols = cols[1:-1]
        ycol = "Class"
        x = data[xcols].to_numpy()
        y = data[ycol].to_numpy()
        _, y = np.unique(y, return_inverse=True)  # map string classes to ints
        self.x = x
        self.y = y

        skf = StratifiedKFold(n_splits=self.n_splits)  # change to Stratified later
        self.in_dim = len(xcols)
        self.out_dim = 1
        self.data_splits = skf.split(x, y)
        self.data_splits = [(idx[0], idx[1]) for idx in self.data_splits]

        self.x = torch.from_numpy(self.x).float()
        self.y = torch.from_numpy(self.y).long().squeeze()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]