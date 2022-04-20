import torch
import torch.distributions as distributions
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from sklearn.datasets import load_digits, load_breast_cancer, load_diabetes
import pyreadr
import os
from os import path
import urllib.request
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np

import scipy.io


class SynthClass1(Dataset):
    """
    Synth1 dataset as described in Yang and Loog

    Consists of a binary classification task of positive
    and negative class samples being generated by a multivariate
    gaussian distribution centered at [1,1] and [-1,-1]
    respectively.
    """

    def __init__(self, n_splits=5, size=500, random_seed=1234):
        super(SynthClass1, self).__init__()
        self.n_splits = n_splits
        pos_distribution = distributions.MultivariateNormal(torch.FloatTensor([3,3]), torch.eye(2))
        neg_distribution = distributions.MultivariateNormal(torch.FloatTensor([0,0]), torch.eye(2))
        
        num_pos = int(size/2.0)
        num_neg = size - num_pos    
        pos_samples = torch.vstack([pos_distribution.sample() for _ in range(num_pos)])
        neg_samples = torch.vstack([neg_distribution.sample() for _ in range(num_neg)])
        pos_targets = torch.ones(num_pos, dtype=torch.long)
        neg_targets = torch.ones(num_neg, dtype=torch.long)*0

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
    """

    def __init__(self, n_splits=5, size=500, random_seed=1234):
        super(SynthClass2, self).__init__()
        self.n_splits = n_splits

        pos_dist_1 = distributions.MultivariateNormal(torch.FloatTensor([0,5]), torch.eye(2))
        neg_dist_1 = distributions.MultivariateNormal(torch.FloatTensor([0,-5]), torch.eye(2))

        pos_dist_2 = distributions.MultivariateNormal(torch.FloatTensor([-5, 10]), torch.eye(2))
        pos_dist_3 = distributions.MultivariateNormal(torch.FloatTensor([-5, -10]), torch.eye(2))

        neg_dist_2 = distributions.MultivariateNormal(torch.FloatTensor([5, 10]), torch.eye(2))
        neg_dist_3 = distributions.MultivariateNormal(torch.FloatTensor([5, -10]), torch.eye(2))

        num_pos = int(size/2.0)
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
        neg_targets = torch.ones(num_neg, dtype=torch.long)*0

        self.x = torch.cat([pos_samples_1, pos_samples_2, pos_samples_3, neg_samples_1, neg_samples_2, neg_samples_3])
        self.y = torch.cat([pos_targets, neg_targets])

        skf = StratifiedKFold(n_splits=n_splits)
        self.data_splits = skf.split(self.x, self.y)
        self.data_splits = [(idx[0], idx[1]) for idx in self.data_splits]

    def _split(self, iterable, n):
        # split the iterable into n approximately same size parts
        k, m = divmod(len(iterable), n)
        return (iterable[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class SynthClass3(Dataset):
    """SynthClass3 dataset as described in Yang and Loog"""
    def __init__(self, n_splits=5, size=500, random_seed=1234):
        super(SynthClass3, self).__init__()
        self.size = size
        self.random_seed = random_seed
        self.n_splits = n_splits

        pos_dist_1 = distributions.MultivariateNormal(torch.FloatTensor([0, 0]), torch.FloatTensor([[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]))
        pos_dist_2 = distributions.MultivariateNormal(torch.FloatTensor([3, 10]), torch.FloatTensor([[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]))

        neg_dist_1 = distributions.MultivariateNormal(torch.FloatTensor([3, 3]), torch.FloatTensor([[1, 2], [2, 7]]))

        num_pos = int(size/2.0)
        num_neg1 = size - num_pos
        num_pos1, num_pos2 = [len(x) for x in self._split(range(num_pos), 2)]

        pos_samples_1 = torch.vstack([pos_dist_1.sample() for _ in range(num_pos1)])
        pos_samples_2 = torch.vstack([pos_dist_2.sample() for _ in range(num_pos2)])
        neg_samples_1 = torch.vstack([neg_dist_1.sample() for _ in range(num_neg1)])

        pos_targets = torch.ones(num_pos, dtype=torch.long)
        neg_targets = torch.ones(num_neg1, dtype=torch.long)*0

        self.x = torch.cat([pos_samples_1, pos_samples_2, neg_samples_1])
        self.y = torch.cat([pos_targets, neg_targets])

        skf = StratifiedKFold(n_splits=n_splits)
        self.data_splits = skf.split(self.x, self.y)
        self.data_splits = [(idx[0], idx[1]) for idx in self.data_splits]

    def _split(self, iterable, n):
        # split the iterable into n approximately same size parts
        k, m = divmod(len(iterable), n)
        return (iterable[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]