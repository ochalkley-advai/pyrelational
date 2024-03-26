"""
This module contains methods for scoring samples based on model uncertainty in
object detection tasks

This module contains functions for computing the informativeness values
of a given probability distribution (outputs of a model/mc-dropout
prediction, etc.)
"""

import math

import torch
from torch import Tensor


def object_detection_least_confidence(
    prob_dist: Tensor, axis: int = -1
) -> Tensor:
    r"""Returns the informativeness score of an array using least confidence
    sampling in a 0-1 range where 1 is the most uncertain

    The least confidence uncertainty is the normalised difference between
    the most confident prediction and 100 percent confidence

    :param prob_dist: real number tensor whose elements add to 1.0 along an axis
    :param axis: axis of prob_dist where probabilities add to 1

    :return: tensor with normalised least confidence scores
    """
    assert torch.allclose(
        prob_dist.sum(axis), torch.tensor(1.0)
    ), "input should be probability distributions along specified axis"

    simple_least_conf, _ = torch.max(prob_dist, dim=axis)
    num_labels = prob_dist.size(axis)
    normalized_least_conf: Tensor = (1 - simple_least_conf) * (
        num_labels / (num_labels - 1)
    )
    return normalized_least_conf


def object_detection_margin_confidence(
    prob_dist: Tensor, axis: int = -1
) -> Tensor:
    r"""Returns the informativeness score of a probability distribution using
    margin of confidence sampling in a 0-1 range where 1 is the most uncertain
    The margin confidence uncertainty is the difference between the top two
    most confident predictions

    :param prob_dist: real number tensor whose elements add to 1.0 along an axis
    :param axis: axis of prob_dist where probabilities add to 1

    :return: tensor with margin confidence scores
    """
    assert torch.allclose(
        prob_dist.sum(axis), torch.tensor(1.0)
    ), "input should be probability distributions along specified axis"

    prob_dist, _ = torch.sort(prob_dist, descending=True, dim=axis)
    difference = prob_dist.select(axis, 0) - prob_dist.select(axis, 1)
    margin_conf: Tensor = 1 - difference
    return margin_conf


def object_detection_ratio_confidence(
    prob_dist: Tensor, axis: int = -1
) -> Tensor:
    r"""Returns the informativeness score of a probability distribution using
    ratio of confidence sampling in a 0-1 range where 1 is the most uncertain
    The ratio confidence uncertainty is the ratio between the top two most
    confident predictions

    :param prob_dist: real number tensor whose elements add to 1.0 along an axis
    :param axis: axis of prob_dist where probabilities add to 1

    :return: tensor of ratio confidence uncertainties
    """
    assert torch.allclose(
        prob_dist.sum(axis), torch.tensor(1.0)
    ), "input should be probability distributions along specified axis"

    prob_dist, _ = torch.sort(
        prob_dist, descending=True, dim=axis
    )  # sort probs so largest is first
    ratio_conf: Tensor = prob_dist.select(axis, 1) / (
        prob_dist.select(axis, 0)
    )  # ratio between top two props
    return ratio_conf


def object_detection_entropy(prob_dist: Tensor, axis: int = -1) -> Tensor:
    r"""Returns the informativeness score of a probability distribution
    using entropy

    The entropy based uncertainty is defined as

    :math:`- \frac{1}{\log(n)} \sum_{i}^{n} p_i \log (p_i)`

    :param prob_dist: real number tensor whose elements add to 1.0 along an axis
    :param axis: axis of prob_dist where probabilities add to 1

    :return: tensor of entropy based uncertainties
    """
    assert torch.allclose(
        prob_dist.sum(axis), torch.tensor(1.0)
    ), "input should be probability distributions along specified axis"

    log_probs = prob_dist * torch.log2(prob_dist)
    raw_entropy = 0 - torch.sum(log_probs, dim=axis)
    normalised_entropy: Tensor = raw_entropy / math.log2(prob_dist.size(axis))
    return normalised_entropy


def object_detection_bald(prob_dist: Tensor) -> Tensor:
    """
    Implementation of Bayesian Active Learning by Disagreement (BALD) for object detection task

    `reference <https://arxiv.org/pdf/1112.5745.pdf>`__
    :param x: 3D pytorch Tensor of shape n_estimators x n_samples x n_classes
    :return: 1D pytorch tensor of scores
    """

    assert torch.allclose(
        prob_dist.sum(-1), torch.tensor(1.0)
    ), "input should be probability distributions along specified axis"

    return object_detection_entropy(
        prob_dist.mean(0), -1
    ) - object_detection_entropy(prob_dist, -1).mean(0)
