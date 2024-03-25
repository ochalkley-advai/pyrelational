from abc import ABC, abstractmethod
from typing import Any, List

import torch
from pyrelational.data_managers import DataManager
from pyrelational.informativeness import softmax
from pyrelational.model_managers import ModelManager
from pyrelational.strategies.abstract_strategy import Strategy
from torch import Tensor


class ObjectDetectionStrategy(Strategy, ABC):
    """
    A base active learning strategy class for object detection in which the top n indices,
    according to user-specified scoring function, are queried at each iteration.
    """

    def __init__(self, aggregation_type: str = "max") -> None:
        self.aggregation_type = aggregation_type
        super(ObjectDetectionStrategy, self).__init__()

    @abstractmethod
    def scoring_function(
        self, predictions: Tensor, aggregation_type: str = "max"
    ) -> Tensor:
        """
        Compute score of each sample.

        :param predictions: model predictions for each sample.
        :param aggregation_type: function to handle with many uncetrainty values in an image.
        :return: scores for each sample
        """

    def __call__(
        self,
        num_annotate: int,
        data_manager: DataManager,
        model_manager: ModelManager[Any, Any],
    ) -> List[int]:
        """
        Call function which identifies samples which need to be labelled based on
        user defined scoring function.

        :param num_annotate: number of samples to annotate
        :param data_manager: A pyrelational data manager
            which keeps track of what has been labelled and creates data loaders for
            active learning
        :param model_manager: A pyrelational model manager
            which wraps a user defined ML model to handle instantiation, training, testing,
            as well as uncertainty quantification

        :return: list of indices to annotate
        """
        output = self.train_and_infer(
            data_manager=data_manager, model_manager=model_manager
        )
        if not torch.allclose(output.sum(1), torch.tensor(1.0)):
            output = softmax(output)

        uncertainty = self.compute_total_uncertainty(
            output, self.aggregation_type
        )
        print("uncertainty:", uncertainty)
        ixs = torch.argsort(
            torch.Tensor(uncertainty), descending=True
        ).tolist()
        return [data_manager.u_indices[i] for i in ixs[:num_annotate]]

    def compute_total_uncertainty(
        self, output: list[list[torch.Tensor]], aggregation_type: str
    ):
        """Computes the total uncertainty of the whole dataset.

        :param output: Output of the model.
        :param aggregation_type: function to handle with many uncetrainty values in an image.
        :return: total uncertainty of the dataset.
        """
        total_uncertainty = []
        for batch_conf_scores in output:
            for img_conf_scores in batch_conf_scores:
                img_uncertainty = []
                for bbox_conf_scores in img_conf_scores:
                    bbox_uncertainty = self.scoring_function(
                        bbox_conf_scores,
                        aggregation_type=self.aggregation_type,
                    )
                    img_uncertainty.append(bbox_uncertainty)

            total_uncertainty.append(
                self.aggregate_image_uncertainty(
                    img_uncertainty, self.aggregation_type
                )
            )
        return total_uncertainty

    def aggregate_image_uncertainty(
        img_uncertainty: list, aggregation_type: str
    ):
        """Helper function that computes a value metric to deal with uncetainties in an image,
        since each image has differnet number of bounding boxes.

        :param img_uncertainty: list containing the uncertainty of each bounding box in an image.
        :param aggregation_type: function to handle with many uncetrainty values in an image.

        :return: Final uncertainty score for the whole image.
        """
        if len(img_uncertainty) == 0:
            final_uncertainty_score = 0
        else:
            img_entropy_torch = torch.stack(img_uncertainty)
            if aggregation_type == "max":
                final_uncertainty_score = max(img_uncertainty)
            elif aggregation_type == "L2":
                final_uncertainty_score = torch.norm(img_entropy_torch)
            else:
                raise ValueError(
                    f'Aggregation type "{aggregation_type}" not recognise'
                )
        return final_uncertainty_score
