"""
Video to Diagram Retrieval metrics
"""

from functools import partial

import torch
from torch import Tensor
from torchmetrics import Metric, MetricCollection
from torchmetrics.functional import kendall_rank_corrcoef


class Top1Accuracy(Metric):
    """
    Top-1 Accuracy
    """

    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        assert preds.shape[0] == target.shape[0], "preds and target should have the same batch_size"
        assert len(preds.shape) == 2, "preds should be of shape (batch_size, num_classes)"
        assert len(target.shape) == 1, "target should be of shape (batch_size,)"
        preds = torch.argmax(preds, dim=1)
        self.correct += torch.sum(preds == target)
        self.total += len(target)

    def compute(self) -> Tensor:
        return self.correct.float() / self.total


class AverageIndexError(Metric):
    """
    Average Index Error (AIE)
    """

    def __init__(self):
        super().__init__()
        self.add_state("index_error", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        assert preds.shape[0] == target.shape[0], "preds and target should have the same batch_size"
        assert len(preds.shape) == 2, "preds should be of shape (batch_size, num_classes)"
        assert len(target.shape) == 1, "target should be of shape (batch_size,)"
        preds = torch.argmax(preds, dim=1)
        self.index_error += torch.sum(torch.abs(preds - target))
        self.total += len(target)

    def compute(self) -> Tensor:
        return self.index_error.float() / self.total


class AverageKendallRankCorrCoef(Metric):
    """
    Average Kendall Rank Correlation Coefficient
    """

    def __init__(self):
        super().__init__()
        self.add_state("kendall", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        assert preds.shape[0] == target.shape[0], "preds and target should have the same batch_size"
        assert len(preds.shape) == 2, "preds should be of shape (batch_size, num_classes)"
        assert len(target.shape) == 1, "target should be of shape (batch_size,)"
        preds = torch.argmax(preds, dim=1)
        kendall = kendall_rank_corrcoef(preds, target)
        if not torch.isnan(kendall):
            self.kendall += kendall
            self.total += 1

    def compute(self) -> Tensor:
        return self.kendall / self.total


VideoToDiagramMetric = partial(
    MetricCollection,
    {"accuracy/top1": Top1Accuracy(), "index_error": AverageIndexError(), "kendall": AverageKendallRankCorrCoef()},
)
