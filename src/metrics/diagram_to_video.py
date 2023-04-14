"""
Diagram to Video Retrieval Metrics

Recall @ 1
Recall @ 3
AUROC
"""

from functools import partial

import torch
from torch import Tensor
from torchmetrics import Metric, MetricCollection
from torchmetrics.functional import auroc, recall


class RecallAtK(Metric):
    """
    Recall @ K
    """

    def __init__(self, k: int = 1):
        super().__init__()
        self.k = k
        self.add_state("recall", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        assert preds.shape[0] == target.shape[0], "preds and target should have the same batch_size"
        assert len(preds.shape) == 2, "preds should be of shape (batch_size, num_classes)"
        assert len(target.shape) == 1, "target should be of shape (batch_size,)"
        self.recall += recall(
            preds,
            target,
            task="multiclass",
            num_classes=preds.shape[-1],
            average="macro",
            top_k=min(self.k, preds.shape[-1]),
        )
        self.total += 1

    def compute(self) -> Tensor:
        return self.recall / self.total


class AUROC(Metric):
    """
    AUROC
    """

    def __init__(self):
        super().__init__()
        self.add_state("auroc", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        assert preds.shape[0] == target.shape[0], "preds and target should have the same batch_size"
        assert len(preds.shape) == 2, "preds should be of shape (batch_size, num_classes)"
        assert len(target.shape) == 1, "target should be of shape (batch_size,)"
        self.auroc += auroc(preds, target, task="multiclass", num_classes=preds.shape[-1], average="macro")
        self.total += 1

    def compute(self) -> Tensor:
        return self.auroc / self.total


DiagramToVideoMetric = partial(
    MetricCollection, {"recall@1": RecallAtK(k=1), "recall@3": RecallAtK(k=3), "AUROC": AUROC()}
)
