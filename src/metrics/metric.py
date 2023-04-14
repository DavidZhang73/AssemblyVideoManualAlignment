import os
import re
from collections import defaultdict

import torch
from dtw import dtw
from pytorch_lightning import LightningModule
from torchmetrics import MetricCollection

from src.metrics.diagram_to_video import DiagramToVideoMetric
from src.metrics.video_to_diagram import VideoToDiagramMetric
from src.models.optimal_transport import sinkhorn


@torch.no_grad()
def calculate_aggregated_accuracy(self: LightningModule, stage):
    if stage == "val":
        outputs = self.validation_outputs
    elif stage == "test":
        outputs = self.test_outputs
    else:
        raise ValueError(f"Invalid stage: {stage}")
    device = outputs[0]["step_index"].device

    segment_to_step_logits = {}
    segment_to_page_logits = {}
    video_to_video_features = {}

    # Load data from batch
    for batch in outputs:
        video_step_logits = batch["video_step_logits"]
        video_page_logits = batch["video_page_logits"]

        # To be compatible with both list and tensor container
        if type(video_step_logits) != list:
            video_step_logits = [item.unsqueeze(dim=0) for item in video_step_logits]
        if type(video_page_logits) != list:
            video_page_logits = [item.unsqueeze(dim=0) for item in video_page_logits]

        furniture_id = batch["furniture_id"]
        video_id = batch["video_id"]
        step_index = batch["step_index"]
        page_index = batch["page_index"]
        clip_index = batch["clip_index"]
        batch_size = step_index.shape[0]

        # Aggregate logits by segment, each logit is in shape (1, M), where M is the number of steps/pages.
        # Note that when random sampling, there is only one clip per segment, the clip_index is always 0.
        # When using constant-5 sampling, there are 5 clips per segment, the clip_index can be 0, 1, 2, 3, 4.
        for i in range(batch_size):
            segment_key = (
                furniture_id[i],
                video_id[i],
                step_index[i].item(),
                page_index[i].item(),
                clip_index[i].item(),
            )
            if segment_key in segment_to_step_logits:
                segment_to_step_logits[segment_key] += video_step_logits[i]
            else:
                segment_to_step_logits[segment_key] = video_step_logits[i].clone()
            if segment_key in segment_to_page_logits:
                segment_to_page_logits[segment_key] += video_page_logits[i]
            else:
                segment_to_page_logits[segment_key] = video_page_logits[i].clone()

            if stage == "test":
                video_features = batch["video_features"]
                if segment_key in video_to_video_features:
                    video_to_video_features[segment_key].append(video_features[i])
                else:
                    video_to_video_features[segment_key] = [video_features[i]]

    # Aggregate logits by video, each logit is in shape (N, M),
    # where N is the number of segments in the video, M is the number of steps/pages.
    video_to_step_logits = defaultdict(list)
    video_to_step_target = defaultdict(list)
    for segment_key, step_logits in segment_to_step_logits.items():
        video_key = segment_key[:2]
        target = segment_key[2]
        video_to_step_logits[video_key].append(step_logits)
        video_to_step_target[video_key].append(target)

    video_to_page_logits = defaultdict(list)
    video_to_page_target = defaultdict(list)
    for segment_key, page_logits in segment_to_page_logits.items():
        video_key = segment_key[:2]
        target = segment_key[3]
        video_to_page_logits[video_key].append(page_logits)
        video_to_page_target[video_key].append(target)

    # Save predictions and video features
    if stage == "test" and self.trainer.ckpt_path:
        ckpt_pathname_reg = re.compile(r"(.+)/checkpoints/(.+)\.ckpt")
        epoch = self.trainer.current_epoch
        path, _ = ckpt_pathname_reg.findall(self.trainer.ckpt_path)[0]
        if self.hparams.save_predictions:
            torch.save(
                dict(
                    video_to_step_logits=video_to_step_logits,
                    video_to_step_target=video_to_step_target,
                    video_to_page_logits=video_to_page_logits,
                    video_to_page_target=video_to_page_target,
                ),
                os.path.join(path, f"epoch_{epoch}_predictions.pt"),
            )
        if self.hparams.save_video_features:
            torch.save(
                video_to_video_features,
                os.path.join(path, f"epoch_{epoch}_video_features.pt"),
            )

    # Define metrics
    video_to_diagram_metric = VideoToDiagramMetric()
    diagram_to_video_metric = DiagramToVideoMetric()
    step_metric = MetricCollection(
        [video_to_diagram_metric, diagram_to_video_metric],
        prefix=f"{stage}/",
        postfix="/video/step",
    ).to(device)
    step_metric.reset()
    ot_step_metric = step_metric.clone(postfix="/ot/video/step")
    dtw_step_metric = step_metric.clone(postfix="/dtw/video/step")
    page_metric = step_metric.clone(postfix="/video/page")
    ot_page_metric = step_metric.clone(postfix="/ot/video/page")
    dtw_page_metric = step_metric.clone(postfix="/dtw/video/page")

    def _optimal_transport(preds):
        preds = torch.pow(preds, self.hparams.ot_power)
        preds -= preds.min(dim=-1, keepdim=True).values
        preds /= preds.max(dim=-1, keepdim=True).values
        return sinkhorn(1 - preds.unsqueeze(0), gamma=self.hparams.ot_gamma).squeeze(0)

    def _dynamic_time_warping(preds):
        preds -= preds.min(dim=-1, keepdim=True).values
        preds /= preds.max(dim=-1, keepdim=True).values
        dtw_preds = torch.zeros_like(preds)
        dtw_alignment = dtw((1 - preds).cpu().to(torch.double), open_end=True)
        for i, index1 in enumerate(dtw_alignment.index1):
            index2 = dtw_alignment.index2[i]
            dtw_preds[index1, index2] = 1
        trimmed_preds = torch.zeros_like(preds)
        for row_i, row in enumerate(dtw_preds):
            trimmed_preds[row_i, (row * preds[row_i]).argmax()] = 1
        return trimmed_preds

    # Compute all metrics and log them
    for video_key, step_logits in video_to_step_logits.items():
        preds = torch.cat(step_logits, dim=0)
        target = torch.tensor(video_to_step_target[video_key], device=preds.device)
        step_metric.update(preds, target)
        ot_step_metric.update(_optimal_transport(preds), target)
        dtw_step_metric.update(_dynamic_time_warping(preds), target)
    self.log_dict(step_metric.compute())
    self.log_dict(ot_step_metric.compute())
    self.log_dict(dtw_step_metric.compute())

    for video_key, page_logits in video_to_page_logits.items():
        preds = torch.cat(page_logits, dim=0)
        target = torch.tensor(video_to_page_target[video_key], device=preds.device)
        page_metric.update(preds, target)
        ot_page_metric.update(_optimal_transport(preds), target)
        dtw_page_metric.update(_dynamic_time_warping(preds), target)
    self.log_dict(page_metric.compute())
    self.log_dict(ot_page_metric.compute())
    self.log_dict(dtw_page_metric.compute())
