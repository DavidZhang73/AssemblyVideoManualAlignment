import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models import make_resnet50, make_slowfast
from src.modules.assembly_video_manual_alignment_module import AssemblyVideoManualAlignmentModule


class CLIPModule(AssemblyVideoManualAlignmentModule):
    def __init__(
        self,
        video_encoder_pretrained: bool = True,
        video_encoder_freeze: bool = True,
        image_encoder_pretrained: bool = True,
        image_encoder_freeze: bool = True,
        feature_dim: int = 1024,
        losses: tuple[str, ...] = ("batch_step", "batch_page"),
        sprf: bool = True,
        ot_gamma: float = 4.0,
        ot_power: float = 7.0,
        save_predictions: bool = True,
        save_video_features: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.video_encoder = make_slowfast(
            pretrained=video_encoder_pretrained, freeze=video_encoder_freeze, keep_last_layer=False
        )
        video_encoder_last_layer_dim = 2304 + 2 if sprf else 2304
        self.video_encoder_head = nn.Linear(
            in_features=video_encoder_last_layer_dim, out_features=feature_dim, bias=True
        )
        self.image_encoder = make_resnet50(
            pretrained=image_encoder_pretrained, freeze=image_encoder_freeze, keep_last_layer=False
        )
        image_encoder_last_layer_dim = 2048 + 2 if sprf else 2048
        self.image_encoder_head = nn.Linear(
            in_features=image_encoder_last_layer_dim, out_features=feature_dim, bias=True
        )

        if "batch_step" in losses:
            self.batch_step_loss_tau = nn.Parameter(torch.log(torch.tensor([1 / 0.07])))
        if "batch_page" in losses:
            self.batch_page_loss_tau = nn.Parameter(torch.log(torch.tensor([1 / 0.07])))

        self.ot_power = ot_power
        self.ot_gamma = ot_gamma

        self.validation_outputs = []
        self.test_outputs = []

    def get_batch_loss(self, batch_video_logits, ground_truth, stage, loss_name):
        batch_size = len(batch_video_logits)
        batch_video_loss = F.cross_entropy(batch_video_logits, ground_truth)
        batch_video_loss_t = F.cross_entropy(batch_video_logits.t(), ground_truth)
        batch_loss = (batch_video_loss + batch_video_loss_t) / 2
        self.log(f"{stage}/loss/batch/video_{loss_name}", batch_video_loss, batch_size=batch_size)
        self.log(f"{stage}/loss/batch/{loss_name}_video", batch_video_loss_t, batch_size=batch_size)
        self.log(f"{stage}/loss/batch/{loss_name}", batch_loss, batch_size=batch_size)
        return batch_loss

    def train_forward(self, batch):
        # video features
        video_features = F.normalize(self.video_encoder(batch["video"]))
        if self.hparams.sprf:
            pf = batch["video_position_feature"].unsqueeze(1).float()
            video_features = F.normalize(
                torch.cat([video_features, torch.sin(torch.pi * pf), torch.cos(torch.pi * pf)], dim=1)
            )
        video_features = F.normalize(self.video_encoder_head(video_features))

        # step features
        batch_step_features = F.normalize(self.image_encoder(batch["step_image"]))
        if self.hparams.sprf:
            batch_step_pf = batch["step_position_feature"].unsqueeze(1).float()
            batch_step_features = F.normalize(
                torch.cat(
                    [batch_step_features, torch.sin(torch.pi * batch_step_pf), torch.cos(torch.pi * batch_step_pf)],
                    dim=1,
                )
            )
        batch_step_features = F.normalize(self.image_encoder_head(batch_step_features))

        # page features
        batch_page_features = F.normalize(self.image_encoder(batch["page_image"]))
        if self.hparams.sprf:
            batch_page_pf = batch["page_position_feature"].unsqueeze(1).float()
            batch_page_features = F.normalize(
                torch.cat(
                    [batch_page_features, torch.sin(torch.pi * batch_page_pf), torch.cos(torch.pi * batch_page_pf)],
                    dim=1,
                )
            )
        batch_page_features = F.normalize(self.image_encoder_head(batch_page_features))

        return video_features, batch_step_features, batch_page_features

    def training_step(self, batch, batch_idx):
        stage = "train"
        batch_size = batch["video"][0].shape[0]
        video_features, batch_step_features, batch_page_features = self.train_forward(
            batch
        )

        loss = torch.zeros(1, device=video_features.device)

        # batch step
        if "batch_step" in self.hparams.losses:
            batch_video_step_logits = video_features @ batch_step_features.t()
            batch_video_step_ground_truth = torch.arange(batch_size, device=video_features.device)
            loss += self.get_batch_loss(
                self.batch_step_loss_tau.exp() * batch_video_step_logits, batch_video_step_ground_truth, stage, "step"
            )

        # batch page
        if "batch_page" in self.hparams.losses:
            batch_video_page_logits = video_features @ batch_page_features.t()
            batch_video_page_ground_truth = torch.arange(batch_size, device=video_features.device)
            loss += self.get_batch_loss(
                self.batch_page_loss_tau.exp() * batch_video_page_logits, batch_video_page_ground_truth, stage, "page"
            )

        # log tau
        with torch.no_grad():
            if "batch_step" in self.hparams.losses:
                self.log(f"{stage}/tau/batch_step", 1 / self.batch_step_loss_tau.exp(), batch_size=batch_size)
            if "batch_page" in self.hparams.losses:
                self.log(f"{stage}/tau/batch_page", 1 / self.batch_page_loss_tau.exp(), batch_size=batch_size)
        # loss
        loss /= len([name for name in self.hparams.losses])
        self.log(f"{stage}/loss", loss, batch_size=batch_size)
        return loss
