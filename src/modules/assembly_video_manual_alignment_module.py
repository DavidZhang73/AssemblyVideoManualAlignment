import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from src.metrics import calculate_aggregated_accuracy
from src.models import make_resnet50, make_slowfast


class AssemblyVideoManualAlignmentModule(pl.LightningModule):
    def __init__(
        self,
        video_encoder_pretrained: bool = True,
        video_encoder_freeze: bool = True,
        image_encoder_pretrained: bool = True,
        image_encoder_freeze: bool = True,
        feature_dim: int = 1024,
        losses: tuple[str, ...] = ("batch_step", "batch_page", "step", "page", "step_intra", "page_intra"),
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
        if "step" in losses:
            self.step_loss_tau = nn.Parameter(torch.log(torch.tensor([1 / 0.07])))
            if "step_intra" in losses:
                self.step_intra_loss_tau = nn.Parameter(torch.log(torch.tensor([1 / 0.07])))
                self.step_intra_loss_sigma = nn.Parameter(torch.tensor([2.0]))
        if "page" in losses:
            self.page_loss_tau = nn.Parameter(torch.log(torch.tensor([1 / 0.07])))
            if "page_intra" in losses:
                self.page_intra_loss_tau = nn.Parameter(torch.log(torch.tensor([1 / 0.07])))
                self.page_intra_loss_sigma = nn.Parameter(torch.tensor([0.5]))
        self.ot_power = ot_power
        self.ot_gamma = ot_gamma

        self.validation_outputs = []
        self.test_outputs = []

    def append_sprf(self, feature):
        length = feature.shape[0]
        pf = torch.arange(length, device=feature.device).unsqueeze(1).float() / length
        return torch.cat([feature, torch.sin(torch.pi * pf), torch.cos(torch.pi * pf)], dim=1)

    def get_batch_ground_truth(self, batch_key_list):
        batch_size = len(batch_key_list[0])
        key_to_index_map = {}
        ret = torch.eye(batch_size, device=batch_key_list[-1].device)
        duplication_count = 0
        for i in range(batch_size):
            batch_key = tuple(
                batch_key_list[j][i] if type(batch_key_list[j][i]) == str else int(batch_key_list[j][i])
                for j in range(len(batch_key_list))
            )
            if batch_key in key_to_index_map:
                for j in key_to_index_map[batch_key]:
                    ret[i][j] = 1
                    ret[j][i] = 1
                    duplication_count += 1
                key_to_index_map[batch_key].append(i)
            else:
                key_to_index_map[batch_key] = [i]
        self.log("train/batch_duplication/count", float(duplication_count), batch_size=batch_size)
        self.log("train/batch_duplication/rate", duplication_count / batch_size, batch_size=batch_size)
        return ret

    def js_div(self, input: torch.tensor, target: torch.tensor):
        input, target = input.view(-1, input.size(-1)), target.view(-1, target.size(-1))
        m = (0.5 * (input + target)).log()
        return 0.5 * (
            F.kl_div(m, input.log(), reduction="batchmean", log_target=True)
            + F.kl_div(m, target.log(), reduction="batchmean", log_target=True)
        )

    def get_batch_loss(self, batch_video_logits, ground_truth, stage, loss_name):
        batch_size = len(batch_video_logits)
        ground_truth = F.softmax(ground_truth * 10, dim=1)
        batch_video_loss = self.js_div(
            F.softmax(batch_video_logits, dim=1),
            ground_truth,
        )
        batch_video_loss_t = self.js_div(
            F.softmax(batch_video_logits.t(), dim=1),
            ground_truth,
        )
        batch_loss = (batch_video_loss + batch_video_loss_t) / 2
        self.log(f"{stage}/loss/batch/video_{loss_name}", batch_video_loss, batch_size=batch_size)
        self.log(f"{stage}/loss/batch/{loss_name}_video", batch_video_loss_t, batch_size=batch_size)
        self.log(f"{stage}/loss/batch/{loss_name}", batch_loss, batch_size=batch_size)
        return batch_loss

    def get_one_to_many_logits(self, features, features_list):
        ret = []
        for i, feature in enumerate(features):
            ret.append(feature.unsqueeze(0) @ features_list[i].t())
        return ret

    def get_one_to_many_loss(self, logits, tau, ground_truth_indexes, stage, loss_name):
        batch_size = len(logits)
        loss = 0
        sum_k = 0
        for i, ground_truth in enumerate(ground_truth_indexes):
            k = len(logits[i])
            sum_k += k
            loss += F.cross_entropy(tau * logits[i], ground_truth.unsqueeze(0)) * k
        loss /= sum_k
        self.log(f"{stage}/loss/{loss_name}", loss, batch_size=batch_size)
        return loss

    def _get_image_list_inter_nd_js_loss(self, image_features, tau, sigma):
        logits = image_features @ image_features.t()
        ground_truth = torch.empty((len(image_features), len(image_features)), device=logits.device)
        normal = Normal(0, sigma)
        for i in range(len(image_features)):
            ground_truth[i] = normal.log_prob(torch.arange(0 - i, len(image_features) - i, device=logits.device)).exp()
        return self.js_div(F.softmax(logits * tau, dim=1), F.softmax(ground_truth, dim=1))

    def get_image_list_intra_loss(self, image_features_list, tau, sigma, stage, loss_name):
        batch_size = len(image_features_list)
        loss = 0
        sum_k = 0
        for i, image_features in enumerate(image_features_list):
            k = len(image_features)
            sum_k += k
            loss += self._get_image_list_inter_nd_js_loss(image_features, tau, sigma) * k
        loss /= sum_k
        self.log(f"{stage}/loss/{loss_name}", loss, batch_size=batch_size)
        return loss

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
        step_features = []
        batch_step_features = []
        if "step_image_list" in batch:
            for i, step_images in enumerate(batch["step_image_list"]):
                step_feature = F.normalize(self.image_encoder(step_images))
                if self.hparams.sprf:
                    step_feature = F.normalize(self.append_sprf(step_feature))
                step_feature = F.normalize(self.image_encoder_head(step_feature))
                step_features.append(step_feature)
                if "batch_step" in self.hparams.losses:
                    batch_step_features.append(step_feature[batch["step_index"][i]])
            if "batch_step" in self.hparams.losses:
                batch_step_features = torch.stack(batch_step_features, dim=0)
        elif "step_image" in batch:
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
        page_features = []
        batch_page_features = []
        if "page_image_list" in batch:
            for i, page_images in enumerate(batch["page_image_list"]):
                page_feature = F.normalize(self.image_encoder(page_images))
                if self.hparams.sprf:
                    page_feature = F.normalize(self.append_sprf(page_feature))
                page_feature = self.image_encoder_head(page_feature)
                page_features.append(F.normalize(page_feature))
                if "batch_page" in self.hparams.losses:
                    batch_page_features.append(page_feature[batch["page_index"][i]])
            if "batch_page" in self.hparams.losses:
                batch_page_features = torch.stack(batch_page_features, dim=0)
        elif "page_image" in batch:
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

        return video_features, step_features, page_features, batch_step_features, batch_page_features

    def training_step(self, batch, batch_idx):
        stage = "train"
        batch_size = batch["video"][0].shape[0]
        video_features, step_features, page_features, batch_step_features, batch_page_features = self.train_forward(
            batch
        )

        loss = torch.zeros(1, device=video_features.device)

        # batch step
        if "batch_step" in self.hparams.losses:
            batch_video_step_logits = video_features @ batch_step_features.t()
            batch_key_list = [batch["furniture_id"], batch["video_id"], batch["step_index"]]
            batch_video_step_ground_truth = self.get_batch_ground_truth(batch_key_list)
            loss += self.get_batch_loss(
                self.batch_step_loss_tau.exp() * batch_video_step_logits, batch_video_step_ground_truth, stage, "step"
            )

        # batch page
        if "batch_page" in self.hparams.losses:
            batch_video_page_logits = video_features @ batch_page_features.t()
            batch_key_list = [batch["furniture_id"], batch["video_id"], batch["page_index"]]
            batch_video_page_ground_truth = self.get_batch_ground_truth(batch_key_list)
            loss += self.get_batch_loss(
                self.batch_page_loss_tau.exp() * batch_video_page_logits, batch_video_page_ground_truth, stage, "page"
            )

        # step
        if "step" in self.hparams.losses:
            video_step_logits = self.get_one_to_many_logits(video_features, step_features)
            step_loss = self.get_one_to_many_loss(
                video_step_logits, self.step_loss_tau.exp(), batch["step_index"], stage, "step"
            )

            # step intra
            if "step_intra" in self.hparams.losses:
                step_intra_loss = self.get_image_list_intra_loss(
                    step_features, self.step_intra_loss_tau.exp(), self.step_intra_loss_sigma, stage, "step_intra"
                )
                loss += step_loss + step_intra_loss
            else:
                loss += step_loss

        # page
        if "page" in self.hparams.losses:
            video_page_logits = self.get_one_to_many_logits(video_features, page_features)
            page_loss = self.get_one_to_many_loss(
                video_page_logits, self.page_loss_tau.exp(), batch["page_index"], stage, "page"
            )

            # page intra
            if "page_intra" in self.hparams.losses:
                page_intra_loss = self.get_image_list_intra_loss(
                    page_features, self.page_intra_loss_tau.exp(), self.page_intra_loss_sigma, stage, "page_intra"
                )
                loss += page_loss + page_intra_loss
            else:
                loss += page_loss

        # log tau
        with torch.no_grad():
            if "batch_step" in self.hparams.losses:
                self.log(f"{stage}/tau/batch_step", 1 / self.batch_step_loss_tau.exp(), batch_size=batch_size)
            if "batch_page" in self.hparams.losses:
                self.log(f"{stage}/tau/batch_page", 1 / self.batch_page_loss_tau.exp(), batch_size=batch_size)
            if "step" in self.hparams.losses:
                self.log(f"{stage}/tau/step", 1 / self.step_loss_tau.exp(), batch_size=batch_size)
                if "step_intra" in self.hparams.losses:
                    self.log(f"{stage}/tau/step_intra", 1 / self.step_intra_loss_tau.exp(), batch_size=batch_size)
                    self.log(f"{stage}/sigma/step_intra", self.step_intra_loss_sigma, batch_size=batch_size)
            if "page" in self.hparams.losses:
                self.log(f"{stage}/tau/page", 1 / self.page_loss_tau.exp(), batch_size=batch_size)
                if "page_intra" in self.hparams.losses:
                    self.log(f"{stage}/tau/page_intra", 1 / self.page_intra_loss_tau.exp(), batch_size=batch_size)
                    self.log(f"{stage}/sigma/page_intra", 1 / self.page_intra_loss_sigma, batch_size=batch_size)
        # loss
        loss /= len([name for name in self.hparams.losses if "intra" not in name])
        self.log(f"{stage}/loss", loss, batch_size=batch_size)
        return loss

    def validation_forward(self, batch):
        video_features = F.normalize(self.video_encoder(batch["video"]))
        if self.hparams.sprf:
            pf = batch["video_position_feature"].unsqueeze(1).float()
            video_features = F.normalize(
                torch.cat([video_features, torch.sin(torch.pi * pf), torch.cos(torch.pi * pf)], dim=1)
            )
        video_features = F.normalize(self.video_encoder_head(video_features))

        step_features = []
        for step_images in batch["step_image_list"]:
            step_feature = F.normalize(self.image_encoder(step_images))
            if self.hparams.sprf:
                step_feature = F.normalize(self.append_sprf(step_feature))
            step_feature = F.normalize(self.image_encoder_head(step_feature))
            step_features.append(step_feature)

        page_features = []
        for page_images in batch["page_image_list"]:
            page_feature = F.normalize(self.image_encoder(page_images))
            if self.hparams.sprf:
                page_feature = F.normalize(self.append_sprf(page_feature))
            page_feature = F.normalize(self.image_encoder_head(page_feature))
            page_features.append(page_feature)

        return video_features, step_features, page_features

    def validation_step(self, batch, batch_idx):
        stage = "val"
        batch_size = batch["video"][0].shape[0]
        video_features, step_features, page_features = self.validation_forward(batch)

        # step
        video_step_logits = self.get_one_to_many_logits(video_features, step_features)
        step_tau = self.step_loss_tau.exp() if "step" in self.hparams.losses else 1
        step_loss = self.get_one_to_many_loss(video_step_logits, step_tau, batch["step_index"], stage, "step")

        # page
        video_page_logits = self.get_one_to_many_logits(video_features, page_features)
        page_tau = self.page_loss_tau.exp() if "page" in self.hparams.losses else 1
        page_loss = self.get_one_to_many_loss(video_page_logits, page_tau, batch["page_index"], stage, "page")

        # loss
        loss = (step_loss + page_loss) / 2
        self.log(f"{stage}/loss", loss, batch_size=batch_size)

        self.validation_outputs.append(
            dict(
                video_step_logits=video_step_logits,
                video_page_logits=video_page_logits,
                step_index=batch["step_index"],
                page_index=batch["page_index"],
                furniture_id=batch["furniture_id"],
                video_id=batch["video_id"],
                clip_index=batch["clip_index"],
            )
        )

    def on_validation_epoch_end(self):
        calculate_aggregated_accuracy(self, "val")
        self.validation_outputs = []

    def test_step(self, batch, batch_idx):
        video_features, step_features, page_features = self.validation_forward(batch)
        # step
        video_step_logits = self.get_one_to_many_logits(video_features, step_features)
        # page
        video_page_logits = self.get_one_to_many_logits(video_features, page_features)
        # batch accuracy
        self.test_outputs.append(
            dict(
                video_features=video_features,
                video_step_logits=video_step_logits,
                video_page_logits=video_page_logits,
                step_index=batch["step_index"],
                page_index=batch["page_index"],
                furniture_id=batch["furniture_id"],
                video_id=batch["video_id"],
                clip_index=batch["clip_index"],
            )
        )

    def on_test_epoch_end(self):
        calculate_aggregated_accuracy(self, "test")
        self.test_outputs = []
