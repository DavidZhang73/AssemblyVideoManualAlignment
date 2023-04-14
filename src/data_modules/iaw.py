import copy
import csv
import math
import os
import pickle
from abc import ABC
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import pytorch_lightning as pl
import torch
from pytorchvideo.data import make_clip_sampler
from pytorchvideo.data.clip_sampling import ClipSampler, ConstantClipsPerVideoSampler, RandomClipSampler
from pytorchvideo.data.utils import MultiProcessSampler
from torch.utils.data import DataLoader, default_collate

from src.data_modules.transforms import make_data_augmentation_transforms, make_transforms


class IKEAAssemblyInTheWildFrameVideo:
    def __init__(self, frame_video_pathname) -> None:
        self._frame_video_pathname = frame_video_pathname
        self._frames = np.load(frame_video_pathname, mmap_mode="r")
        self._fps = 30.0

    @property
    def name(self) -> str:
        return self._frame_video_pathname

    @property
    def duration(self) -> float:
        return float(self._frames.shape[0]) / self._fps

    def _get_frame_index_for_time(self, time_sec: float) -> int:
        return math.ceil(self._fps * time_sec)

    def get_clip(self, clip_start: float, clip_end: float) -> Dict[str, Optional[torch.Tensor]]:
        start_index = self._get_frame_index_for_time(clip_start)
        end_index = self._get_frame_index_for_time(clip_end)
        frame_indices = list(range(start_index, end_index))
        clip_frames = self._frames[start_index:end_index]
        # thwc -> cthw
        clip_frames = torch.tensor(clip_frames).permute(3, 0, 1, 2)
        return dict(video=clip_frames, frame_indices=frame_indices)


class IKEAAssemblyInTheWildFrameDataset(torch.utils.data.IterableDataset, ABC):
    def __init__(
        self,
        image_pkl_pathname: str,
        video_clip_info_dict: List[Tuple[str, Optional[dict]]],
        clip_sampler: ClipSampler,
        include: str = "both",
        include_image_list: bool = True,
        video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
        transform: Optional[Callable[[dict], Any]] = None,
    ) -> None:
        self._image_pkl_pathname = image_pkl_pathname
        self._video_info_dict = video_clip_info_dict
        self._include = include
        self._include_image_list = include_image_list
        self._clip_sampler = clip_sampler
        self._transform = transform

        # If a RandomSampler is used we need to pass in a custom random generator that
        # ensures all PyTorch multiprocess workers have the same random seed.
        self._video_random_generator = None
        if video_sampler == torch.utils.data.RandomSampler:
            self._video_random_generator = torch.Generator()
            self._video_sampler = video_sampler(self._video_info_dict, generator=self._video_random_generator)
        else:
            self._video_sampler = video_sampler(self._video_info_dict)

        self._image_pkl_file = None  # Initialized on first call to self.__next__()
        self._video_sampler_iter = None  # Initialized on first call to self.__next__()

        self._loaded_video = None
        self._last_clip_end_time = None

    def __next__(self) -> dict:
        if not self._image_pkl_file:
            # Setup pickle.load here - after PyTorch DataLoader workers are spawned.
            with open(os.path.join(self._image_pkl_pathname), "rb") as f:
                self._image_pkl_file = pickle.load(f)

        if not self._video_sampler_iter:
            # Setup MultiProcessSampler here - after PyTorch DataLoader workers are spawned.
            self._video_sampler_iter = iter(MultiProcessSampler(self._video_sampler))

        if self._loaded_video:
            video, info_dict, video_index = self._loaded_video
        else:
            video_index = next(self._video_sampler_iter)
            frame_video_pathname, info_dict = self._video_info_dict[video_index]
            info_dict = copy.deepcopy(info_dict)
            furniture_id = info_dict["furniture_id"]
            if self._include_image_list:
                if self._include in ["step", "both"]:
                    info_dict["step_image_list"] = self._image_pkl_file[furniture_id]["step_image_list"]
                if self._include in ["page", "both"]:
                    info_dict["page_image_list"] = self._image_pkl_file[furniture_id]["page_image_list"]
            else:
                if self._include in ["step", "both"]:
                    info_dict["step_image"] = self._image_pkl_file[furniture_id]["step_image_list"][
                        info_dict["step_index"]
                    ]
                if self._include in ["page", "both"]:
                    info_dict["page_image"] = self._image_pkl_file[furniture_id]["page_image_list"][
                        info_dict["page_index"]
                    ]
            video = IKEAAssemblyInTheWildFrameVideo(frame_video_pathname)
            self._loaded_video = (video, info_dict, video_index)

        (
            clip_start,
            clip_end,
            clip_index,
            aug_index,
            is_last_clip,
        ) = self._clip_sampler(0.0, video.duration, {})
        self._last_clip_end_time = clip_end

        if is_last_clip:
            self._loaded_video = None
            self._last_clip_end_time = None
            self._clip_sampler.reset()

        frames = video.get_clip(clip_start, clip_end)["video"]
        sample_dict = {
            "video": frames,
            "video_index": video_index,
            **info_dict,
        }
        return self._transform(sample_dict)

    def __iter__(self):
        self._video_sampler_iter = None  # Reset video sampler

        # If we're in a PyTorch DataLoader multiprocessing context, we need to use the
        # same seed for each worker's RandomSampler generator. The workers at each
        # __iter__ call are created from the unique value: worker_info.seed - worker_info.id,
        # which we can use for this seed.
        worker_info = torch.utils.data.get_worker_info()
        if self._video_random_generator is not None and worker_info is not None:
            base_seed = worker_info.seed - worker_info.id
            self._video_random_generator.manual_seed(base_seed)

        return self

    def __len__(self):
        clip_sampler = self._clip_sampler
        if isinstance(clip_sampler, RandomClipSampler):
            return len(self._video_info_dict)
        elif isinstance(clip_sampler, ConstantClipsPerVideoSampler):
            return len(self._video_info_dict) * clip_sampler._clips_per_video
        else:
            raise ValueError(f"Unsupported clip sampler type: {type(clip_sampler)}")


class IKEAAssemblyInTheWildFrameDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_path: str,
        train_file: str = os.path.join("split", "train_clip.csv"),
        val_file: str = os.path.join("split", "val_clip.csv"),
        test_file: str = os.path.join("split", "test_clip.csv"),
        image_file: str = "image.pkl",
        alpha: int = 4,
        clip_duration: float = 2.1333,
        num_frames: int = 32,
        include: str = "both",
        include_image_list: bool = True,
        # Data Loader
        batch_size: int = 128,
        val_batch_size: int = 128,
        test_batch_size: int = 128,
        num_workers: int = 8,
        pin_memory: bool = True,
        drop_last: bool = True,
        shuffle: bool = False,
        prefetch_factor: int = 2,
    ):
        super().__init__()
        self.save_hyperparameters()

        self._image_pkl_pathname = os.path.join(dataset_path, image_file)

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

        self._include = include
        self._include_image_list = include_image_list

    def _get_video_clip_info_dict(self, file_name: str):
        ret = []
        with open(os.path.join(self.hparams.dataset_path, file_name), "r", encoding="utf8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                (
                    furniture_id,
                    furniture_sub_category,
                    video_id,
                    page_index,
                    step_index,
                    clip_index,
                    step_list_length,
                    page_list_length,
                    start_time,
                    end_time,
                    full_video_duration,
                ) = row
                frame_video_pathname = os.path.join(
                    self.hparams.dataset_path,
                    "Furniture",
                    furniture_sub_category,
                    furniture_id,
                    "video",
                    video_id,
                    f"step_{step_index}_clip_{clip_index}.npy",
                )
                page_index = int(page_index)
                step_index = int(step_index)
                clip_index = int(clip_index)

                page_list_length = int(page_list_length)
                step_list_length = int(step_list_length)

                start_time = float(start_time)
                end_time = float(end_time)
                full_video_duration = float(full_video_duration)

                ret.append(
                    (
                        frame_video_pathname,
                        dict(
                            furniture_id=furniture_id,
                            video_id=video_id,
                            page_index=page_index,
                            step_index=step_index,
                            clip_index=clip_index,
                            page_position_feature=page_index / page_list_length,
                            step_position_feature=step_index / step_list_length,
                            video_position_feature=(start_time + (end_time - start_time) / 2) / full_video_duration,
                        ),
                    )
                )
        return ret

    def _get_train_dataset(self):
        return IKEAAssemblyInTheWildFrameDataset(
            image_pkl_pathname=self._image_pkl_pathname,
            video_clip_info_dict=self._get_video_clip_info_dict(self.hparams.train_file),
            include=self._include,
            include_image_list=self._include_image_list,
            clip_sampler=make_clip_sampler("random", self.hparams.clip_duration),
            transform=make_data_augmentation_transforms(
                num_frames=self.hparams.num_frames,
                alpha=self.hparams.alpha,
                crop_size=224,
                include=self._include,
                include_image_list=self._include_image_list,
            ),
        )

    def _get_val_dataset(self):
        return IKEAAssemblyInTheWildFrameDataset(
            image_pkl_pathname=self._image_pkl_pathname,
            video_clip_info_dict=self._get_video_clip_info_dict(self.hparams.val_file),
            include="both",
            include_image_list=True,
            video_sampler=torch.utils.data.SequentialSampler,
            clip_sampler=make_clip_sampler("constant_clips_per_video", self.hparams.clip_duration, 1),
            transform=make_transforms(num_frames=self.hparams.num_frames, alpha=self.hparams.alpha, crop_size=224),
        )

    def _get_test_dataset(self):
        return IKEAAssemblyInTheWildFrameDataset(
            image_pkl_pathname=self._image_pkl_pathname,
            video_clip_info_dict=self._get_video_clip_info_dict(self.hparams.test_file),
            include="both",
            include_image_list=True,
            video_sampler=torch.utils.data.SequentialSampler,
            clip_sampler=make_clip_sampler("constant_clips_per_video", self.hparams.clip_duration, 5),
            transform=make_transforms(num_frames=self.hparams.num_frames, alpha=self.hparams.alpha, crop_size=224),
        )

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            self._train_dataset = self._get_train_dataset()
        if stage in ["fit", "validate"] or stage is None:
            self._val_dataset = self._get_val_dataset()
        if stage == "test" or stage is None:
            self._test_dataset = self._get_test_dataset()

    def collate_fn(self, batch):
        ret = {}
        for key in batch[0].keys():
            if key in ["step_image_list", "page_image_list"]:
                ret[key] = [default_collate(item[key]) for item in batch]
            else:
                ret[key] = default_collate([sample[key] for sample in batch])

        return ret

    def train_dataloader(self):
        return DataLoader(
            self._train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=self.hparams.drop_last,
            shuffle=self.hparams.shuffle,
            collate_fn=self.collate_fn,
            prefetch_factor=self.hparams.prefetch_factor,
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_dataset,
            batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=False,
            shuffle=False,
            collate_fn=self.collate_fn,
            prefetch_factor=self.hparams.prefetch_factor,
        )

    def test_dataloader(self):
        return DataLoader(
            self._test_dataset,
            batch_size=self.hparams.test_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=False,
            shuffle=False,
            collate_fn=self.collate_fn,
            prefetch_factor=self.hparams.prefetch_factor,
        )
