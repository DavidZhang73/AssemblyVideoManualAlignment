from typing import Callable

import pytorchvideo.transforms as VT
import torch
import torchvision.transforms as T
from torch import Tensor
from torchvision.transforms import Compose


class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """

    def __init__(self, alpha: int = 4):
        super().__init__()
        self.alpha = alpha

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(0, frames.shape[1] - 1, frames.shape[1] // self.alpha).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


def _make_avm_transforms(num_frames: int = 32, alpha: int = 4, crop_size: int = 224):
    return Compose(
        [
            VT.ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        VT.UniformTemporalSubsample(num_frames),
                        VT.Div255(),
                        VT.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
                        T.CenterCrop(crop_size),
                        PackPathway(alpha=alpha),
                    ]
                ),
            ),
            VT.ApplyTransformToKey(
                key="step_image",
                transform=Compose(
                    [
                        T.ToTensor(),
                        T.Normalize(mean=0.45, std=0.225),
                    ]
                ),
            ),
        ]
    )


class ApplyTransformToList:
    """
    Apply a transform to each tensor in the list.
    """

    def __init__(self, transform: Callable):
        self._transform = transform

    def __call__(self, tensor_list: list[Tensor]) -> list[Tensor]:
        return [self._transform(tensor) for tensor in tensor_list]


class ApplyTransformToKeySkipNone:
    """
    Apply a transform to a specific key in a dictionary. If the key is not present,
    the dictionary is returned unchanged.
    """

    def __init__(self, key: str, transform: Callable):
        self._key = key
        self._transform = transform

    def __call__(self, data: dict) -> dict:
        if self._key in data and data[self._key] is not None:
            data[self._key] = self._transform(data[self._key])
        return data


def make_data_augmentation_transforms(
    num_frames: int = 32, alpha: int = 4, crop_size: int = 224, include: str = "both", include_image_list: bool = True
):
    video_transform = Compose(
        [
            VT.UniformTemporalSubsample(num_frames),
            VT.Div255(),
            VT.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
            T.RandomResizedCrop(crop_size, antialias=True),
            PackPathway(alpha=alpha),
        ]
    )
    image_transform = Compose(
        [
            T.ToTensor(),
            T.RandomInvert(p=1.0),
            T.Normalize(mean=0.45, std=0.225),
            T.RandomHorizontalFlip(),
            T.RandomApply(transforms=[T.RandomResizedCrop(crop_size, antialias=True), T.RandomRotation(15)]),
        ]
    )
    transforms = [VT.ApplyTransformToKey(key="video", transform=video_transform)]
    if include_image_list:
        image_list_transform = ApplyTransformToList(transform=image_transform)
        if include in ["step", "both"]:
            transforms.append(ApplyTransformToKeySkipNone(key="step_image_list", transform=image_list_transform))
        if include in ["page", "both"]:
            transforms.append(ApplyTransformToKeySkipNone(key="page_image_list", transform=image_list_transform))
    else:
        if include in ["step", "both"]:
            transforms.append(ApplyTransformToKeySkipNone(key="step_image", transform=image_transform))
        if include in ["page", "both"]:
            transforms.append(ApplyTransformToKeySkipNone(key="page_image", transform=image_transform))
    return Compose(transforms)


def make_transforms(num_frames: int = 32, alpha: int = 4, crop_size: int = 224):
    return Compose(
        [
            VT.ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        VT.UniformTemporalSubsample(num_frames),
                        VT.Div255(),
                        VT.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
                        T.CenterCrop(crop_size),
                        PackPathway(alpha=alpha),
                    ]
                ),
            ),
            ApplyTransformToKeySkipNone(
                key="step_image_list",
                transform=ApplyTransformToList(
                    transform=Compose([T.ToTensor(), T.RandomInvert(p=1.0), T.Normalize(mean=0.45, std=0.225)])
                ),
            ),
            ApplyTransformToKeySkipNone(
                key="page_image_list",
                transform=ApplyTransformToList(
                    transform=Compose([T.ToTensor(), T.RandomInvert(p=1.0), T.Normalize(mean=0.45, std=0.225)]),
                ),
            ),
        ]
    )
