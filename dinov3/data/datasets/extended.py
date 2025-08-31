# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from typing import Any, Tuple

from torchvision.datasets import VisionDataset

from .decoders import Decoder, ImageDataDecoder, TargetDecoder

import torch
from torchvision.transforms.functional import to_pil_image
class ExtendedVisionDataset(VisionDataset):
    def __init__(
        self,
        image_decoder: Decoder = ImageDataDecoder,
        target_decoder: Decoder = TargetDecoder,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore
        self.image_decoder = image_decoder
        self.target_decoder = target_decoder

    def get_image_data(self, index: int) -> bytes:
        raise NotImplementedError

    def get_target(self, index: int) -> Any:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            image_data = self.get_image_data(index)
            image = self.image_decoder(image_data).decode()
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e
        
                # ---- Fix: make sure transforms get a PIL image if they expect one ----
        if isinstance(image, torch.Tensor):
            # to_pil_image handles CHW/HWC and uint8 or float in [0, 1]
            image = to_pil_image(image)
        target = self.get_target(index)
        target = self.target_decoder(target).decode()

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        raise NotImplementedError
