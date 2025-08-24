# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging

import numpy as np
from torch import nn
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np
import torch
from dinov3.data.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, GaussianBlur, make_normalize_transform

logger = logging.getLogger("dinov3")


class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        gram_teacher_crops_size=None,
        gram_teacher_no_distortions=False,
        teacher_no_color_jitter=False,
        local_crops_subset_of_global_crops=False,
        patch_size=16,
        share_color_jitter=False,
        horizontal_flips=True,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        mask_as_third_channel: bool = True,  # NEW
        disable_color_ops_if_mask: bool = True
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.gram_teacher_crops_size = gram_teacher_crops_size
        self.gram_teacher_no_distortions = gram_teacher_no_distortions
        self.teacher_no_color_jitter = teacher_no_color_jitter
        self.local_crops_subset_of_global_crops = local_crops_subset_of_global_crops
        self.patch_size = patch_size
        self.share_color_jitter = share_color_jitter
        self.mean = mean
        self.std = std
        self.mask_as_third_channel = mask_as_third_channel
        self.disable_color_ops_if_mask = disable_color_ops_if_mask

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info(f"gram_crops_size: {gram_teacher_crops_size}")
        logger.info(f"gram_teacher_no_distortions: {gram_teacher_no_distortions}")
        logger.info(f"teacher_no_color_jitter: {teacher_no_color_jitter}")
        logger.info(f"local_crops_subset_of_global_crops: {local_crops_subset_of_global_crops}")
        logger.info(f"patch_size if local_crops_subset_of_global_crops: {patch_size}")
        logger.info(f"share_color_jitter: {share_color_jitter}")
        logger.info(f"horizontal flips: {horizontal_flips}")
        logger.info("###################################")

        # Global crops and gram teacher crops can have different sizes. We first take a crop of the maximum size
        # and then resize it to the desired size for global and gram teacher crops.
        global_crop_max_size = max(global_crops_size, gram_teacher_crops_size if gram_teacher_crops_size else 0)

        # random resized crop and flip
        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crop_max_size,
                    scale=global_crops_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5 if horizontal_flips else 0.0),
            ]
        )

        resize_global = nn.Identity()  # Resize transform applied to global crops after random crop
        self.resize_global_post_transf = (
            nn.Identity()
        )  # Resize transform applied to global crops after all other transforms
        self.resize_gram_teacher = None  # Resize transform applied to crops for gram teacher
        if gram_teacher_crops_size is not None:
            # All resize transforms will do nothing if the crop size is already the desired size.
            if gram_teacher_no_distortions:
                # When there a no distortions for the gram teacher crop, we can resize before the distortions.
                # This is the preferred order, because it keeps the image size for the augmentations consistent,
                # which matters e.g. for GaussianBlur.
                resize_global = transforms.Resize(
                    global_crops_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                )
            else:
                # When there a no distortions for the gram teacher crop, we need to resize after the distortions,
                # because the distortions are shared between global and gram teacher crops.
                self.resize_global_post_transf = transforms.Resize(
                    global_crops_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                )

            self.resize_gram_teacher = transforms.Resize(
                gram_teacher_crops_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
            )

        self.geometric_augmentation_local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crops_size,
                    scale=local_crops_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5 if horizontal_flips else 0.0),
            ]
        )

        # color distortions / blurring
        color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0)

        global_transfo2_extra = transforms.Compose(
            [
                GaussianBlur(p=0.1),
                transforms.RandomSolarize(threshold=128, p=0.2),
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform(mean=mean, std=std),
            ]
        )

        if self.share_color_jitter:
            self.color_jittering = color_jittering
            self.global_transfo1 = transforms.Compose([resize_global, global_transfo1_extra, self.normalize])
            self.global_transfo2 = transforms.Compose([resize_global, global_transfo2_extra, self.normalize])
            self.local_transfo = transforms.Compose([local_transfo_extra, self.normalize])
        else:
            self.global_transfo1 = transforms.Compose(
                [resize_global, color_jittering, global_transfo1_extra, self.normalize]
            )
            self.global_transfo2 = transforms.Compose(
                [resize_global, color_jittering, global_transfo2_extra, self.normalize]
            )
            self.local_transfo = transforms.Compose([color_jittering, local_transfo_extra, self.normalize])

    def _normalize_tensor(self, x: torch.Tensor, mean, std):
        # x: C×H×W in [0,1]
        mean = torch.as_tensor(mean, dtype=x.dtype, device=x.device)[:, None, None]
        std  = torch.as_tensor(std,  dtype=x.dtype, device=x.device)[:, None, None]
        return (x - mean) / std

    def _augment_view_mask3c(self, x3: torch.Tensor, out_size: int, scale):
        # x3: 3×H×W tensor [I, I, M]; I in [0,1], M in {0,1}
        I2, M = x3[:2], x3[2:3]

        # sample crop once (like RandomResizedCrop)
        H, W = x3.shape[1:]
        # emulate get_params using an RGB dummy (only dims matter)
        dummy = Image.new("RGB", (W, H))
        i, j, h, w = transforms.RandomResizedCrop.get_params(dummy, scale=scale, ratio=(1.0, 1.0))

        I2 = F.resized_crop(I2, i, j, h, w, (out_size, out_size), interpolation=InterpolationMode.BICUBIC)
        M  = F.resized_crop(M,  i, j, h, w, (out_size, out_size), interpolation=InterpolationMode.NEAREST)

        # shared horizontal flip
        if np.random.rand() < 0.5:
            I2 = F.hflip(I2)
            M  = F.hflip(M)

        # optional color ops only on intensities
        if not self.disable_color_ops_if_mask:
            # ColorJitter & Grayscale on I2 (tensor-safe in torchvision)
            jitter = transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            I2 = jitter(I2)
            if np.random.rand() < 0.2:
                I2 = transforms.RandomGrayscale(p=1.0)(I2)
            # Blur is fine on I2, skip Solarize (would corrupt mask semantics anyway)

        # normalize (mask channel unchanged semantically)
        mean = self.mean if len(self.mean) == 3 else [0.5, 0.5, 0.0]
        std  = self.std  if len(self.std)  == 3 else [0.5, 0.5, 1.0]
        x3n = torch.cat([I2, M], dim=0)
        x3n = self._normalize_tensor(x3n, mean=mean, std=std)
        return x3n

    def __call__(self, image):
        output = {}
        output["weak_flag"] = True

        # ---- NEW: mask-aware tensor branch ----
        if self.mask_as_third_channel and isinstance(image, torch.Tensor):
            # Expect 3×H×W [I,I,M]
            # Global crops (2)
            g1 = self._augment_view_mask3c(image, self.global_crops_size, self.global_crops_scale)
            g2 = self._augment_view_mask3c(image, self.global_crops_size, self.global_crops_scale)
            output["global_crops"] = [g1, g2]

            # Teacher crops: either same as student without extra jitter, or same as above
            if self.teacher_no_color_jitter:
                # re-augment without extra color ops (we already kept mask clean)
                t1 = self._augment_view_mask3c(image, self.global_crops_size, self.global_crops_scale)
                t2 = self._augment_view_mask3c(image, self.global_crops_size, self.global_crops_scale)
                output["global_crops_teacher"] = [t1, t2]
            else:
                output["global_crops_teacher"] = [g1, g2]

            # Local crops
            locals_ = [self._augment_view_mask3c(image, self.local_crops_size, self.local_crops_scale)
                    for _ in range(self.local_crops_number)]
            output["local_crops"] = locals_
            output["offsets"] = ()
            # gram teacher crops (optional): skip or mirror g1/g2 sizes if you need them
            return output

        # ---- original PIL path (RGB) ----
        if self.share_color_jitter:
            image = self.color_jittering(image)

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1_transf = self.global_transfo1(im1_base)
        global_crop_1 = self.resize_global_post_transf(global_crop_1_transf)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2_transf = self.global_transfo2(im2_base)
        global_crop_2 = self.resize_global_post_transf(global_crop_2_transf)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        if self.teacher_no_color_jitter:
            output["global_crops_teacher"] = [
                self.normalize(im1_base),
                self.normalize(im2_base),
            ]
        else:
            output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # gram teacher crops:
        if self.gram_teacher_crops_size is not None:
            if self.gram_teacher_no_distortions:
                gram_crop_1 = self.normalize(self.resize_gram_teacher(im1_base))
                gram_crop_2 = self.normalize(self.resize_gram_teacher(im2_base))
            else:
                gram_crop_1 = self.resize_gram_teacher(global_crop_1_transf)
                gram_crop_2 = self.resize_gram_teacher(global_crop_2_transf)
            output["gram_teacher_crops"] = [gram_crop_1, gram_crop_2]

        # local crops:
        if self.local_crops_subset_of_global_crops:
            _local_crops = [self.local_transfo(im1_base) for _ in range(self.local_crops_number // 2)] + [
                self.local_transfo(im2_base) for _ in range(self.local_crops_number // 2)
            ]
            local_crops, offsets = [], []
            gs, ls = self.global_crops_size, self.local_crops_size
            for img in _local_crops:
                rx, ry = np.random.randint(0, (gs - ls) // self.patch_size, 2) * self.patch_size
                local_crops.append(img[:, rx : rx + ls, ry : ry + ls])
                offsets.append((rx, ry))
            output["local_crops"] = local_crops
            output["offsets"] = offsets
        else:
            output["local_crops"] = [self.local_transfo(self.geometric_augmentation_local(image))
                                    for _ in range(self.local_crops_number)]
            output["offsets"] = ()

        return output

