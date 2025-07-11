# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import numpy as np
from typing import Callable, List, Union
import torch
from panopticapi.utils import rgb2id

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from .target_generator import PanopticDeepLabTargetGenerator
# import sys
# import os
# sys.path.insert(0, ("/").join(os.path.abspath('.').split("/")[:-3]))
# from oneformer.data.datasets.register_yeast_panoptic_annos_semseg import register_all_yeast_panoptic_annos_sem_seg
from configs.yeast_panoptics.cityscapes_panoptic import register_all_yeastcity_panoptic


__all__ = ["PanopticDeeplabDatasetMapper"]


class PanopticDeeplabDatasetMapper:
    """
    The callable currently does the following:

    1. Read the image from "file_name" and label from "pan_seg_file_name"
    2. Applies random scale, crop and flip transforms to image and label
    3. Prepare data to Tensor and generate training targets from label
    """

    @configurable
    def __init__(
        self,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        panoptic_target_generator: Callable,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            panoptic_target_generator: a callable that takes "panoptic_seg" and
                "segments_info" to generate training targets for the model.
        """
        # fmt: off
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        # fmt: on
        logger = logging.getLogger(__name__)
        logger.info("Augmentations used in training: " + str(augmentations))

        self.panoptic_target_generator = panoptic_target_generator

    @classmethod
    def from_config(cls, cfg):
        # augs = [
        #     T.ResizeShortestEdge(
        #         cfg.INPUT.MIN_SIZE_TRAIN,
        #         cfg.INPUT.MAX_SIZE_TRAIN,
        #         cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
        #     )
        # ]
        # if cfg.INPUT.CROP.ENABLED:
        #     augs.append(T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
        # augs.append(T.RandomFlip())
        # augs.append(T.RandomBrightness(intensity_min=0.5, intensity_max=1.5))
        # augs.append(T.RandomContrast(intensity_min=0.5, intensity_max=1.5))
        augs = [
            T.RescaleShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        augs.append(T.RandomApply(T.RandomFlip(), 0.5))
        augs.append(T.RandomApply(T.RandomBrightness(intensity_min=0.5, intensity_max=1.5), 0.5))
        augs.append(T.RandomApply(T.RandomContrast(intensity_min=0.5, intensity_max=1.5), 0.5))
        if cfg.INPUT.CROP.ENABLED:
            augs.append(T.FixedSizeCrop(cfg.INPUT.CROP.SIZE))

        # Assume always applies to the training set.
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        panoptic_target_generator = PanopticDeepLabTargetGenerator(
            ignore_label=meta.ignore_label,
            thing_ids=list(meta.thing_dataset_id_to_contiguous_id.values()),
            sigma=cfg.INPUT.GAUSSIAN_SIGMA,
            ignore_stuff_in_offset=cfg.INPUT.IGNORE_STUFF_IN_OFFSET,
            small_instance_area=cfg.INPUT.SMALL_INSTANCE_AREA,
            small_instance_weight=cfg.INPUT.SMALL_INSTANCE_WEIGHT,
            ignore_crowd_in_semantic=cfg.INPUT.IGNORE_CROWD_IN_SEMANTIC,
        )

        ret = {
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "panoptic_target_generator": panoptic_target_generator,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # Load image.
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)
        # Panoptic label is encoded in RGB image.
        pan_seg_gt = utils.read_image(dataset_dict.pop("pan_seg_file_name"), "RGB")

        # Reuses semantic transform for panoptic labels.
        aug_input = T.AugInput(image, sem_seg=pan_seg_gt)
        transform = self.augmentations(aug_input)
        image, pan_seg_gt = aug_input.image, aug_input.sem_seg

        annos = [
            utils.transform_instance_annotations(obj, transform, image.shape[:2])
            for obj in dataset_dict.pop("segments_info")
        ]
        dataset_dict["segments_info"] = annos
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        # Generates training targets for Panoptic-DeepLab.
        targets = self.panoptic_target_generator(rgb2id(pan_seg_gt), dataset_dict["segments_info"])
        dataset_dict.update(targets)

        return dataset_dict
