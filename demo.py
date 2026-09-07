#!/usr/bin/env python3
"""
Yeast Panoptic Segmentation Demo (command-line)

Loads the trained Panoptic-DeepLab model, runs it on an input image, and
saves the predicted panoptic segmentation mask and instance predictions
as PNG files.

Usage:
    python demo.py --image ./20260821_germination/kambucha03.tif

    python demo.py \
        --image ./20260821_germination/kambucha03.tif \
        --config /home/liwa/data/PytrochDeepyeast/projects/Panoptic-DeepLab/configs/yeast_panoptics/config.yaml \
        --weights /home/liwa/data/oneformer_output/output/model_0159999_v2.pth \
        --output-dir ./output \
        --device cuda \
        --score-thresh 0.3

Run from the project root (same directory this notebook's cells expected),
so that `./detectron2` and `./projects/Panoptic-DeepLab/` resolve correctly,
and DETECTRON2_DATASETS points at your dataset root if the config needs it.
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
from skimage.io import imread, imsave


def parse_args():
    parser = argparse.ArgumentParser(description="Run panoptic segmentation on a single image.")
    parser.add_argument("--image", required=True, help="Path to the input image (e.g. .tif, .png).")
    parser.add_argument(
        "--config",
        default="/home/liwa/data/PytrochDeepyeast/projects/Panoptic-DeepLab/configs/yeast_panoptics/config.yaml",
        help="Path to the Panoptic-DeepLab config yaml.",
    )
    parser.add_argument(
        "--weights",
        default="/home/liwa/data/oneformer_output/output/model_0159999_v2.pth",
        help="Path to the model weights (.pth).",
    )
    parser.add_argument(
        "--dataset-root",
        default="/home/liwa/data/datasets/",
        help="Value for the DETECTRON2_DATASETS environment variable.",
    )
    parser.add_argument("--output-dir", default="./output", help="Directory to save output images.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device to run inference on.")
    parser.add_argument("--score-thresh", type=float, default=0.3, help="Score threshold for instance predictions.")
    parser.add_argument("--scale", type=float, default=0.5, help="Visualization scale for the panoptic mask overlay.")
    parser.add_argument(
        "--infer-size",
        type=int,
        default=1024,
        help=(
            "Resize the image so its longer side is this many pixels before running the model "
            "(e.g. 1024 or 512), then scale the predicted mask/boxes back up to the original size. "
            "Lowers GPU memory use on large images. Set to 0 to disable and predict at full resolution."
        ),
    )
    return parser.parse_args()


def resize_long_side(image, target):
    """Resize `image` so its longer side equals `target`, keeping aspect ratio."""
    h, w = image.shape[:2]
    scale = target / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
    return resized


def resize_panoptic_seg(panoptic_seg, orig_size):
    """Nearest-neighbor resize a panoptic label map back to `orig_size` (H, W)."""
    arr = panoptic_seg.to("cpu").numpy()
    orig_h, orig_w = orig_size
    resized = cv2.resize(arr.astype(np.int32), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return torch.as_tensor(resized.astype(np.int64))


def resize_instances(instances, orig_size):
    """Scale an Instances object's boxes/masks from its current image_size up to `orig_size`."""
    from detectron2.structures import Instances

    instances = instances.to("cpu")
    orig_h, orig_w = orig_size
    cur_h, cur_w = instances.image_size
    scale_x, scale_y = orig_w / cur_w, orig_h / cur_h

    new_instances = Instances((orig_h, orig_w))
    if instances.has("scores"):
        new_instances.scores = instances.scores
    if instances.has("pred_classes"):
        new_instances.pred_classes = instances.pred_classes
    if instances.has("pred_boxes"):
        boxes = instances.pred_boxes.clone()
        boxes.scale(scale_x, scale_y)
        new_instances.pred_boxes = boxes
    if instances.has("pred_masks"):
        masks = instances.pred_masks.numpy()
        if len(masks):
            resized_masks = np.stack(
                [
                    cv2.resize(m.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST).astype(bool)
                    for m in masks
                ]
            )
        else:
            resized_masks = np.zeros((0, orig_h, orig_w), dtype=bool)
        new_instances.pred_masks = torch.as_tensor(resized_masks)
    return new_instances


def build_predictor(config_path, weights_path, dataset_root, device, score_thresh):
    sys.path.insert(0, os.path.abspath("./detectron2"))
    sys.path.insert(0, os.path.abspath("."))
    sys.path.insert(0, os.path.abspath("./projects/Panoptic-DeepLab/"))
    os.environ["DETECTRON2_DATASETS"] = dataset_root

    from detectron2.config import get_cfg
    from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config
    from prediction import Predictor

    cfg = get_cfg()
    add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(config_path)
    cfg.MODEL.DEVICE = device
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.WEIGHTS = os.path.abspath(weights_path)

    return cfg, Predictor(cfg)


def build_segments_info(panoptic_seg, instances):
    scores = [1] + list(instances.scores.to("cpu"))
    segments_info = []
    for i, panoptic_label in enumerate(np.unique(panoptic_seg.to("cpu").numpy())[1:]):
        if panoptic_label == -1:
            # VOID region
            continue
        segments_info.append(
            {
                "id": int(panoptic_label),
                "category_id": int(panoptic_label // 1000),
                "isthing": i != 0,
                "score": float(scores[i]),
            }
        )
    return segments_info


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Building predictor (device={args.device}) ...")
    cfg, predictor = build_predictor(
        args.config, args.weights, args.dataset_root, args.device, args.score_thresh
    )

    from run_notebooks.post_process_utils import convert_to_rgb
    from detectron2.data import MetadataCatalog
    from detectron2.utils.visualizer import Visualizer

    print(f"Loading image: {args.image}")
    im = imread(args.image)
    image = convert_to_rgb(im)
    orig_size = image.shape[:2]

    if args.infer_size > 0 and max(orig_size) > args.infer_size:
        infer_image = resize_long_side(image, args.infer_size)
        print(f"Resized {orig_size} -> {infer_image.shape[:2]} for inference")
    else:
        infer_image = image

    print("Running prediction ...")
    prediction_output = predictor(infer_image)
    instances = prediction_output["instances"]
    panoptic_seg, _ = prediction_output["panoptic_seg"]

    if infer_image.shape[:2] != orig_size:
        print(f"Scaling predictions {infer_image.shape[:2]} -> {orig_size}")
        panoptic_seg = resize_panoptic_seg(panoptic_seg, orig_size)
        instances = resize_instances(instances, orig_size)

    segments_info = build_segments_info(panoptic_seg, instances)

    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    print("Rendering panoptic segmentation mask ...")
    v = Visualizer(image[:, :, ::-1], metadata, scale=args.scale)
    out = v.draw_panoptic_seg(panoptic_seg.to("cpu"), segments_info, area_threshold=20.0, alpha=0.8)
    panoptic_path = os.path.join(args.output_dir, "panoptic_seg.png")
    imsave(panoptic_path, out.get_image()[:, :, ::-1])
    print(f"Saved: {panoptic_path}")

    print("Rendering instance predictions ...")
    v = Visualizer(image[:, :, ::-1], metadata, scale=1)
    out = v.draw_instance_predictions(instances.to("cpu"), segments_info)
    instances_path = os.path.join(args.output_dir, "instance_predictions.png")
    imsave(instances_path, out.get_image()[:, :, ::-1])
    print(f"Saved: {instances_path}")

    print(f"Found {len(segments_info)} segments ({len(instances)} instances).")


if __name__ == "__main__":
    main()
