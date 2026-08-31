from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from skimage.io import imread

# ============================================================
# Configuration
# ============================================================

IMAGE_FOLDER = "/path/to/validation/images"
MASK_FOLDER = "/path/to/validation/panoptic_masks"

TARGET_LAYER = "ins_embed_head.contrast_head"

INSTANCE_DIVISOR = 1000
BACKGROUND_IDS = {0}

MAX_IMAGES = None           # Example: 100; None means all images
MIN_ORIGINAL_PIXELS = 20    # Minimum cell size in original mask
MIN_FEATURE_PIXELS = 2      # Minimum after resizing to feature resolution

OUTPUT_DATA = "contrast_embeddings.npz"
OUTPUT_FIGURE = "contrast_embedding_tsne.png"

# Images larger than this are divided into four quadrants. Each quadrant is
# passed through the predictor independently.
TILE_THRESHOLD = 1000

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"
}

MASK_EXTENSIONS = [
    ".png", ".tif", ".tiff"
]

# Change according to your actual class IDs
PHENOTYPE_NAMES = {
    1: "Vegetative",
    2: "Shmoo",
    3: "Fusion",
    4: "Ascus",
    5: "Lysis",
    6: "Spores",
}

PHENOTYPE_COLORS = {
    1: "#E45756",
    2: "#54A24B",
    3: "#4C78A8",
    4: "#F58518",
    5: "#B279A2",
    6: "#72B7B2",
}


# ============================================================
# Hook the contrast head
# ============================================================

def find_mask(image_path, mask_folder):
    """
    Match:
        xxx_leftImg8bit.png
    with:
        xxx_gtFine_instanceIds.png
    """
    mask_folder = Path(mask_folder)

    if "_leftImg8bit" not in image_path.stem:
        print(
            f"Unexpected image filename, skipping: {image_path.name}"
        )
        return None

    mask_stem = image_path.stem.replace(
        "_leftImg8bit",
        "_gtFine_instanceIds",
    )

    for extension in MASK_EXTENSIONS:
        candidate = mask_folder / f"{mask_stem}{extension}"

        if candidate.exists():
            return candidate

    return None

class FeatureHook:
    def __init__(self, model, layer_name):
        modules = dict(model.named_modules())

        if layer_name not in modules:
            candidates = [
                name for name in modules
                if "contrast" in name.lower()
            ]

            raise ValueError(
                f"Layer not found: {layer_name}\n"
                f"Available contrast layers:\n"
                + "\n".join(candidates)
            )

        self.layer_name = layer_name
        self.feature = None

        self.handle = modules[layer_name].register_forward_hook(
            self._hook
        )

        print(f"Hooked layer: {layer_name}")

    def _hook(self, module, inputs, output):
        self.feature = self._extract_tensor(output)

    def _extract_tensor(self, output):
        if torch.is_tensor(output):
            return output.detach()

        if isinstance(output, dict):
            preferred_keys = [
                "contrast",
                "contrast_embedding",
                "embedding",
                "embeddings",
                "features",
            ]

            for key in preferred_keys:
                if key in output and torch.is_tensor(output[key]):
                    return output[key].detach()

            for value in output.values():
                try:
                    return self._extract_tensor(value)
                except (TypeError, ValueError):
                    pass

        if isinstance(output, (list, tuple)):
            for value in output:
                try:
                    return self._extract_tensor(value)
                except (TypeError, ValueError):
                    pass

        raise TypeError(
            f"Cannot find tensor in output of {self.layer_name}"
        )

    def clear(self):
        self.feature = None

    def close(self):
        self.handle.remove()


def collect_image_mask_pairs(image_folder, mask_folder):
    image_folder = Path(image_folder)

    image_paths = sorted([
        path
        for path in image_folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and path.stem.endswith("_leftImg8bit")
    ])

    pairs = []

    for image_path in image_paths:
        mask_path = find_mask(
            image_path=image_path,
            mask_folder=mask_folder,
        )

        if mask_path is None:
            print(
                f"Corresponding mask not found: "
                f"{image_path.name}"
            )
            continue

        pairs.append((image_path, mask_path))

    print(f"Found {len(pairs)} image-mask pairs.")

    return pairs

# ============================================================
# Image and mask loading
# ============================================================

# def read_image(image_path):
#     """
#     Detectron2 DefaultPredictor expects a BGR image.

#     Supports ordinary 8-bit and 16-bit microscopy images.
#     """
#     # image = cv2.imread(
#     #     str(image_path),
#     #     cv2.IMREAD_UNCHANGED,
#     # )

#     # if image is None:
#     #     raise ValueError(f"Could not read image: {image_path}")

#     # # Grayscale -> BGR
#     # if image.ndim == 2:
#     #     image = convert_to_uint8(image)
#     #     image = cv2.cvtColor(
#     #         image,
#     #         cv2.COLOR_GRAY2BGR,
#     #     )

#     # # Single-channel image
#     # elif image.ndim == 3 and image.shape[2] == 1:
#     #     image = convert_to_uint8(image[:, :, 0])
#     #     image = cv2.cvtColor(
#     #         image,
#     #         cv2.COLOR_GRAY2BGR,
#     #     )

#     # # BGRA -> BGR
#     # elif image.ndim == 3 and image.shape[2] == 4:
#     #     image = convert_to_uint8(image)
#     #     image = cv2.cvtColor(
#     #         image,
#     #         cv2.COLOR_BGRA2BGR,
#     #     )

#     # else:
#     #     image = convert_to_uint8(image)
#     image = imread(image_path)
#     # if image.ndim == 3:
#     #     image = image[:,:,0]
#     return image
#     # return convert_to_rgb(image)


# def convert_to_rgb(image):
#     """
#     Convert a 2D grayscale image to 3-channel RGB format for visualization.

#     Parameters:
#     -----------
#     image : np.ndarray
#         A 2D grayscale image (single channel).

#     Returns:
#     --------
#     image : np.ndarray
#         A 3D RGB image (H x W x 3) scaled to 8-bit unsigned integers.
#         If the input is already 3D (e.g., RGB), it is returned unchanged.
#     """
#     if image.ndim == 2:
#         image = image - image.min()
#         image = image / image.max()
#         image = (image * 255).astype(np.uint8)
#         image = np.stack((image,)*3, axis=-1)
#     return image


# def convert_to_uint8(image):
#     if image.dtype == np.uint8:
#         return image

#     image = image.astype(np.float32)

#     finite = np.isfinite(image)

#     if not finite.any():
#         return np.zeros(image.shape, dtype=np.uint8)

#     low, high = np.percentile(
#         image[finite],
#         [0.5, 99.5],
#     )

#     if high <= low:
#         return np.zeros(image.shape, dtype=np.uint8)

#     image = np.clip(image, low, high)
#     image = (image - low) / (high - low)
#     image = image * 255

#     return image.astype(np.uint8)


# def read_panoptic_mask(mask_path, remove_border_instances=True, background_id=0):
#     """
#     Read a panoptic mask while preserving integer IDs.
#     Optionally remove all instances that touch the image border.
#     """

#     mask = cv2.imread(
#         str(mask_path),
#         cv2.IMREAD_UNCHANGED,
#     )

#     if mask is None:
#         raise ValueError(f"Could not read mask: {mask_path}")

#     if mask.ndim == 3:
#         # This pipeline expects an integer-ID mask, not RGB colors
#         if (
#             np.array_equal(mask[:, :, 0], mask[:, :, 1])
#             and np.array_equal(mask[:, :, 1], mask[:, :, 2])
#         ):
#             mask = mask[:, :, 0]
#         else:
#             raise ValueError(
#                 f"{mask_path.name} is an RGB mask. "
#                 "It must first be converted to integer panoptic IDs."
#             )

#     mask = mask.astype(np.int64)

#     if remove_border_instances:
#         # IDs appearing on the image border
#         border_width = 10
        
#         border_ids = np.unique(
#             np.concatenate([
#                 mask[:border_width, :].ravel(),      # top
#                 mask[-border_width:, :].ravel(),     # bottom
#                 mask[:, :border_width].ravel(),      # left
#                 mask[:, -border_width:].ravel(),     # right
#             ])
#         )
#         # Keep background
#         border_ids = border_ids[border_ids != background_id]

#         if len(border_ids) > 0:
#             mask[np.isin(mask, border_ids)] = background_id

#     return mask
    
# def split_into_four(image, panoptic_mask, threshold=1240):
#     """
#     Return one full-size image/mask pair, or four matching quadrants.

#     Splitting is triggered when either image dimension is larger than the
#     threshold. Odd dimensions are preserved: the bottom/right tiles receive
#     the extra pixel.
#     """
#     height, width = image.shape[:2]

#     if height <= threshold and width <= threshold:
#         return [("full", image, panoptic_mask)]

#     middle_y = height // 2
#     middle_x = width // 2

#     tile_slices = {
#         "top_left": (slice(0, middle_y), slice(0, middle_x)),
#         "top_right": (slice(0, middle_y), slice(middle_x, width)),
#         "bottom_left": (slice(middle_y, height), slice(0, middle_x)),
#         "bottom_right": (slice(middle_y, height), slice(middle_x, width)),
#     }

#     return [
#         (tile_name, image[y_slice, x_slice], panoptic_mask[y_slice, x_slice])
#         for tile_name, (y_slice, x_slice) in tile_slices.items()
#     ]


# ============================================================
# Decode panoptic IDs and pool cell embeddings
# ============================================================

def decode_panoptic_id(panoptic_id, instance_divisor=1000):
    class_id = int(panoptic_id // instance_divisor)
    instance_id = int(panoptic_id % instance_divisor)

    return class_id, instance_id


def prepare_feature_map(feature):
    """
    Return feature as [C,H,W].
    """
    if feature.ndim == 4:
        if feature.shape[0] != 1:
            raise ValueError(
                f"Expected batch size 1, got {feature.shape}"
            )

        return feature[0]

    if feature.ndim == 3:
        return feature

    raise ValueError(
        f"Expected [1,C,H,W] or [C,H,W], "
        f"received {tuple(feature.shape)}"
    )


def extract_instances_from_mask(
    feature_map,
    panoptic_mask,
    image_name,
    image_position="full",
    instance_divisor=1000,
    min_original_pixels=200,
    min_feature_pixels=20,
):
    """
    Mean-pool contrast features inside each cell instance.
    """
    channels, feature_height, feature_width = feature_map.shape

    mask_tensor = torch.as_tensor(
        panoptic_mask.astype(np.float32),
        device=feature_map.device,
    )[None, None]

    resized_mask = F.interpolate(
        mask_tensor,
        size=(feature_height, feature_width),
        mode="nearest",
    )[0, 0].long()

    embeddings = []
    labels = []
    panoptic_ids = []
    instance_ids = []
    image_names = []
    image_positions=[]
    original_pixel_counts = []

    unique_ids = np.unique(panoptic_mask)

    for panoptic_id in unique_ids:
        panoptic_id = int(panoptic_id)

        if panoptic_id in BACKGROUND_IDS:
            continue

        class_id, instance_id = decode_panoptic_id(
            panoptic_id,
            instance_divisor=instance_divisor,
        )

        # Safety checks
        if class_id == 0:
            continue

        original_pixels = int(
            np.sum(panoptic_mask == panoptic_id)
        )

        if original_pixels < min_original_pixels:
            continue

        feature_region = resized_mask == panoptic_id
        feature_pixels = int(feature_region.sum())

        if feature_pixels < min_feature_pixels:
            continue

        # [C,H,W] -> [C,number of pixels] -> [C]
        embedding = feature_map[:, feature_region].mean(dim=1)

        # Appropriate for contrastive embeddings
        embedding = F.normalize(
            embedding,
            p=2,
            dim=0,
        )

        embeddings.append(
            embedding.detach().cpu().float().numpy()
        )

        labels.append(class_id)
        panoptic_ids.append(panoptic_id)
        instance_ids.append(instance_id)
        image_names.append(image_name)
        image_positions.append(image_position)
        original_pixel_counts.append(original_pixels)

    return {
        "embeddings": embeddings,
        "labels": labels,
        "panoptic_ids": panoptic_ids,
        "instance_ids": instance_ids,
        "image_names": image_names,
        "image_positions": image_positions,
        "pixel_counts": original_pixel_counts,
    }

###
import cv2
import numpy as np
from pathlib import Path


def get_four_crop_positions(height, width, crop_size=1200):
    """
    Return four corner-aligned crop positions:
    top-left, top-right, bottom-left, bottom-right.
    """
    if height < crop_size or width < crop_size:
        raise ValueError(
            f"Image ({height}, {width}) is smaller than "
            f"crop size {crop_size}"
        )

    return {
        "top_left": (0, 0),
        "top_right": (0, width - crop_size),
        "bottom_left": (height - crop_size, 0),
        "bottom_right": (
            height - crop_size,
            width - crop_size,
        ),
    }


def touches_border(binary_mask, border_width=10):
    """Check whether an object touches the crop border region."""
    return (
        binary_mask[:border_width, :].any()
        or binary_mask[-border_width:, :].any()
        or binary_mask[:, :border_width].any()
        or binary_mask[:, -border_width:].any()
    )


def crop_image_and_mask(
    image,
    mask,
    crop_size=1024,
    border_width=10,
    background_id=0,
):
    """
    Split an image into four overlapping crops.

    For an instance appearing in multiple crops:
    1. Keep it only in the crop containing its largest area.
    2. Remove it from the other crops.
    3. Remove it completely if the retained object touches
       the retained crop's border.

    Parameters
    ----------
    image : ndarray, shape (H, W) or (H, W, C)
    mask : ndarray, shape (H, W)
        Integer instance/panoptic mask.
    crop_size : int
        Output crop height and width.
    border_width : int
        Width of the excluded crop boundary.
    background_id : int
        Background value in the mask.

    Returns
    -------
    crops : dict
        Each value contains:
        {
            "image": cropped image,
            "mask": cleaned mask,
            "position": (y0, x0)
        }
    """
    height, width = mask.shape[:2]

    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"Image and mask dimensions do not match: "
            f"{image.shape[:2]} versus {mask.shape[:2]}"
        )

    # No cropping needed
    if height <= crop_size and width <= crop_size:
        cleaned_mask = mask.copy()

        for instance_id in np.unique(cleaned_mask):
            if instance_id == background_id:
                continue

            region = cleaned_mask == instance_id

            if touches_border(region, border_width):
                cleaned_mask[region] = background_id

        return {
            "full": {
                "image": image.copy(),
                "mask": cleaned_mask,
                "position": (0, 0),
            }
        }

    if height < crop_size or width < crop_size:
        raise ValueError(
            "Four 1200 × 1200 crops require both image "
            "dimensions to be at least 1200."
        )

    positions = get_four_crop_positions(
        height,
        width,
        crop_size,
    )

    crops = {}

    # First create all crops without removing anything
    for crop_name, (y0, x0) in positions.items():
        y1 = y0 + crop_size
        x1 = x0 + crop_size

        crops[crop_name] = {
            "image": image[y0:y1, x0:x1].copy(),
            "mask": mask[y0:y1, x0:x1].copy(),
            "position": (y0, x0),
        }

    instance_ids = np.unique(mask)
    instance_ids = instance_ids[instance_ids != background_id]

    for instance_id in instance_ids:
        areas = {}

        # Calculate the area of this instance in every crop
        for crop_name, crop_data in crops.items():
            areas[crop_name] = np.count_nonzero(
                crop_data["mask"] == instance_id
            )

        maximum_area = max(areas.values())

        if maximum_area == 0:
            continue

        # Keep the instance in only one crop
        best_crop_name = max(areas, key=areas.get)

        for crop_name, crop_data in crops.items():
            region = crop_data["mask"] == instance_id

            if crop_name != best_crop_name:
                crop_data["mask"][region] = background_id

        # Remove it if the retained region touches the crop border
        best_mask = crops[best_crop_name]["mask"]
        best_region = best_mask == instance_id

        if touches_border(best_region, border_width):
            best_mask[best_region] = background_id

    return crops




import numpy as np


def touches_border(binary_mask, border_width=10):
    return (
        binary_mask[:border_width, :].any()
        or binary_mask[-border_width:, :].any()
        or binary_mask[:, :border_width].any()
        or binary_mask[:, -border_width:].any()
    )


def get_crop_starts(length, crop_size, stride):
    """
    Generate crop starting positions while ensuring that the
    final crop reaches the image boundary.
    """
    if length <= crop_size:
        return [0]

    starts = list(range(0, length - crop_size + 1, stride))

    final_start = length - crop_size

    if starts[-1] != final_start:
        starts.append(final_start)

    return starts


def crop_image_and_mask(
    image,
    mask,
    crop_size=512,
    overlap=200,
    border_width=10,
    background_id=0,
):
    """
    Split an image and instance mask into overlapping square crops.

    For each instance:
    1. Find all crops containing the instance.
    2. Exclude crops where the instance touches the crop border.
    3. Among the remaining crops, keep the instance only in the
       crop containing its largest visible area.
    4. If it touches the border in every crop, remove it completely.

    Parameters
    ----------
    image : ndarray
        Shape (H, W) or (H, W, C).
    mask : ndarray
        Integer instance mask with shape (H, W).
    crop_size : int
        Crop width and height.
    overlap : int
        Overlap between adjacent crops in pixels.
    border_width : int
        Border region used to identify incomplete instances.
    background_id : int
        Background value in the mask.

    Returns
    -------
    crops : dict
        Dictionary containing cropped images, cleaned masks
        and original-image positions.
    """
    height, width = mask.shape[:2]

    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"Image and mask dimensions do not match: "
            f"{image.shape[:2]} versus {mask.shape[:2]}"
        )

    if overlap < 0 or overlap >= crop_size:
        raise ValueError(
            "overlap must satisfy: 0 <= overlap < crop_size"
        )

    if height < crop_size or width < crop_size:
        raise ValueError(
            f"Image size {(height, width)} is smaller than "
            f"crop size {crop_size}. Padding would be required."
        )

    stride = crop_size - overlap

    y_starts = get_crop_starts(
        height,
        crop_size,
        stride,
    )
    x_starts = get_crop_starts(
        width,
        crop_size,
        stride,
    )

    crops = {}

    # Create all crops
    for row_index, y0 in enumerate(y_starts):
        for column_index, x0 in enumerate(x_starts):
            y1 = y0 + crop_size
            x1 = x0 + crop_size

            crop_name = (
                f"row_{row_index:02d}_col_{column_index:02d}"
            )

            crops[crop_name] = {
                "image": image[y0:y1, x0:x1].copy(),
                "mask": mask[y0:y1, x0:x1].copy(),
                "position": (y0, x0),
            }

    instance_ids = np.unique(mask)
    instance_ids = instance_ids[
        instance_ids != background_id
    ]

    for instance_id in instance_ids:
        crop_information = {}

        # Check the instance in every crop
        for crop_name, crop_data in crops.items():
            region = crop_data["mask"] == instance_id
            area = int(region.sum())

            if area == 0:
                continue

            crop_information[crop_name] = {
                "area": area,
                "touches_border": touches_border(
                    region,
                    border_width,
                ),
            }

        # Crops where the instance is not cut by the border
        valid_crops = {
            crop_name: information
            for crop_name, information
            in crop_information.items()
            if not information["touches_border"]
        }

        if valid_crops:
            # Keep the largest complete occurrence
            best_crop_name = max(
                valid_crops,
                key=lambda name: valid_crops[name]["area"],
            )
        else:
            # The instance is incomplete in every crop
            best_crop_name = None

        # Remove all redundant or incomplete occurrences
        for crop_name, crop_data in crops.items():
            if crop_name != best_crop_name:
                crop_data["mask"][
                    crop_data["mask"] == instance_id
                ] = background_id

    return crops

import numpy as np
import pandas as pd


def touches_border(binary_mask, border_width=10):
    return (
        binary_mask[:border_width, :].any()
        or binary_mask[-border_width:, :].any()
        or binary_mask[:, :border_width].any()
        or binary_mask[:, -border_width:].any()
    )


def check_masks_across_crops(
    crops,
    background_id=0,
    border_width=10,
):
    """
    Check whether each instance:
    - appears in more than two crops;
    - touches the border in any crop;
    - touches the border in its largest-area crop.

    Parameters
    ----------
    crops : dict
        Output from crop_image_and_mask(), before duplicate removal.
        Each item must contain crops[name]["mask"].
    background_id : int
        Background value.
    border_width : int
        Border width in pixels.

    Returns
    -------
    pandas.DataFrame
    """
    instance_ids = set()

    for crop_data in crops.values():
        ids = np.unique(crop_data["mask"])
        instance_ids.update(ids.tolist())

    instance_ids.discard(background_id)

    records = []

    for instance_id in sorted(instance_ids):
        present_crops = []
        border_crops = []
        crop_areas = {}

        for crop_name, crop_data in crops.items():
            region = crop_data["mask"] == instance_id
            area = int(region.sum())

            if area == 0:
                continue

            present_crops.append(crop_name)
            crop_areas[crop_name] = area

            if touches_border(region, border_width):
                border_crops.append(crop_name)

        best_crop = max(crop_areas, key=crop_areas.get)
        best_crop_touches_border = best_crop in border_crops

        records.append({
            "instance_id": instance_id,
            "number_of_crops": len(present_crops),
            "more_than_two_crops": len(present_crops) > 2,
            "present_crops": present_crops,
            "crop_areas": crop_areas,
            "best_crop": best_crop,
            "best_crop_area": crop_areas[best_crop],
            "touches_any_border": len(border_crops) > 0,
            "border_crops": border_crops,
            "best_crop_touches_border":
                best_crop_touches_border,
        })

    return pd.DataFrame(records)


# ============================================================
# Process folders
# ============================================================

@torch.no_grad()
def extract_embeddings_from_folders(
    predictor,
    image_folder,
    mask_folder,
    target_layer,
    max_images=None,
    start_index=0,
    end_index=-1
):
    model = predictor.model
    model.eval()

    pairs = collect_image_mask_pairs(
        image_folder=image_folder,
        mask_folder=mask_folder,
    )[start_index:end_index]

    if max_images is not None:
        pairs = pairs[:max_images]

    feature_hook = FeatureHook(
        model=model,
        layer_name=target_layer,
    )

    collected = {
        "embeddings": [],
        "labels": [],
        "panoptic_ids": [],
        "instance_ids": [],
        "image_names": [],
        "image_positions": [],
        "pixel_counts": [],
    }

    try:
        for image_index, (image_path, mask_path) in enumerate(pairs):
            image = imread(image_path)
            # image = np.stack((image[:,:,0],)*3, axis=-1)
            panoptic_mask = imread(mask_path)

            if image.shape[:2] != panoptic_mask.shape:
                raise ValueError(
                    f"Image-mask size mismatch for {image_path.name}:\n"
                    f"image: {image.shape[:2]}\n"
                    f"mask:  {panoptic_mask.shape}"
                )

            crops = crop_image_and_mask(image, panoptic_mask)
            for tile_name, item in crops.items():
            # for tile_name, image_tile, mask_tile in tiles:
                feature_hook.clear()

                # DefaultPredictor handles resizing and normalization.
                # Each tile is predicted independently.
                image_tile = item["image"]
                mask_tile = item["mask"]
                _ = predictor(image_tile)

                if feature_hook.feature is None:
                    raise RuntimeError(
                        f"No feature captured from {target_layer}.\n"
                        "The contrast head is probably only executed during "
                        "training. In that case its forward method must be "
                        "changed to calculate the contrast feature during "
                        "inference as well."
                    )

                feature_map = prepare_feature_map(
                    feature_hook.feature
                )

                if image_index == 0:
                    print(
                        f"Tile {tile_name}: image={image_tile.shape}, "
                        f"mask={mask_tile.shape}, "
                        f"feature={tuple(feature_map.shape)}"
                    )

                image_results = extract_instances_from_mask(
                    feature_map=feature_map,
                    panoptic_mask=mask_tile,
                    image_name=image_path.name,
                    image_position=tile_name,
                    instance_divisor=INSTANCE_DIVISOR,
                    min_original_pixels=MIN_ORIGINAL_PIXELS,
                    min_feature_pixels=MIN_FEATURE_PIXELS,
                )

                for key in collected:
                    collected[key].extend(image_results[key])

            if (image_index + 1) % 10 == 0:
                print(
                    f"Processed {image_index + 1}/{len(pairs)} images; "
                    f"collected {len(collected['embeddings'])} cells."
                )

    finally:
        feature_hook.close()

    if not collected["embeddings"]:
        raise RuntimeError(
            "No embeddings were extracted. Check the mask encoding, "
            "class IDs and minimum pixel settings."
        )

    collected["embeddings"] = np.stack(
        collected["embeddings"]
    )

    collected["labels"] = np.asarray(
        collected["labels"],
        dtype=np.int64,
    )

    collected["panoptic_ids"] = np.asarray(
        collected["panoptic_ids"],
        dtype=np.int64,
    )

    collected["instance_ids"] = np.asarray(
        collected["instance_ids"],
        dtype=np.int64,
    )

    collected["image_names"] = np.asarray(
        collected["image_names"],
    )

    collected["image_positions"] = np.asarray(
        collected["image_positions"],
    )

    collected["pixel_counts"] = np.asarray(
        collected["pixel_counts"],
        dtype=np.int64,
    )

    print("\nFinished:")
    print("Embeddings:", collected["embeddings"].shape)
    print(
        "Phenotype counts:",
        np.unique(
            collected["labels"],
            return_counts=True,
        ),
    )

    return collected


# ============================================================
# t-SNE
# ============================================================

def calculate_tsne(
    embeddings,
    perplexity=30,
    random_state=42,
):
    embeddings = normalize(
        embeddings,
        norm="l2",
        axis=1,
    )

    n_samples, n_features = embeddings.shape

    if n_samples < 3:
        raise ValueError(
            f"t-SNE requires more samples; only {n_samples} found."
        )

    pca_components = min(
        50,
        n_samples - 1,
        n_features,
    )

    if pca_components >= 2:
        embeddings_reduced = PCA(
            n_components=pca_components,
            random_state=random_state,
        ).fit_transform(embeddings)
    else:
        embeddings_reduced = embeddings

    # Must satisfy perplexity < n_samples
    perplexity = min(
        perplexity,
        max(2, (n_samples - 1) // 3),
    )
    perplexity = min(perplexity, n_samples - 1)

    print(
        f"Running t-SNE: samples={n_samples}, "
        f"dimensions={embeddings_reduced.shape[1]}, "
        f"perplexity={perplexity}"
    )

    coordinates = TSNE(
        n_components=2,
        perplexity=perplexity,
        metric="cosine",
        init="random",
        learning_rate="auto",
        n_iter=1500,             # older sklearn
        random_state=random_state,
    ).fit_transform(embeddings_reduced)

    return coordinates


def plot_tsne(
    coordinates,
    labels,
    output_path=None,
    cutoff_number=500,
    random_state=42,
    class_ids=range(1, 7),
):
    """
    Plot a publication-style t-SNE figure.

    Parameters
    ----------
    coordinates : array-like, shape (n_samples, 2)
        Calculated t-SNE coordinates.

    labels : array-like, shape (n_samples,)
        Class labels.

    output_path : str or Path, optional
        Save as PDF/SVG for vector output or PNG/TIFF at 600 dpi.

    cutoff_number : int or None
        Maximum number of displayed samples per class.
        Use None to display all samples.

    random_state : int
        Seed used for reproducible random subsampling.

    class_ids : iterable
        Classes to display.
    """
    coordinates = np.asarray(coordinates)
    labels = np.asarray(labels)

    if coordinates.shape[0] != labels.shape[0]:
        raise ValueError(
            "coordinates and labels must contain the same "
            "number of samples."
        )

    rng = np.random.default_rng(random_state)

    fig, ax = plt.subplots(
        figsize=(3.5, 3.2),
        dpi=300,
    )

    for class_id in class_ids:
        all_indices = np.flatnonzero(labels == class_id)
        total_number = len(all_indices)

        if total_number == 0:
            continue

        if (
            cutoff_number is not None
            and total_number > cutoff_number
        ):
            plot_indices = rng.choice(
                all_indices,
                size=cutoff_number,
                replace=False,
            )
        else:
            plot_indices = all_indices

        phenotype_name = PHENOTYPE_NAMES.get(
            int(class_id),
            f"Class {class_id}",
        )
        color = PHENOTYPE_COLORS.get(
            int(class_id),
            "#808080",
        )

        ax.scatter(
            coordinates[plot_indices, 0],
            coordinates[plot_indices, 1],
            s=8,
            color=color,
            alpha=0.7,
            edgecolors="none",
            rasterized=True,
            label=(
                f"{phenotype_name} "
                f"(shown {len(plot_indices):,}/{total_number:,})"
            ),
        )

    ax.set_xlabel("t-SNE 1", fontsize=8)
    ax.set_ylabel("t-SNE 2", fontsize=8)

    # Absolute t-SNE coordinate values are not interpretable
    ax.set_xticks([])
    ax.set_yticks([])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    ax.legend(
        frameon=False,
        fontsize=7,
        markerscale=1.5,
        handletextpad=0.4,
        labelspacing=0.35,
        borderaxespad=0,
        loc="upper left",
        # bbox_to_anchor=(1.02, 1),
    )

    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        save_kwargs = {
            "bbox_inches": "tight",
            "pad_inches": 0.03,
            "transparent": True,
        }

        if output_path.suffix.lower() in {
            ".png",
            ".tif",
            ".tiff",
        }:
            save_kwargs["dpi"] = 600

        fig.savefig(
            output_path,
            **save_kwargs,
        )

    # plt.show()

    return fig, ax


def get_original_name(image_name):
    """Remove crop-position suffix from a filename."""
    for suffix in (
        "_top_left",
        "_top_right",
        "_bottom_left",
        "_bottom_right",
    ):
        image_name = image_name.replace(suffix, "")
    return image_name