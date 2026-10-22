from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scipy import ndimage as ndi

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from skimage.io import imread
from skimage.morphology import remove_small_holes

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

def touches_border(binary_mask, border_width=10):
    """Check whether an object touches the crop border region."""
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

def extract_image_embeddings_from_folder(
    predictor,
    image_folder,
    output_folder,
    target_layer,
    max_images=None,
    start_index=0,
    end_index=None,
):
    """Save the full [C, H_feature, W_feature] embedding for each image.

    No masks, instance pooling, or phenotype filtering are used. Images are
    passed whole to the predictor, which handles its usual preprocessing.
    Returns a mapping from image filename to saved .npy path. Unlike the
    mask-based function, end_index=None includes the final image.
    """
    image_folder = Path(image_folder)
    output_folder = Path(output_folder)
    image_paths = sorted(
        path for path in image_folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )[start_index:end_index]
    if max_images is not None:
        if max_images < 0:
            raise ValueError("max_images must be nonnegative or None")
        image_paths = image_paths[:max_images]
    if not image_paths:
        raise ValueError(f"No input images selected from {image_folder}")

    output_folder.mkdir(parents=True, exist_ok=True)
    predictor.model.eval()
    feature_hook = FeatureHook(
        model=predictor.model,
        layer_name=target_layer,
    )
    saved_paths = {}
    try:
        with torch.no_grad():
            for image_index, image_path in enumerate(image_paths):
                image = imread(image_path)
                feature_hook.clear()
                predictor(image)
                if feature_hook.feature is None:
                    raise RuntimeError(
                        f"No feature captured from {target_layer}. "
                        "Ensure this layer executes during inference."
                    )
                embedding = prepare_feature_map(
                    feature_hook.feature
                ).detach().cpu().numpy()
                # Include the image extension to avoid same-stem collisions.
                output_path = output_folder / f"{image_path.name}.npy"
                np.save(output_path, embedding, allow_pickle=False)
                saved_paths[image_path.name] = output_path
                if (image_index + 1) % 10 == 0:
                    print(f"Saved {image_index + 1}/{len(image_paths)} embeddings.")
    finally:
        feature_hook.close()

    print(f"Saved {len(saved_paths)} image embeddings to {output_folder}")
    return saved_paths


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


def calculate_umap(
    embeddings,
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric="cosine",
    random_state=42,
):
    """
    Project cell embeddings to 2D with UMAP for visualizing their
    distribution and spotting candidate sub-classes.

    Requires the `umap-learn` package (``pip install umap-learn``).
    """
    try:
        import umap
    except ImportError as error:
        raise ImportError(
            "UMAP projection requires the 'umap-learn' package. "
            "Install it with: pip install umap-learn"
        ) from error

    embeddings = normalize(embeddings, norm="l2", axis=1)

    n_samples = embeddings.shape[0]
    if n_samples < 3:
        raise ValueError(
            f"UMAP requires more samples; only {n_samples} found."
        )

    neighbors = min(n_neighbors, n_samples - 1)

    print(
        f"Running UMAP: samples={n_samples}, "
        f"n_neighbors={neighbors}, min_dist={min_dist}, metric={metric}"
    )

    reducer = umap.UMAP(
        n_neighbors=neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=random_state,
    )

    return reducer.fit_transform(embeddings)


def evaluate_kmeans_k(
    embeddings,
    k_values=range(2, 15),
    random_state=42,
    sample_size=5000,
):
    """
    Sweep KMeans cluster counts to help choose how many sub-classes
    are present in a population of cell embeddings.

    Returns a DataFrame with inertia and silhouette score per k.
    Silhouette is computed on a random subsample (`sample_size`) of
    cells for speed on large datasets.
    """
    embeddings = normalize(embeddings, norm="l2", axis=1)
    n_samples = embeddings.shape[0]

    rng = np.random.default_rng(random_state)
    if n_samples > sample_size:
        sample_indices = rng.choice(
            n_samples, size=sample_size, replace=False
        )
    else:
        sample_indices = np.arange(n_samples)

    records = []
    for k in k_values:
        if k < 2 or k >= n_samples:
            continue

        kmeans = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=10,
        ).fit(embeddings)

        silhouette = silhouette_score(
            embeddings[sample_indices],
            kmeans.labels_[sample_indices],
            metric="cosine",
        )

        records.append({
            "k": k,
            "inertia": kmeans.inertia_,
            "silhouette": silhouette,
        })

        print(f"k={k}: inertia={kmeans.inertia_:.2f}, silhouette={silhouette:.3f}")

    return pd.DataFrame(records)


def plot_kmeans_k_sweep(sweep_results, output_path=None):
    """Plot inertia (elbow) and silhouette score against k side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 3), dpi=150)

    axes[0].plot(sweep_results["k"], sweep_results["inertia"], marker="o")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow plot")

    axes[1].plot(sweep_results["k"], sweep_results["silhouette"], marker="o", color="#E45756")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette score (cosine)")
    axes[1].set_title("Cluster separation")

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight", dpi=200)

    return fig, axes


def cluster_embeddings(
    embeddings,
    n_clusters=8,
    random_state=42,
):
    """
    Cluster L2-normalized cell embeddings with KMeans to propose
    sub-classes within (or across) the existing phenotype labels.

    Returns
    -------
    cluster_ids : ndarray, shape (n_samples,)
        Zero-based cluster assignment for each cell.
    kmeans : sklearn.cluster.KMeans
        The fitted model (inspect `.cluster_centers_` if needed).
    """
    normalized = normalize(embeddings, norm="l2", axis=1)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    ).fit(normalized)

    return kmeans.labels_, kmeans


def _qualitative_palette(n_colors):
    """Return n_colors distinct hex colors from a qualitative colormap."""
    cmap_name = "tab10" if n_colors <= 10 else "tab20"
    cmap = plt.get_cmap(cmap_name)
    return [
        "#%02x%02x%02x" % tuple(int(255 * c) for c in cmap(i % cmap.N)[:3])
        for i in range(n_colors)
    ]


def plot_embedding_scatter(
    coordinates,
    labels,
    label_names=None,
    label_colors=None,
    axis_label="Component",
    output_path=None,
    cutoff_number=500,
    random_state=42,
    class_ids=None,
    highlight_indices=None,
    ax=None,
):
    """
    Plot a publication-style 2D scatter of cell embeddings (t-SNE,
    UMAP, or any other 2D projection), colored by a discrete label.

    Parameters
    ----------
    coordinates : array-like, shape (n_samples, 2)
    labels : array-like, shape (n_samples,)
        Any discrete label: phenotype id, KMeans cluster id, etc.
    label_names : dict, optional
        Maps label value -> display name. If None, labels are shown
        as plain "Class {id}" text (pass PHENOTYPE_NAMES explicitly
        to show phenotype names).
    label_colors : dict, optional
        Maps label value -> hex color. If None, an auto-generated
        qualitative palette is used (pass PHENOTYPE_COLORS explicitly
        to use the fixed phenotype palette).
    axis_label : str
        Prefix used for the x/y axis labels, e.g. "t-SNE" or "UMAP".
    cutoff_number : int or None
        Maximum number of displayed samples per class/cluster.
        Use None to display all samples.
    class_ids : iterable, optional
        Labels to display, in legend order. Defaults to all unique
        values found in `labels`.
    highlight_indices : array-like of int, optional
        Row indices to redraw on top with a black outline, e.g. to
        show where a specific image's cells, or a selected
        sub-cluster, sit within the full distribution.
    ax : matplotlib.axes.Axes, optional
        Draw into an existing axes instead of creating a new figure.
    """
    coordinates = np.asarray(coordinates)
    labels = np.asarray(labels)

    if coordinates.shape[0] != labels.shape[0]:
        raise ValueError(
            "coordinates and labels must contain the same "
            "number of samples."
        )

    if class_ids is None:
        class_ids = np.unique(labels)

    class_ids = list(class_ids)

    if label_colors is None:
        auto_colors = dict(zip(class_ids, _qualitative_palette(len(class_ids))))

    rng = np.random.default_rng(random_state)

    if ax is None:
        fig, ax = plt.subplots(figsize=(3.5, 3.2), dpi=300)
    else:
        fig = ax.figure

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

        if label_names is not None:
            display_name = label_names.get(class_id, f"Class {class_id}")
        else:
            display_name = f"Class {class_id}"

        if label_colors is not None:
            color = label_colors.get(class_id, "#808080")
        else:
            color = auto_colors[class_id]

        ax.scatter(
            coordinates[plot_indices, 0],
            coordinates[plot_indices, 1],
            s=8,
            color=color,
            alpha=0.7,
            edgecolors="none",
            rasterized=True,
            label=(
                f"{display_name} "
                f"(shown {len(plot_indices):,}/{total_number:,})"
            ),
        )

    if highlight_indices is not None:
        highlight_indices = np.asarray(highlight_indices)
        ax.scatter(
            coordinates[highlight_indices, 0],
            coordinates[highlight_indices, 1],
            s=24,
            facecolors="none",
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
            label=f"Highlighted ({len(highlight_indices):,})",
        )

    ax.set_xlabel(f"{axis_label} 1", fontsize=8)
    ax.set_ylabel(f"{axis_label} 2", fontsize=8)

    # Absolute projection coordinate values are not interpretable
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

    return fig, ax


def plot_tsne(
    coordinates,
    labels,
    output_path=None,
    cutoff_number=500,
    random_state=42,
    class_ids=range(1, 7),
    label_names=None,
    label_colors=None,
    **kwargs,
):
    """
    Backward-compatible wrapper around plot_embedding_scatter(), colored
    by phenotype (PHENOTYPE_NAMES/PHENOTYPE_COLORS) by default.
    """
    return plot_embedding_scatter(
        coordinates=coordinates,
        labels=labels,
        label_names=PHENOTYPE_NAMES if label_names is None else label_names,
        label_colors=PHENOTYPE_COLORS if label_colors is None else label_colors,
        axis_label="t-SNE",
        output_path=output_path,
        cutoff_number=cutoff_number,
        random_state=random_state,
        class_ids=class_ids,
        **kwargs,
    )


def plot_umap(
    coordinates,
    labels,
    output_path=None,
    cutoff_number=None,
    random_state=42,
    class_ids=None,
    label_names=None,
    label_colors=None,
    **kwargs,
):
    """
    Convenience wrapper around plot_embedding_scatter() for UMAP
    coordinates. By default colors use an auto-generated palette, so
    this works equally well for phenotype labels or KMeans cluster ids
    -- pass label_names=PHENOTYPE_NAMES, label_colors=PHENOTYPE_COLORS
    explicitly to use the fixed phenotype palette instead.
    """
    return plot_embedding_scatter(
        coordinates=coordinates,
        labels=labels,
        label_names=label_names,
        label_colors=label_colors,
        axis_label="UMAP",
        output_path=output_path,
        cutoff_number=cutoff_number,
        random_state=random_state,
        class_ids=class_ids,
        **kwargs,
    )


def postprocess_cell_mask(
    cell_mask,
    fill_holes=True,
    hole_area_threshold=None,
    remove_if_touches_border=True,
    border_width=1,
):
    """
    Smooth a single predicted cell's binary mask before it is scored
    and saved: fill interior holes left by noisy per-pixel predictions,
    and drop cells that are truncated by the image border.

    Parameters
    ----------
    cell_mask : ndarray(bool), shape (H, W)
        One instance's raw binary mask (e.g. predicted_mask == segment_id).
    fill_holes : bool
        Fill enclosed background pixels inside the cell.
    hole_area_threshold : int, optional
        None (default) fills every enclosed hole, via
        scipy.ndimage.binary_fill_holes. Pass an integer instead to
        only fill holes up to that many pixels (skimage's
        remove_small_holes) -- use this if a phenotype can legitimately
        contain another instance inside it (e.g. spores inside an
        ascus), since a large "hole" there is really a neighboring
        cell, not segmentation noise.
    remove_if_touches_border : bool
        Drop cells that touch the image border: they are truncated by
        the field of view and not full cells.
    border_width : int
        How many edge pixels count as "the border". Use 1 for the real
        image edge; a larger value also drops cells sitting very close
        to it.

    Returns
    -------
    mask : ndarray(bool), shape (H, W)
        The smoothed mask, or all-False if the cell was dropped.
    kept : bool
        False if the cell was dropped for touching the border.
    """
    cell_mask = np.asarray(cell_mask, dtype=bool)

    if fill_holes and cell_mask.any():
        if hole_area_threshold is None:
            cell_mask = ndi.binary_fill_holes(cell_mask)
        else:
            cell_mask = remove_small_holes(
                cell_mask, area_threshold=hole_area_threshold
            )

    if remove_if_touches_border and touches_border(cell_mask, border_width):
        return np.zeros_like(cell_mask), False

    return cell_mask, True


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


def extract_cell_embeddings_from_folder(
    predictor,
    image_folder,
    target_layer,
    output_folder="./cell_embeddings",
    max_images=None,
    start_index=0,
    end_index=None,
    min_original_pixels=20,
    min_feature_pixels=2,
    instance_divisor=1000,
    category_id_offset=0,
    overwrite=False,
    fill_holes=True,
    hole_area_threshold=None,
    remove_border_cells=True,
    border_width=1,
):
    """
    Save per-cell embeddings and their corresponding full-resolution masks.

    This is the main extraction entry point: for every image in
    `image_folder` it runs the panoptic predictor, saves a full-resolution
    instance mask, and pools the contrastive feature map into one
    embedding per cell, together with its predicted phenotype and other
    bookkeeping fields needed to later plot "original image + embedding
    distribution" figures.

    Each cell's raw predicted mask is first smoothed with
    postprocess_cell_mask(): interior holes are filled, and cells that
    touch the image border (truncated by the field of view) are
    dropped entirely -- see `fill_holes`, `hole_area_threshold`,
    `remove_border_cells`, `border_width` below. Pixel counts, bboxes,
    centroids, the pooled embedding, and the saved mask are all
    computed from the smoothed mask.

    Output:
        output_folder/cells.npz
        output_folder/masks/<original_filename>.cells.npy

    cells.npz fields (one row per cell):
        embeddings:       [N, C], L2-normalized cell embeddings
        cell_ids:         [N], globally unique positive IDs
        label_ids:        [N], predicted phenotype IDs
        segment_ids:      [N], original predictor segment IDs
        image_names:      [N], source filenames (relative to image_folder)
        image_paths:      [N], absolute source image paths
        mask_files:       [N], mask paths relative to output_folder
        pixel_counts:     [N], full-resolution cell areas
        bboxes_yxyx:      [N, 4], (ymin, xmin, ymax, xmax);
                          maximum coordinates are exclusive
        centroids_yx:     [N, 2], (y, x) center of each cell's bbox
        image_heights:    [N], source image height in pixels
        image_widths:     [N], source image width in pixels

    Each saved mask has shape [H, W]:
        0 = background or excluded cell
        cell_id = the corresponding saved cell

    category_id_offset:
        Applied only when segments_info is provided.
        Use 1 if predictor categories are 0–5 and phenotype IDs are 1–6.

    When segments_info is None, assumes:
        segment_id = label_id * instance_divisor + instance_id.

    overwrite:
        If False (default), refuses to run when output_folder already
        contains a previous extraction. Set True to replace it.
    """
    image_folder = Path(image_folder)
    output_folder = Path(output_folder)
    mask_folder = output_folder / "masks"

    files = sorted(
        p for p in image_folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in {
            ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"
        }
    )[start_index:end_index]

    if max_images is not None:
        if max_images < 0:
            raise ValueError("max_images must be nonnegative.")
        files = files[:max_images]

    if not files:
        raise ValueError("No input images selected.")

    # Protect an existing extraction from partial overwriting.
    output_folder.mkdir(parents=True, exist_ok=True)
    existing_extraction = (
        (output_folder / "cells.npz").exists() or mask_folder.exists()
    )
    if existing_extraction and not overwrite:
        raise FileExistsError(
            f"Extraction already exists in {output_folder}. "
            "Choose a new output_folder, or pass overwrite=True."
        )
    if existing_extraction and overwrite:
        if (output_folder / "cells.npz").exists():
            (output_folder / "cells.npz").unlink()
        if mask_folder.exists():
            for mask_file in mask_folder.glob("*.npy"):
                mask_file.unlink()
    mask_folder.mkdir(exist_ok=True)

    collected = {
        "embeddings": [],
        "cell_ids": [],
        "label_ids": [],
        "segment_ids": [],
        "image_names": [],
        "image_paths": [],
        "mask_files": [],
        "pixel_counts": [],
        "bboxes_yxyx": [],
        "centroids_yx": [],
        "image_heights": [],
        "image_widths": [],
    }

    predictor.model.eval()
    hook = FeatureHook(
        model=predictor.model,
        layer_name=target_layer,
    )

    next_cell_id = 1

    try:
        with torch.no_grad():
            for image_index, image_path in enumerate(files):
                image = imread(image_path)

                hook.clear()
                outputs = predictor(image)

                if hook.feature is None:
                    raise RuntimeError(
                        f"No feature captured from {target_layer}."
                    )
                if "panoptic_seg" not in outputs:
                    raise RuntimeError(
                        "Predictor must return outputs['panoptic_seg']."
                    )

                feature_map = prepare_feature_map(
                    hook.feature
                ).detach().float().clone()

                predicted_mask, segments_info = outputs["panoptic_seg"]
                predicted_mask = torch.as_tensor(
                    predicted_mask,
                    device=feature_map.device,
                ).long()

                if tuple(predicted_mask.shape) != tuple(image.shape[:2]):
                    raise ValueError(
                        f"Mask/image size mismatch for {image_path.name}: "
                        f"{tuple(predicted_mask.shape)} vs {image.shape[:2]}"
                    )

                if segments_info is not None:
                    segments = [
                        (
                            int(segment["id"]),
                            int(segment["category_id"]) + category_id_offset,
                        )
                        for segment in segments_info
                        if segment.get("isthing", True)
                    ]
                else:
                    segments = [
                        (
                            int(segment_id),
                            int(segment_id) // instance_divisor,
                        )
                        for segment_id in torch.unique(
                            predicted_mask
                        ).cpu().tolist()
                        if int(segment_id) > 0
                        and int(segment_id) // instance_divisor > 0
                    ]

                original_mask = predicted_mask.cpu().numpy()
                saved_mask = np.zeros(original_mask.shape, dtype=np.int64)
                relative_mask_path = (
                    Path("masks") / f"{image_path.name}.cells.npy"
                )
                image_cell_count = 0

                for segment_id, label_id in segments:
                    raw_cell_mask = original_mask == segment_id

                    cell_mask_np, kept = postprocess_cell_mask(
                        raw_cell_mask,
                        fill_holes=fill_holes,
                        hole_area_threshold=hole_area_threshold,
                        remove_if_touches_border=remove_border_cells,
                        border_width=border_width,
                    )

                    if not kept:
                        continue

                    pixel_count = int(cell_mask_np.sum())

                    if pixel_count < min_original_pixels:
                        continue

                    cell_mask = torch.as_tensor(
                        cell_mask_np, device=feature_map.device
                    )

                    feature_mask = F.interpolate(
                        cell_mask[None, None].float(),
                        size=feature_map.shape[-2:],
                        mode="nearest",
                    )[0, 0].bool()

                    if int(feature_mask.sum().item()) < min_feature_pixels:
                        continue

                    embedding = feature_map[:, feature_mask].mean(dim=1)
                    embedding = F.normalize(embedding, p=2, dim=0)

                    yy, xx = np.nonzero(cell_mask_np)
                    bbox = (
                        yy.min(),
                        xx.min(),
                        yy.max() + 1,
                        xx.max() + 1,
                    )
                    centroid = (
                        (bbox[0] + bbox[2]) / 2,
                        (bbox[1] + bbox[3]) / 2,
                    )

                    cell_id = next_cell_id
                    next_cell_id += 1
                    saved_mask[cell_mask_np] = cell_id

                    collected["embeddings"].append(
                        embedding.cpu().numpy()
                    )
                    collected["cell_ids"].append(cell_id)
                    collected["label_ids"].append(label_id)
                    collected["segment_ids"].append(segment_id)
                    collected["image_names"].append(image_path.name)
                    collected["image_paths"].append(str(image_path.resolve()))
                    collected["mask_files"].append(
                        relative_mask_path.as_posix()
                    )
                    collected["pixel_counts"].append(pixel_count)
                    collected["bboxes_yxyx"].append(bbox)
                    collected["centroids_yx"].append(centroid)
                    collected["image_heights"].append(image.shape[0])
                    collected["image_widths"].append(image.shape[1])
                    image_cell_count += 1

                np.save(
                    output_folder / relative_mask_path,
                    saved_mask,
                    allow_pickle=False,
                )

                print(
                    f"{image_index + 1}/{len(files)}: "
                    f"{image_path.name}: {image_cell_count} cells"
                )

    finally:
        hook.close()

    if not collected["embeddings"]:
        raise RuntimeError(
            "No cells extracted. Check predictions and size thresholds."
        )

    collected["embeddings"] = np.stack(collected["embeddings"])

    for key in (
        "cell_ids", "label_ids", "segment_ids",
        "pixel_counts", "bboxes_yxyx",
        "image_heights", "image_widths",
    ):
        collected[key] = np.asarray(collected[key], dtype=np.int64)

    collected["centroids_yx"] = np.asarray(
        collected["centroids_yx"], dtype=np.float32
    )

    for key in ("image_names", "image_paths", "mask_files"):
        collected[key] = np.asarray(collected[key], dtype=str)

    np.savez_compressed(output_folder / "cells.npz", **collected)

    print(
        f"Saved {len(collected['cell_ids'])} cells "
        f"to {output_folder / 'cells.npz'}"
    )
    return collected


def load_cell_table(output_folder):
    """
    Reload the output of extract_cell_embeddings_from_folder() for
    downstream analysis (UMAP, clustering, plotting) without needing the
    model or predictor again -- this is all a second machine needs,
    together with the `masks/` folder and (optionally) the original
    images referenced by `image_paths`/`image_names`.

    Returns
    -------
    table : pandas.DataFrame
        One row per cell, with embeddings excluded (use `embeddings`
        below, same row order) so the table stays CSV-friendly.
    embeddings : ndarray, shape (N, C)
        L2-normalized cell embeddings, aligned with `table`.
    """
    output_folder = Path(output_folder)
    with np.load(output_folder / "cells.npz") as data:
        table = pd.DataFrame({
            "cell_id": data["cell_ids"],
            "label_id": data["label_ids"],
            "segment_id": data["segment_ids"],
            "image_name": data["image_names"],
            "image_path": data["image_paths"],
            "mask_file": data["mask_files"],
            "pixel_count": data["pixel_counts"],
            "bbox_ymin": data["bboxes_yxyx"][:, 0],
            "bbox_xmin": data["bboxes_yxyx"][:, 1],
            "bbox_ymax": data["bboxes_yxyx"][:, 2],
            "bbox_xmax": data["bboxes_yxyx"][:, 3],
            "centroid_y": data["centroids_yx"][:, 0],
            "centroid_x": data["centroids_yx"][:, 1],
            "image_height": data["image_heights"],
            "image_width": data["image_widths"],
        })
        embeddings = data["embeddings"]

    table["phenotype_name"] = table["label_id"].map(
        lambda label_id: PHENOTYPE_NAMES.get(int(label_id), f"Class {label_id}")
    )

    return table, embeddings


def add_projection_and_clusters(
    table,
    embeddings,
    method="umap",
    n_clusters=None,
    random_state=42,
    **projection_kwargs,
):
    """
    Compute a 2D embedding projection (and, optionally, KMeans
    sub-classes) and add them as columns to `table`.

    Parameters
    ----------
    table : pandas.DataFrame
        As returned by load_cell_table(); modified copy is returned.
    embeddings : ndarray, shape (N, C)
        Aligned with `table` (same row order).
    method : {"umap", "tsne"}
    n_clusters : int, optional
        If given, also runs cluster_embeddings() and adds a
        "cluster_id" column with proposed sub-classes.

    Returns
    -------
    table : pandas.DataFrame
        Copy of the input with "{method}_1"/"{method}_2" columns added
        (and "cluster_id" if n_clusters was given).
    """
    table = table.copy()

    if method == "umap":
        coordinates = calculate_umap(embeddings, random_state=random_state, **projection_kwargs)
    elif method == "tsne":
        coordinates = calculate_tsne(embeddings, random_state=random_state, **projection_kwargs)
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'umap' or 'tsne'.")

    table[f"{method}_1"] = coordinates[:, 0]
    table[f"{method}_2"] = coordinates[:, 1]

    if n_clusters is not None:
        cluster_ids, _ = cluster_embeddings(
            embeddings, n_clusters=n_clusters, random_state=random_state
        )
        table["cluster_id"] = cluster_ids

    return table


def plot_cell_embedding_distribution(
    output_folder,
    table=None,
    embeddings=None,
    figure_path=None,
    method="umap",
    color_by="label_id",
    class_ids=None,
    random_state=42,
    **projection_kwargs,
):
    """
    Plot the cell-embedding distribution with UMAP (default) or t-SNE,
    colored by phenotype or by a KMeans cluster id.

    table, embeddings : optional
        Reuse an already-loaded/annotated table (e.g. from
        load_cell_table() + add_projection_and_clusters(), with a
        "cluster_id" column) instead of reloading from disk.
    color_by : {"label_id", "cluster_id", ...}
        Any column present in `table` to color points by. Phenotype
        names/colors are used automatically for "label_id"; any other
        column gets an auto-generated qualitative palette.
    """
    if table is None or embeddings is None:
        table, embeddings = load_cell_table(output_folder)

    if len(embeddings) < 3:
        raise ValueError("At least three cells are required for a projection.")

    if method == "umap":
        coordinates = calculate_umap(embeddings, random_state=random_state, **projection_kwargs)
        axis_label = "UMAP"
    elif method == "tsne":
        coordinates = calculate_tsne(embeddings, random_state=random_state, **projection_kwargs)
        axis_label = "t-SNE"
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'umap' or 'tsne'.")

    labels = table[color_by].to_numpy()

    if color_by == "label_id":
        label_names, label_colors = PHENOTYPE_NAMES, PHENOTYPE_COLORS
    else:
        label_names, label_colors = None, None

    fig, ax = plot_embedding_scatter(
        coordinates=coordinates,
        labels=labels,
        label_names=label_names,
        label_colors=label_colors,
        axis_label=axis_label,
        output_path=figure_path,
        cutoff_number=None,
        class_ids=class_ids if class_ids is not None else np.unique(labels),
    )

    table = table.copy()
    table[f"{method}_1"] = coordinates[:, 0]
    table[f"{method}_2"] = coordinates[:, 1]

    return table, coordinates


def crop_cell(image, mask, bbox_yxyx, cell_id=None, pad=20):
    """
    Crop `image`/`mask` around a bbox (ymin, xmin, ymax, xmax) with
    `pad` extra pixels on each side, clipped to image bounds.

    If `cell_id` is given, returns a boolean array marking that cell's
    pixels within the crop (useful to overlay just one cell instead of
    every object in the neighborhood).
    """
    height, width = image.shape[:2]
    ymin, xmin, ymax, xmax = bbox_yxyx

    y0 = max(0, int(ymin) - pad)
    x0 = max(0, int(xmin) - pad)
    y1 = min(height, int(ymax) + pad)
    x1 = min(width, int(xmax) + pad)

    image_crop = image[y0:y1, x0:x1]
    mask_crop = mask[y0:y1, x0:x1]

    if cell_id is not None:
        return image_crop, mask_crop == cell_id

    return image_crop, mask_crop


def plot_cell_gallery(
    table,
    cell_ids,
    output_folder,
    image_folder=None,
    ncols=6,
    pad=20,
    figure_path=None,
):
    """
    Plot a grid of per-cell crops (original image with that cell's mask
    outlined), for visually inspecting what a UMAP sub-cluster actually
    looks like morphologically.

    Parameters
    ----------
    table : pandas.DataFrame
        As returned by load_cell_table().
    cell_ids : sequence of int
        Which cells to display, e.g. table.loc[selection, "cell_id"].
    output_folder : str or Path
        Folder passed to extract_cell_embeddings_from_folder()
        (mask_file paths in `table` are relative to it).
    image_folder : str or Path, optional
        Overrides the image folder; defaults to each cell's stored
        `image_path`, which is only valid on the machine that ran the
        extraction. Pass this when working on a different machine.
    """
    output_folder = Path(output_folder)
    cell_ids = list(cell_ids)
    nrows = int(np.ceil(len(cell_ids) / ncols))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.2 * ncols, 2.2 * nrows), dpi=150,
    )
    axes = np.atleast_1d(axes).ravel()

    mask_cache = {}
    image_cache = {}

    for ax, cell_id in zip(axes, cell_ids):
        row = table.loc[table["cell_id"] == cell_id]
        if row.empty:
            ax.set_visible(False)
            continue
        row = row.iloc[0]

        if image_folder is not None:
            image_path = Path(image_folder) / row["image_name"]
        else:
            image_path = Path(row["image_path"])

        mask_path = output_folder / row["mask_file"]

        if image_path not in image_cache:
            image_cache[image_path] = imread(image_path)
        if mask_path not in mask_cache:
            mask_cache[mask_path] = np.load(mask_path)

        image = image_cache[image_path]
        mask = mask_cache[mask_path]

        bbox = (
            row["bbox_ymin"], row["bbox_xmin"],
            row["bbox_ymax"], row["bbox_xmax"],
        )
        image_crop, cell_region = crop_cell(image, mask, bbox, cell_id=cell_id, pad=pad)

        ax.imshow(image_crop)
        ax.contour(cell_region, levels=[0.5], colors="#E45756", linewidths=1.2)
        ax.set_title(
            f"{row['phenotype_name']} (#{cell_id})", fontsize=7,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[len(cell_ids):]:
        ax.set_visible(False)

    fig.tight_layout()

    if figure_path is not None:
        fig.savefig(figure_path, bbox_inches="tight", dpi=200)

    return fig, axes


def plot_image_with_cells(
    image_name,
    table,
    output_folder,
    image_folder=None,
    color_by="label_id",
    label_colors=None,
    highlight_cell_ids=None,
    ax=None,
):
    """
    Plot one full source image with its predicted cell masks overlaid,
    colored by phenotype (or cluster), optionally with a subset of
    cells outlined to show where they came from (e.g. a UMAP
    sub-cluster selected with plot_embedding_scatter()).
    """
    image_rows = table.loc[table["image_name"] == image_name]
    if image_rows.empty:
        raise ValueError(f"No cells found for image: {image_name}")

    output_folder = Path(output_folder)
    mask_path = output_folder / image_rows.iloc[0]["mask_file"]
    mask = np.load(mask_path)

    if image_folder is not None:
        image_path = Path(image_folder) / image_name
    else:
        image_path = Path(image_rows.iloc[0]["image_path"])
    image = imread(image_path)

    if label_colors is None:
        if color_by == "label_id":
            label_colors = PHENOTYPE_COLORS
        else:
            unique_values = sorted(image_rows[color_by].unique())
            label_colors = dict(zip(unique_values, _qualitative_palette(len(unique_values))))

    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    for _, row in image_rows.iterrows():
        color_hex = label_colors.get(row[color_by], "#808080")
        rgb = tuple(int(color_hex.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4))
        cell_region = mask == row["cell_id"]
        overlay[cell_region] = (*rgb, 0.45)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    else:
        fig = ax.figure

    ax.imshow(image)
    ax.imshow(overlay)

    if highlight_cell_ids is not None:
        highlight_mask = np.isin(mask, list(highlight_cell_ids)) & (mask > 0)
        ax.contour(highlight_mask, levels=[0.5], colors="white", linewidths=1.5)

    legend_handles = [
        mpatches.Patch(
            color=color,
            label=PHENOTYPE_NAMES.get(value, str(value)) if color_by == "label_id" else str(value),
        )
        for value, color in label_colors.items()
        if value in image_rows[color_by].unique()
    ]
    ax.legend(handles=legend_handles, frameon=False, fontsize=7, loc="upper right")
    ax.set_title(image_name, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

    return fig, ax


def plot_umap_with_image_panel(
    coordinates,
    table,
    image_name,
    output_folder,
    image_folder=None,
    color_by="label_id",
    figure_path=None,
):
    """
    Two-panel figure linking a selected image to its position in the
    embedding distribution: left = original image with its cell masks
    overlaid, right = the full UMAP/t-SNE scatter with that image's
    cells highlighted.
    """
    fig, (ax_image, ax_scatter) = plt.subplots(
        1, 2, figsize=(10, 5), dpi=150,
    )

    image_indices = np.flatnonzero(table["image_name"].to_numpy() == image_name)

    plot_image_with_cells(
        image_name=image_name,
        table=table,
        output_folder=output_folder,
        image_folder=image_folder,
        color_by=color_by,
        ax=ax_image,
    )

    labels = table[color_by].to_numpy()
    label_names = PHENOTYPE_NAMES if color_by == "label_id" else None
    label_colors = PHENOTYPE_COLORS if color_by == "label_id" else None
    plot_embedding_scatter(
        coordinates=coordinates,
        labels=labels,
        label_names=label_names,
        label_colors=label_colors,
        class_ids=np.unique(labels),
        cutoff_number=500,
        highlight_indices=image_indices,
        ax=ax_scatter,
    )

    fig.tight_layout()

    if figure_path is not None:
        fig.savefig(figure_path, bbox_inches="tight", dpi=200)

    return fig, (ax_image, ax_scatter)