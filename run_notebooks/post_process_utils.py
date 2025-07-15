from skimage import measure
import numpy as np


def get_largest_object(mask):
    """
    Extract the largest connected component from a binary mask.

    Parameters:
    -----------
    mask : np.ndarray
        A 2D binary mask (non-zero values are considered foreground).

    Returns:
    --------
    count : int
        The total number of connected components in the mask.

    largest_mask : np.ndarray
        A binary mask (same shape as input) where only the largest object is marked True (or 1).
        If no objects are found, returns a mask of all False.
    """
    labeled = measure.label(mask)  # Assigns a unique label to each connected region
    props = measure.regionprops(labeled)

    if not props:
        return 0, np.zeros_like(mask, dtype=bool)

    # Find region with largest area
    largest = max(props, key=lambda x: x.area)

    # Create a new mask for the largest region
    largest_mask = labeled == largest.label
    return len(props), largest_mask


def convert_to_rgb(image):
    """
    Convert a 2D grayscale image to 3-channel RGB format for visualization.

    Parameters:
    -----------
    image : np.ndarray
        A 2D grayscale image (single channel).

    Returns:
    --------
    image : np.ndarray
        A 3D RGB image (H x W x 3) scaled to 8-bit unsigned integers.
        If the input is already 3D (e.g., RGB), it is returned unchanged.
    """
    if image.ndim == 2:
        image = image - image.min()
        image = image / image.max()
        image = (image * 255).astype(np.uint8)
        image = np.stack((image,)*3, axis=-1)
    return image


def __merge_mask(whole_mask, sub_new_mask, c1, c2, c3, c4, new_label):
    """
    Merge a segmented sub-mask into the corresponding region of the full image mask.

    This function inserts `sub_new_mask` into the region `c1:c2, c3:c4` of `whole_mask`,
    replacing any overlapping labels with 0 before inserting the new label.

    Parameters:
    -----------
    whole_mask : np.ndarray
        The full-size mask to update (2D, labeled).

    sub_new_mask : np.ndarray
        The predicted binary mask for the subregion, where foreground is > 0.

    c1, c2, c3, c4 : int
        Coordinates defining the crop region in the full image (rows: c1 to c2, cols: c3 to c4).

    new_label : int
        The new label to assign to the subregion object in `whole_mask`.

    Returns:
    --------
    whole_mask : np.ndarray
        The updated full-size mask with the new region inserted and overlapping labels removed.
    """
    over_lapped_labels = np.unique(whole_mask[c1:c2,c3:c4][sub_new_mask>0])[1:]

    whole_mask[c1:c2, c3:c4][np.isin(whole_mask[c1:c2, c3:c4], over_lapped_labels)] = 0
    whole_mask[c1:c2, c3:c4][sub_new_mask>0] = new_label
    return whole_mask


def segment_slide_window(predictor, orginal_image, crops,
                         score_threshold=0.1,
                         ins_threshold=0.6,
                         area_threshold = 1000,
                         ):
    """
    Perform instance segmentation over an image using a sliding window approach.

    Parameters:
    -----------
    predictor : callable
        A segmentation model function that takes an RGB image and returns:
        - 'instances' with attributes: pred_masks, scores, center_scores, panoptic_label.

    original_image : np.ndarray
        The full-size input image (2D), used to define the shape of the output mask.

    crops : list of tuples
        List of crop coordinates (c1, c2, c3, c4) defining subregions in the image.

    score_threshold : float
        Minimum detection confidence score to keep an instance.

    ins_threshold : float
        Minimum instance-level confidence (e.g., center heatmap score) to keep an instance.

    area_threshold : int
        Minimum area (in pixels) for a predicted instance to be accepted.

    Returns:
    --------
    mask_merged : np.ndarray
        A 2D labeled mask (dtype=np.uint16), same shape as `original_image`,
        where each segmented instance is assigned a unique label.
    """
    mask_merged = np.zeros(orginal_image.shape, dtype=np.uint16)
    object_id = 1

    for idx, (c1, c2, c3, c4) in enumerate(crops):
        image = orginal_image[c1:c2, c3:c4]
        image = convert_to_rgb(image)
        prediction_output = predictor(image)

        if "instances" not in prediction_output.keys():
            continue
        instances = prediction_output['instances']
        panopitc_labels = instances.panoptic_label.to("cpu").numpy()
        pred_masks = instances.pred_masks.to("cpu").numpy()
        scores = instances.scores
        instance_scores = instances.center_scores

        for pred_mask, score, ins_score, label in zip(pred_masks, scores, instance_scores, panopitc_labels):
            if score < score_threshold:
                continue
            if ins_score < ins_threshold:
                continue
            if pred_mask[0,:].any() or pred_mask[-1, :].any():
                continue
            if pred_mask[:, 0].any() or pred_mask[:, -1].any():
                continue
            area = pred_mask.sum()
            if area < area_threshold:
                continue
            num, mask = get_largest_object(np.array(pred_mask))
            if num == 0:
                continue

            new_label = (label//1000)*1000 + object_id
            object_id += 1

            # over_lapped_labels = np.unique(mask_merged[c1:c2,c3:c4][mask>0])[1:]

            # mask_merged[c1:c2, c3:c4][np.isin(mask_merged[c1:c2, c3:c4], over_lapped_labels)] = 0
            # mask_merged[c1:c2, c3:c4][mask>0] = new_label
            mask_merged = __merge_mask(mask_merged, mask, c1, c2, c3, c4, new_label)
    return mask_merged



def segment_post_process(prediction_output,
                              seg_threshold=0.1,
                              ins_threshold=0.6,
                              area_threshold=1000):
    instances = prediction_output['instances']
    panopitc_labels = instances.panoptic_label.to("cpu").numpy()
    pred_masks = instances.pred_masks.to("cpu").numpy()
    scores = instances.scores
    instance_scores = instances.center_scores

    post_precessed_mask = np.zeros(pred_masks[0].shape, dtype=np.uint16)
    for pred_mask, score, ins_score, label in zip(pred_masks, scores, instance_scores, panopitc_labels):
        if score < seg_threshold:
            continue
        if ins_score < ins_threshold:
            continue
        if pred_mask[0,:].any() or pred_mask[-1, :].any():
            continue
        if pred_mask[:, 0].any() or pred_mask[:, -1].any():
            continue
        area = pred_mask.sum()
        if area < area_threshold:
            continue
        num, mask = get_largest_object(np.array(pred_mask))
        if num == 0:
            continue
        post_precessed_mask[mask] = label
    return post_precessed_mask