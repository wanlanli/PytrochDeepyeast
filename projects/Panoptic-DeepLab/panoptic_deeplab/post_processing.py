# Copyright (c) Facebook, Inc. and its affiliates.
# Reference: https://github.com/bowenc0221/panoptic-deeplab/blob/master/segmentation/model/post_processing/instance_post_processing.py  # noqa

from collections import Counter
import torch
import torch.nn.functional as F
from skimage.morphology import disk, remove_small_objects, opening, closing, erosion, remove_small_holes

from datetime import datetime
import kornia.morphology as km


def find_instance_center(center_heatmap, threshold=0.1, nms_kernel=3, top_k=None):
    """
    Find the center points from the center heatmap.
    Args:
        center_heatmap: A Tensor of shape [1, H, W] of raw center heatmap output.
        threshold: A float, threshold applied to center heatmap score.
        nms_kernel: An integer, NMS max pooling kernel size.
        top_k: An integer, top k centers to keep.
    Returns:
        A Tensor of shape [K, 2] where K is the number of center points. The
            order of second dim is (y, x).
    """
    # Thresholding, setting values below threshold to -1.
    center_heatmap = F.threshold(center_heatmap, threshold, -1)

    # NMS
    nms_padding = (nms_kernel - 1) // 2
    center_heatmap_max_pooled = F.max_pool2d(
        center_heatmap, kernel_size=nms_kernel, stride=1, padding=nms_padding
    )
    center_heatmap[center_heatmap != center_heatmap_max_pooled] = -1

    # Squeeze first two dimensions.
    center_heatmap = center_heatmap.squeeze()
    assert len(center_heatmap.size()) == 2, "Something is wrong with center heatmap dimension."

    # Find non-zero elements.
    if top_k is None:
        return torch.nonzero(center_heatmap > 0)
    else:
        # find top k centers.
        top_k_scores, _ = torch.topk(torch.flatten(center_heatmap), top_k)
        return torch.nonzero(center_heatmap > top_k_scores[-1].clamp_(min=0))


def group_pixels(center_points, offsets):
    """
    Gives each pixel in the image an instance id.
    Args:
        center_points: A Tensor of shape [K, 2] where K is the number of center points.
            The order of second dim is (y, x).
        offsets: A Tensor of shape [2, H, W] of raw offset output. The order of
            second dim is (offset_y, offset_x).
    Returns:
        A Tensor of shape [1, H, W] with values in range [1, K], which represents
            the center this pixel belongs to.
    """
    height, width = offsets.size()[1:]

    # Generates a coordinate map, where each location is the coordinate of
    # that location.
    y_coord, x_coord = torch.meshgrid(
        torch.arange(height, dtype=offsets.dtype, device=offsets.device),
        torch.arange(width, dtype=offsets.dtype, device=offsets.device),
    )
    coord = torch.cat((y_coord.unsqueeze(0), x_coord.unsqueeze(0)), dim=0)

    center_loc = coord + offsets
    center_loc = center_loc.flatten(1).T.unsqueeze_(0)  # [1, H*W, 2]
    center_points = center_points.unsqueeze(1)  # [K, 1, 2]

    # Distance: [K, H*W].
    distance = torch.norm(center_points - center_loc, dim=-1)

    # Finds center with minimum distance at each location, offset by 1, to
    # reserve id=0 for stuff.
    instance_id = torch.argmin(distance, dim=0).reshape((1, height, width)) + 1
    return instance_id


def get_instance_segmentation(
    sem_seg, center_heatmap, offsets, thing_seg, thing_ids, threshold=0.1, nms_kernel=3, top_k=None
):
    """
    Post-processing for instance segmentation, gets class agnostic instance id.
    Args:
        sem_seg: A Tensor of shape [1, H, W], predicted semantic label.
        center_heatmap: A Tensor of shape [1, H, W] of raw center heatmap output.
        offsets: A Tensor of shape [2, H, W] of raw offset output. The order of
            second dim is (offset_y, offset_x).
        thing_seg: A Tensor of shape [1, H, W], predicted foreground mask,
            if not provided, inference from semantic prediction.
        thing_ids: A set of ids from contiguous category ids belonging
            to thing categories.
        threshold: A float, threshold applied to center heatmap score.
        nms_kernel: An integer, NMS max pooling kernel size.
        top_k: An integer, top k centers to keep.
    Returns:
        A Tensor of shape [1, H, W] with value 0 represent stuff (not instance)
            and other positive values represent different instances.
        A Tensor of shape [1, K, 2] where K is the number of center points.
            The order of second dim is (y, x).
    """
    center_points = find_instance_center(
        center_heatmap, threshold=threshold, nms_kernel=nms_kernel, top_k=top_k
    )
    if center_points.size(0) == 0:
        return torch.zeros_like(sem_seg), center_points.unsqueeze(0)
    ins_seg = group_pixels(center_points, offsets)
    return thing_seg * ins_seg, center_points.unsqueeze(0)


def merge_semantic_and_instance(
    sem_seg, ins_seg, semantic_thing_seg, label_divisor, thing_ids, stuff_area, void_label,
    area_threshold
):
    """
    Post-processing for panoptic segmentation, by merging semantic segmentation
        label and class agnostic instance segmentation label.
    Args:
        sem_seg: A Tensor of shape [1, H, W], predicted category id for each pixel.
        ins_seg: A Tensor of shape [1, H, W], predicted instance id for each pixel.
        semantic_thing_seg: A Tensor of shape [1, H, W], predicted foreground mask.
        label_divisor: An integer, used to convert panoptic id =
            semantic id * label_divisor + instance_id.
        thing_ids: Set, a set of ids from contiguous category ids belonging
            to thing categories.
        stuff_area: An integer, remove stuff whose area is less tan stuff_area.
        void_label: An integer, indicates the region has no confident prediction.
    Returns:
        A Tensor of shape [1, H, W].
    """
    # In case thing mask does not align with semantic prediction.
    pan_seg = torch.zeros_like(sem_seg) + void_label
    is_thing = (ins_seg > 0) & (semantic_thing_seg > 0)

    # Keep track of instance id for each class.
    class_id_tracker = Counter()

    # Paste thing by majority voting.
    instance_ids = torch.unique(ins_seg)
    for ins_id in instance_ids:
        if ins_id == 0:
            continue
        # Make sure only do majority voting within `semantic_thing_seg`.
        thing_mask = (ins_seg == ins_id) & is_thing
        if torch.nonzero(thing_mask).size(0) < area_threshold:
            continue
        
        # thing_mask[0] = torch.from_numpy(self_defined_post_process(thing_mask[0].cpu().numpy()))
        thing_mask[0] = torch_self_defined_post_process(thing_mask[0])

        class_id, _ = torch.mode(sem_seg[thing_mask].view(-1))
        class_id_tracker[class_id.item()] += 1
        new_ins_id = class_id_tracker[class_id.item()]
        pan_seg[thing_mask] = class_id * label_divisor + new_ins_id

    # Paste stuff to unoccupied area.
    class_ids = torch.unique(sem_seg)
    for class_id in class_ids:
        if class_id.item() in thing_ids:
            # thing class
            continue
        # Calculate stuff area.
        stuff_mask = (sem_seg == class_id) & (ins_seg == 0)
        if stuff_mask.sum().item() >= stuff_area:
            pan_seg[stuff_mask] = class_id * label_divisor

    return pan_seg


def get_panoptic_segmentation(
    sem_seg,
    center_heatmap,
    offsets,
    thing_ids,
    label_divisor,
    stuff_area,
    void_label,
    threshold=0.1,
    nms_kernel=7,
    top_k=200,
    area_threshold=200,
    foreground_mask=None,
):
    """
    Post-processing for panoptic segmentation.
    Args:
        sem_seg: A Tensor of shape [1, H, W] of predicted semantic label.
        center_heatmap: A Tensor of shape [1, H, W] of raw center heatmap output.
        offsets: A Tensor of shape [2, H, W] of raw offset output. The order of
            second dim is (offset_y, offset_x).
        thing_ids: A set of ids from contiguous category ids belonging
            to thing categories.
        label_divisor: An integer, used to convert panoptic id =
            semantic id * label_divisor + instance_id.
        stuff_area: An integer, remove stuff whose area is less tan stuff_area.
        void_label: An integer, indicates the region has no confident prediction.
        threshold: A float, threshold applied to center heatmap score.
        nms_kernel: An integer, NMS max pooling kernel size.
        top_k: An integer, top k centers to keep.
        foreground_mask: Optional, A Tensor of shape [1, H, W] of predicted
            binary foreground mask. If not provided, it will be generated from
            sem_seg.
    Returns:
        A Tensor of shape [1, H, W], int64.
    """
    if sem_seg.dim() != 3 and sem_seg.size(0) != 1:
        raise ValueError("Semantic prediction with un-supported shape: {}.".format(sem_seg.size()))
    if center_heatmap.dim() != 3:
        raise ValueError(
            "Center prediction with un-supported dimension: {}.".format(center_heatmap.dim())
        )
    if offsets.dim() != 3:
        raise ValueError("Offset prediction with un-supported dimension: {}.".format(offsets.dim()))
    if foreground_mask is not None:
        if foreground_mask.dim() != 3 and foreground_mask.size(0) != 1:
            raise ValueError(
                "Foreground prediction with un-supported shape: {}.".format(sem_seg.size())
            )
        thing_seg = foreground_mask
    else:
        # inference from semantic segmentation
        thing_seg = torch.zeros_like(sem_seg)
        for thing_class in list(thing_ids):
            thing_seg[sem_seg == thing_class] = 1
    instance, center = get_instance_segmentation(
        sem_seg,
        center_heatmap,
        offsets,
        thing_seg,
        thing_ids,
        threshold=threshold,
        nms_kernel=nms_kernel,
        top_k=top_k,
    )
    panoptic = merge_semantic_and_instance(
        sem_seg, instance, thing_seg, label_divisor, thing_ids, stuff_area, void_label, area_threshold=area_threshold
    )
    return panoptic, center


# def self_defined_post_process(image, selem=disk(9), area_threshold=1000, erosion_factor=2):
#     """
#     Apply morphological post-processing to clean up a binary image.

#     Steps:
#     1. Opening to remove small white noise.
#     2. Closing to fill small black holes.
#     3. Remove small objects below a specified area threshold.
#     4. Erode the regions to shrink them slightly and reduce noise.

#     Parameters:
#     ----------
#     image : 2D binary numpy array
#         Input binary mask to be cleaned.
#     selem : ndarray, optional
#         Structuring element used for morphological operations (default: disk(3)).
#     area_threshold : int, optional
#         Minimum area of objects to keep (default: 100 pixels).
#     erosion_factor : int, optional
#         Radius of erosion to apply after cleaning (default: 2).

#     Returns:
#     -------
#     shrunk : 2D binary numpy array
#         Cleaned and eroded binary mask.
#     """

#     # Morphological opening: remove small white noise
#     opened = opening(image, selem)

#     # Morphological closing: fill small black holes
#     closed = closing(opened, selem)

#     # Remove small connected components
#     img_rm_small = remove_small_objects(closed, min_size=area_threshold, connectivity=2)
#     img_fill_holes = remove_small_holes(img_rm_small, area_threshold*2, connectivity=2)


#     # Erode the regions slightly to refine the boundaries
#     shrunk = erosion(img_fill_holes, disk(erosion_factor))

#     return shrunk




def torch_self_defined_post_process(mask: torch.Tensor,
                                    kernel_size: int = 7,):
    """
    Fast GPU-based morphological post-processing.

    Args:
        mask (torch.Tensor): Binary mask (H, W), torch.bool or torch.uint8
        area_threshold (int): Minimum size for connected components
        erosion_factor (int): Radius for erosion

    Returns:
        torch.Tensor: Refined binary mask (H, W), dtype=torch.bool
    """
    device = mask.device
    mask = mask.unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)
    kernel = torch.ones(kernel_size, kernel_size).to(device)
    # Opening (remove small white noise)
    opened = km.opening(mask, kernel)

    # Closing (fill small holes)
    closed = km.closing(opened, kernel)

    return closed.squeeze().bool()