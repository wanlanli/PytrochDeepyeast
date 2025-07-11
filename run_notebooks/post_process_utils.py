from skimage import measure
def get_largest_object(mask):
    """
    Given a binary mask (2D numpy array), returns:
    - count: number of connected components (objects)
    - largest_mask: binary mask of the largest object
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