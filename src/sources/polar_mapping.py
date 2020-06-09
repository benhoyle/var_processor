"""Methods to map between a Cartesian and polar image representation."""

import math
import numpy as np
import matplotlib.pyplot as plt


def calculatebackLUT(max_radius):
    """Precalculate a lookup table for mapping from x,y to polar."""
    LUT = np.zeros((max_radius*2, max_radius*2, 2), dtype=np.int16)
    # Iterate around x and y
    for row in range(0, max_radius*2):
        for col in range(0, max_radius*2):
            # Translate to centre - minus row makes angle from +ve y axis
            m_row = max_radius - row
            m_col = col - max_radius
            # Calculate angle w.r.t. y axis
            angle = math.atan2(m_col, m_row)
            # Convert to degrees
            degrees = math.degrees(angle)
            # Calculate radius
            radius = math.sqrt(m_row*m_row+m_col*m_col)
            # print(angle, radius)
            LUT[row, col] = [int(radius), int(degrees)]
    return LUT


def build_mask(img, backLUT, ticks=20):
    """Build a mask showing polar co-ord system."""
    overlay = np.zeros(shape=img.shape, dtype=np.bool)
    # We need to set origin backLUT has origin at radius, radius
    row_adjust = backLUT.shape[0]//2 - img.shape[0]//2
    col_adjust = backLUT.shape[1]//2 - img.shape[1]//2
    for row in range(0, img.shape[0]):
        for col in range(0, img.shape[1]):
            m_row = row + row_adjust
            m_col = col + col_adjust
            (r, theta) = backLUT[m_row, m_col]
            if (r % ticks) == 0 or (theta % ticks) == 0:
                overlay[row, col] = 1
    masked = np.ma.masked_where(overlay == 0, overlay)
    return masked


# see https://stackoverflow.com/questions/31877353/
# overlay-an-image-segmentation-with-numpy-and-matplotlib
def show_field(img, backLUT, ticks=20, transparency=0.5):
    """Show an indication of the field of view on the image."""
    masked = build_mask(img, backLUT, ticks)
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.imshow(masked, cmap='hsv', alpha=transparency)
    plt.show()


def norm_scale(a):
    """Normalise array to a scale of 0 to 1."""
    return (a - np.min(a))/np.ptp(a)


def standardise(a):
    """Subtract mean and divide by standard deviation."""
    return (a - np.mean(a)) / np.std(a)


def split_field(input_image, LUT=None):
    """Convert image to polar and split across the visual field."""
    # If a look-up table is not passed, create one
    if LUT is None:
        max_size = max(input_image.shape)
        LUT = generateLUT(max_size)
    out_image = convert_image(input_image, LUT)
    # I could possibly change the mapping above to avoid flipping
    # Flipping arranges similar to cortex
    right_image = np.flip(out_image[:180, :])
    left_image = np.flipud(out_image[180:, :])
    return norm_scale(left_image), norm_scale(right_image)


def polar2cart(r, theta):
    """Convert polar co-ordinates to Cartesian."""
    # x = r * np.cos(theta + np.pi/2) + center[0]
    # y = r * np.sin(theta + np.pi/2) + center[1]
    row = r * np.cos(theta)
    col = r * np.sin(theta)
    return row, col


def generateLUT(radius, phase_width=256):
    """Generate a look-up table for polar mapping.

    Args:
        radius - integer specifying radius in pixels.
        phase_width - integer specifying angle resolution."""
    theta, R = np.meshgrid(
        np.linspace(0, 2*np.pi, phase_width),
        np.arange(0, radius)
    )

    rows, cols = polar2cart(R, theta)

    rows = rows.astype(int)
    cols = cols.astype(int)
    # Create a new LUT of shape (radius, angle, 2)
    LUT = np.stack([rows, cols], axis=-1)
    return LUT


def convert_image(image, LUT):
    """Precalculate a lookup table with the image maths."""
    # Determine image size
    rows, cols = image.shape[:2]
    # Use centre of image as origin
    centre_row = rows // 2
    centre_col = cols // 2
    # Determine LUT size
    max_radius, angles, _ = LUT.shape
    # Determine radius
    radius = min(max_radius, centre_row, centre_col)
    # Adjust LUT to centre and clip so indices are within bounds
    rows = np.clip(centre_row - LUT[..., 0], 0, centre_row+radius-1)
    cols = np.clip(centre_col + LUT[..., 1], 0, centre_col+radius-1)
    # If multiple components
    if image.ndim == 3:
        output = image[rows, cols, :]
    else:
        output = image[rows, cols]
    # Crop based on min radius
    return output[:radius, ...]


def setup_reduced_res(image_width, first_group=10):
    """Generate data for reducing resolution.

    Args:
        image_width - width (in pixels)of polar-mapped image.
        first_group - width (in pixels) of most detailed area.

    Returns:
        groupings - np array of sizes of grouped pixels.
        grouping_ranges - list of np arrays representing ranges for the
            grouping.
        spacings - np array representing the width of each area.

    """
    # Get width of image as a power of 2
    base_power = int(np.log2(image_width))
    # Highest resolution is set by rough science
    # But we need to change this based on
    start_group = (base_power-6)
    # Determine the number of pixels to group
    # across the angular (rotation) dimension
    groupings = 2**(np.arange(start_group, base_power))
    # Determine the ends of the ranges for the different groups
    spacings = 2**(np.arange(0, groupings.shape[0]))*5
    # Determine the ranges outside of the loop
    grouping_ranges = [
        np.arange(0, image_width, g) for g in groupings
    ]
    return groupings, grouping_ranges, spacings


def reduce_resolution(image, output_display=False, precomputed=None):
    """Reduce resolution as per visual acuity.

    Args:
        image - 2D numpy array in polar domain - width (cols)
            - needs to be a power of 2.
        output_display - boolean indicating whether to
            calculate an output image for display.
        precomputed - triple of (groupings, grouping_ranges, spacings)
            - optionally precomputed to speed up

    Returns:
        tuple of:
            output_list - list of reduced image portions.
            output_image - image for display if output_display = True.

    """
    if precomputed is None:
        groupings, grouping_ranges, spacings = setup_reduced_res(
            image.shape[1]
        )
    else:
        groupings, grouping_ranges, spacings = precomputed
    # Build a list of different resolutions
    start = 0
    output_list = list()
    if output_display:
        # This needs to be changed in case sum > image.shape
        width = min(spacings.sum(), image.shape[0])
        shape = (width, image.shape[1])
        output_image = np.zeros(shape=shape, dtype=image.dtype)
    else:
        output_image = None
    # Loop over the groupings
    for i in range(groupings.shape[0]):
        # Average over each set of groupings and add to list
        reduced = np.add.reduceat(
            image[start:start + spacings[i], :],
            grouping_ranges[i],
            axis=1
        ) // groupings[i]
        output_list.append(reduced)
        # If output_display flag is set, generate an image for output
        if output_display:
            end = start + spacings[i]
            if end > output_image.shape[0]:
                end = output_image.shape[0]
            output_image[
                start:end, :
            ] = np.repeat(reduced, groupings[i], axis=1)
        # Set the start of the next range as the end of the previous range
        start += spacings[i]
    # Return outputs
    return output_list, output_image
