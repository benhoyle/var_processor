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
        phase_width - integer specifying angle resolution.
    """
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


def cart2polar(row, col):
    """Convert cartesian co-ordinates to polar."""
    # Compute angle in radians
    angle = np.arctan2(col, row)
    # Compute radius
    radius = np.sqrt(row**2+col**2)
    return radius, angle


# Original method
def generate_backLUT(max_radius):
    """Precalculate a lookup table for mapping from x,y to polar.

    Assumes a centre at 0, 0 - need to translate when back converting.

    Args:
        max_radius - integer representing the maximum radius -
            forms the LUT height and width.
    Returns:
        LUT - numpy array of shape (rows, cols, 2) that provides
            the backward mapping.

    """
    cart_range = np.arange(-1*max_radius, max_radius)
    row, col = np.meshgrid(
        cart_range,
        cart_range
    )

    radius, angle = cart2polar(row, col)

    radius = radius.astype(int)
    # Create a new LUT of shape (radius, angle, 2)
    backLUT = np.stack([radius, angle], axis=-1)
    return backLUT


def back_convert_image(polar_image, backLUT):
    """Convert a polar image to cartesian using the backLUT.

    Output image is square with dimensions 2*radius in each dimension.


    Args:
        polar_image - numpy array with radius along rows and
            angles along columns.
        backLUT - a numpy array generated as above.
    Returns:
        output_image - a numpy array representing the cartesian image.

    """
    # Determine image size
    range_radius, range_angles = polar_image.shape[:2]
    # Determine LUT size
    rows, cols, _ = backLUT.shape
    # Adjust radius so it is within the scale of the polar image
    radius = np.clip(backLUT[..., 0], 0, range_radius-1).astype(int)
    # Convert radian range to discrete pixel range
    angles = (
        (backLUT[..., 1]+np.pi)*(1/(2*np.pi))*(range_angles-1)
    ).astype(int)
    # If multiple components
    if polar_image.ndim == 3:
        output_image = polar_image[radius, angles, :]
    else:
        output_image = polar_image[radius, angles]
    # Image is upside down so flip the right way around
    return np.flip(output_image)


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


def forward_quad(input_data):
    """Split an image into 4.

    Args:
        input_data - 2D numpy array with polar image.
    """
    rows, cols = input_data.shape
    right_image = input_data[:, :cols//2].T
    left_image = np.flip(input_data[:, cols//2:].T)
    # Split in half again vertically to show
    rows, cols = right_image.shape
    output_images = list()
    output_images.append(right_image[:rows//2, :])
    output_images.append(left_image[:rows//2, :])
    output_images.append(right_image[rows//2:, :])
    output_images.append(left_image[rows//2:, :])
    return output_images


def backward_quad(image_list):
    """Take four images and recombine into one.

    Reverses forward_quad.

    Args:
        image_list - list of 2D numpy array with each
            quadrant image.
    """
    # Take quadrants [0] and [2] and concatenate along axis 0
    right_image = np.concatenate((image_list[0], image_list[2]), axis=0)
    # Take quadrants [1] and [3] and concatenate along axis 0
    left_image = np.concatenate((image_list[1], image_list[3]), axis=0)
    # Take above results, transpose, flip and concatenate along axis 1
    flipped_right = right_image.T
    flipped_left = np.flip(left_image.T)
    combined_image = np.concatenate((flipped_right, flipped_left), axis=1)
    return combined_image
