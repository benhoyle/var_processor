"""Video Helper Functions."""

# Imports
import numpy as np
import matplotlib.pyplot as plt
import cv2


def reduce_to_2(image, reduce="centre"):
    """Reduce image size to a square that is a factor of 2."""
    rows, cols = image.shape
    scales = int(np.log2(rows))
    size = 2**scales
    row_diff = rows - size
    col_diff = cols - size
    # Watch out for when array is well-formed
    upper_row = rows
    lower_row = 0
    upper_col = cols
    lower_col = 0
    if row_diff:
        upper_row = -row_diff//2
        lower_row = row_diff//2
    if col_diff:
        # This is needed when processing left/right visual fields
        if reduce == "centre":
            # Crop either side of the centre
            upper_col = -col_diff//2
            lower_col = col_diff//2
        elif reduce == "left":
            # Crop from the left
            lower_col = col_diff
        elif reduce == "right":
            # Crop from the right
            upper_col = -col_diff
    return image[lower_row:upper_row, lower_col:upper_col]


def separate_components(frame, square=True):
    """Separate frame into YUV components.

    square - boolean - if true subsample the colour images so they are square
    """
    if square:
        return frame[:, :, 0], frame[::2, 1::2, 1], frame[::2, 0::2, 1]
    else:
        return frame[:, :, 0], frame[:, 1::2, 1], frame[:, 0::2, 1]


def create_pyramid(image, reduce="centre"):
    """Create a gaussian image pyramid from the image."""
    # Crop image at "left", "right" or "centre" based if reduced
    if reduce:
        image = reduce_to_2(image, reduce)
    scales = image.shape[0].bit_length()-1
    # Create list to hold scales
    pyramid = [image]
    current_image = image
    for _ in range(scales):
        # Reduce size by 2
        new_rows = current_image.shape[0]//2
        # Check for 1D array
        cols = current_image.shape[1]
        new_cols = cols//2 if cols != 1 else 1
        lower_level = cv2.pyrDown(
            current_image,
            (new_rows, new_cols)
        )
        # Save downsampled array in pyramid
        pyramid.append(lower_level)
        current_image = lower_level
    return pyramid


def upsample(pyramid):
    """Upsample pyramid."""
    # Add the 1x1 pixel reconstruction to the rebuilt list
    upsampled = [pyramid[-1]]
    # Check for 1D pyramid
    one_d = True if pyramid[0].shape[1] == 1 else False
    # Interate through the images and add reconstructed versions
    for image in reversed(pyramid[1:]):
        rows, cols = image.shape
        new_rows = rows*2
        # Set cols to 1 if one_d else double size
        new_cols = 1 if one_d else cols*2
        upsampled.append(cv2.pyrUp(image, dstsize=(new_cols, new_rows)))
    return upsampled


def rebuild_from_diff(positives, negatives, base=None, log2=False):
    """Rebuild from a list of differences."""
    # If not passed base recreate through differences alone
    if not base:
        # We need to add 128 so we can view as an image
        base = np.zeros(shape=(1, 1)) + 128
    for pos, neg in zip(reversed(positives), reversed(negatives)):
        if log2:
            pos[pos > 0] = np.power(2, pos[pos > 0] - 1)
            neg[neg > 0] = np.power(2, neg[neg > 0] - 1)
        base += pos
        base -= neg
        base = cv2.pyrUp(base)
    return base


def get_differences(channel, clip=False):
    """Get stack of differences for a channel."""
    # Create downsampled pyramid
    pyramid = create_pyramid(channel)
    # Create upsampled pyramid
    upsampled = upsample(pyramid)
    # Calculate differences
    if clip:
        # Clip the differences at -1 and 1
        diffs = [
            np.clip((d-u).astype(np.int8), -1, 1)
            for d, u in zip(pyramid, reversed(upsampled))
        ]
    else:
        # Don't clip the differences
        diffs = [
            (d-u).astype(np.int8)
            for d, u in zip(pyramid, reversed(upsampled))
        ]
    # Swap last diff for base of pyramid
    diffs = diffs[:-1] + [pyramid[-1]]
    return diffs


def get_differences_2D(channel, clip=False):
    """Get stack of differences for a channel and return as 2D array."""
    # Create downsampled pyramid
    pyramid = create_pyramid(channel, reduce=None)
    pyr_len = len(pyramid)
    # Create upsampled pyramid
    upsampled = upsample(pyramid)
    diffs = np.zeros(shape=(pyramid[0].shape[0], pyr_len), dtype=np.int8)
    # Calculate differences
    i = 0
    for d, u in zip(pyramid, reversed(upsampled)):
        diff = (d-u).astype(np.int8)
        if clip:
            # Clip the differences at -1 and 1
            diff = np.clip(diff, -1, 1)
        diffs[:, i] = np.repeat(diff, 2**i)
        i += 1
    # Swap last diff for base of pyramid
    diffs[:, -1] = np.repeat(pyramid[-1], 2**(pyr_len-1))
    return diffs


def rebuild_channel(channel, log2=False, clip=True):
    """Rebuild for a channel."""
    positives, negatives = get_differences(channel, log2=log2, clip=clip)
    return rebuild_from_diff(positives, negatives, log2=log2)


def get_multi_channel_differences(frame):
    """Get differences for multiple channels."""
    channels = separate_components(frame)
    differences = list()
    for c in channels:
        c_diff = list()
        pos, neg = get_differences(c)
        for p, n in zip(pos, neg):
            c_diff.append((p-n).astype(np.int8))
        differences.append(c_diff)
    return differences


def plot_pyramid(pyramid):
    """Plot an image pyramid as a stack of subplots."""
    figure, axes = plt.subplots(len(pyramid), 1)
    for i, im in enumerate(reversed(pyramid)):
        axes[i].imshow(im)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_aspect('equal')
    plt.show()
