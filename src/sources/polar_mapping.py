"""Methods to map between a Cartesian and polar image representation."""

import math
import numpy as np
import matplotlib.pyplot as plt


def calculateLUT(radius):
    """Precalculate a lookup table with the image maths."""
    LUT = np.zeros((radius, 360, 2), dtype=np.int16)
    # Iterate around angles of field of view
    for angle in range(0, 360):
        # Iterate over diameter
        for r in range(0, radius):
            theta = math.radians(angle)
            # Take angles from the vertical
            col = math.floor(r*math.sin(theta))
            row = math.floor(r*math.cos(theta))
            # rows and cols will be +ve and -ve representing
            # at offset from an origin
            LUT[r, angle] = [row, col]
    return LUT


def convert_image(img, LUT):
    """
    Convert image from cartesian to polar co-ordinates.

    img is a numpy 2D array having shape (height, width)
    LUT is a numpy array having shape (diameter, 180, 2)
    storing [x, y] co-ords corresponding to [r, angle]
    """
    # Use centre of image as origin
    centre_row = img.shape[0] // 2
    centre_col = img.shape[1] // 2
    # Determine the largest radius
    if centre_row > centre_col:
        radius = centre_col
    else:
        radius = centre_row
    # Theta on Y-axis is closer to cortex maps
    output_image = np.zeros(shape=(360, radius))
    # Iterate around angles of field of view
    for angle in range(0, 360):
        # Iterate over radius
        for r in range(0, radius):
            # Get mapped x, y
            (row, col) = tuple(LUT[r, angle])
            # Translate origin to centre
            # This makes rotation clockwise from positive y axis
            m_row = centre_row - row
            m_col = col+centre_col
            output_image[angle, r] = img[m_row, m_col]
    return output_image


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
        LUT = calculateLUT(max_size)
    out_image = convert_image(input_image, LUT)
    # I could possibly change the mapping above to avoid flipping
    # Flipping arranges similar to cortex
    right_image = np.flip(out_image[:180, :])
    left_image = np.flipud(out_image[180:, :])
    return norm_scale(left_image), norm_scale(right_image)


def polar2cart(r, theta, center):
    """Convert polar co-ordinates to Cartesian."""
    x = r * np.cos(theta + np.pi/2) + center[0]
    y = r * np.sin(theta + np.pi/2) + center[1]
    return x, y


def generateLUT(center, final_radius, phase_width=256):
    """Generate a look-up table for polar mapping."""
    initial_radius = 0
    theta, R = np.meshgrid(
        np.linspace(0, 2*np.pi, phase_width),
        np.arange(initial_radius, final_radius)
    )

    Xcart, Ycart = polar2cart(R, theta, center)

    Xcart = Xcart.astype(int)
    Ycart = Ycart.astype(int)
    return (Xcart, Ycart)


def img2polar(img, center, final_radius, LUT, phase_width=256):
    """Map an image to polar co-ordinates."""
    Xcart, Ycart = LUT
    if img.ndim == 3:
        polar_img = img[Ycart, Xcart, :]
        polar_img = np.reshape(
            polar_img, (final_radius, phase_width, 3)
        )
    else:
        polar_img = img[Ycart, Xcart]
        polar_img = np.reshape(
            polar_img, (final_radius, phase_width)
        )

    return polar_img
