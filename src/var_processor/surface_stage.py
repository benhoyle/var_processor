import numpy as np


def upsample(surface_list):
    """Upsample a 2D array."""
    # See https://stackoverflow.com/questions/32846846/quick-way-to-upsample-numpy-array-by-nearest-neighbor-tiling
    upsampled = [s.repeat(2, axis=0).repeat(2, axis=1) for s in surface_list]
    return upsampled


def downsample(array):
    """Downsample a 2D array."""
    return array[::2, ::2]


class AbstractStage:

    def __init__(self):
        self.surfaces = []

    def forward_pre_processing(self, forward_data):
        """Optional pre_processing."""
        return forward_data

    def forward_post_processing(self, processed_data):
        """Optional post_processing."""
        return processed_data

    def forward_processing(self, data):
        """Forward processing."""
        return data

    def forward(self, forward_data):
        # Perform optional pre-processing
        pre_processed_data = self.forward_pre_processing(forward_data)
        # Perform forward processing
        processed_data = self.forward_processing(pre_processed_data)
        # Perform optional post-processing
        processed_output = self.forward_post_processing(processed_data)
        self.surfaces = processed_output
        return self.get_surfaces()

    def backward_pre_processing(self, backward_data):
        """Optional pre_processing."""
        return backward_data

    def backward_post_processing(self, processed_data):
        """Optional post_processing."""
        return processed_data

    def backward_processing(self, data):
        """Backward processing."""
        return data

    def backward(self, backward_data):
        """Same as reconstruct.

        backward data = lower level surfaces.
        """
        # Perform optional pre-processing
        pre_processed_data = self.backward_pre_processing(backward_data)
        # Inverse Transform
        processed_data = self.backward_processing(pre_processed_data)
        # Perform optional post-processing
        processed_output = self.backward_post_processing(processed_data)
        return processed_output

    def residuals(self, lower_data):
        """Determine difference from lower level.

        lower_data = list of lower level surfaces.
        """
        upsampled_surfaces = upsample(lower_data)
        # This has a stage storing its current surfaces
        residual_surfaces = [s - u for s, u in zip(self.surfaces, upsampled_surfaces)]
        return residual_surfaces

    def get_surfaces(self):
        return self.surfaces

    def lower_input(self):
        """Get the input for a lower stage."""
        average = self.surfaces[0]
        downsampled = downsample(average)
        return downsampled


def decompose(image: np.ndarray, scale=True) -> list:
    """Decompose an image using the Hadamard transform."""
    # If unsigned 8-bit convert to signed 16-bit to allow computations
    if image.dtype == np.uint8:
        image = image.astype(np.int16)
    # Horizontal difference - TL - BL + TR - BR
    H = image[:-1, :-1] - image[1:, :-1] + image[:-1, 1:] - image[1:, 1:]
    # Vertical difference - TL - TR + BL - BR
    V = image[:-1, :-1] - image[:-1, 1:] + image[1:, :-1] - image[1:, 1:]
    # Diagonal difference
    # Work out top (TL + BR) - (TR + BL)
    D = (image[:-1, :-1] + image[1:, 1:]) - (image[1:, :-1] + image[:-1, 1:])
    # Average is sum of four shifted versions - TL + BL + TR + BR
    A = image[:-1, :-1] + image[1:, :-1] + image[:1, -1] + image[1:, 1:]
    # Subsample and scale based on variable
    if scale:
        surfaces = [A[::2, ::2] // 4, H[::2, ::2] // 4, V[::2, ::2] // 4, D[::2, ::2]// 4]
    else:
        surfaces = [A[::2, ::2], H[::2, ::2], V[::2, ::2], D[::2, ::2]]
    return surfaces


def recompose(surfaces: list) -> np.ndarray:
    """Recompose a list of surfaces into an image using the Hadamard transform."""
    # Data = surfaces set
    [A, H, V, D] = surfaces
    # Create new blank image at x2 size
    blank_shape = (A.shape[0] * 2, A.shape[1] * 2)
    blank = np.zeros(blank_shape, dtype=np.int16)
    # Then recombine surfaces into a single image
    blank[::2, ::2] = A + H + V + D
    blank[::2, 1::2] = A + H - V - D
    blank[1::2, ::2] = A - H + V - D
    blank[1::2, 1::2] = A - H - V + D
    # Then convert back to 8-bit
    return blank.astype(np.uint8)


class DecomposeStage(AbstractStage):
    """Stage for decomposing a signal."""
    def forward_processing(self, data):
        """Forward processing."""
        # Remember to convert to 16 bit signed
        surfaces = decompose(data.astype(np.int16))
        return surfaces

    def backward_processing(self, data):
        """Backward processing."""

        surfaces = decompose(data.astype(np.int16))
        return surfaces
