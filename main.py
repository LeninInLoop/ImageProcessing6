import os
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

class BColors:
    HEADER = '\033[95m'
    OkBLUE = '\033[94m'
    OkCYAN = '\033[96m'
    OkGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class ImageUtils:
    @staticmethod
    def load_image(filepath: str) -> np.ndarray:
        """Load an image from file as a numpy array."""
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        return np.array(Image.open(filepath))

    @staticmethod
    def save_image(image_array: np.ndarray, filepath: str) -> None:
        """Save a numpy array as an image file."""
        if image_array.dtype != np.uint8:
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)
        Image.fromarray(image_array).save(filepath)

    @staticmethod
    def normalize_image(image_array: np.ndarray) -> np.ndarray:
        """Normalize image to range [0, 255]."""
        min_val = np.min(image_array)
        max_val = np.max(image_array)

        if max_val > min_val:
            normalized = (image_array - min_val) / (max_val - min_val) * 255
            return normalized
        else:
            return np.zeros_like(image_array)


class ConvolutionProcessor:
    @staticmethod
    def calculate_padding_size(kernel: np.ndarray) -> Tuple[int, int]:
        """Calculate required padding size for a convolution kernel."""
        if kernel.ndim != 2:
            raise ValueError("Convolution kernel must be 2D")

        pad_width = (kernel.shape[1] - 1) // 2
        pad_height = (kernel.shape[0] - 1) // 2

        return pad_width, pad_height

    @staticmethod
    def apply_padding(image: np.ndarray, pad_width: int, pad_height: int) -> np.ndarray:
        """Apply zero padding to an image."""
        if image.ndim != 2:
            raise ValueError("Image array must be 2D")

        padded_image = np.zeros(
            (image.shape[0] + 2 * pad_height, image.shape[1] + 2 * pad_width),
            dtype=image.dtype
        )
        padded_image[pad_height:-pad_height, pad_width:-pad_width] = image

        return padded_image

    @staticmethod
    def apply_convolution(
            image: np.ndarray,
            kernel: np.ndarray,
            pad_width: Optional[int] = None,
            pad_height: Optional[int] = None
    ) -> np.ndarray:
        """Apply convolution operation to an image using the given kernel."""
        if image.ndim != 2:
            raise ValueError("Image array must be 2D")

        if kernel.ndim != 2:
            raise ValueError("Convolution kernel must be 2D")

        if pad_width is None or pad_height is None:
            pad_width, pad_height = ConvolutionProcessor.calculate_padding_size(kernel)

        new_height = max(1, image.shape[0] - 2 * pad_height)
        new_width = max(1, image.shape[1] - 2 * pad_width)

        result = np.zeros((new_height, new_width), dtype=np.float32)
        k_height, k_width = kernel.shape

        for i in range(new_height):
            for j in range(new_width):
                roi = image[i:i + k_height, j:j + k_width]
                result[i, j] = np.sum(roi * kernel)

        return result


class ImageFilter:
    """Class for different image filtering operations."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def get_output_path(self, filename: str) -> str:
        """Create a filepath within the output directory."""
        return os.path.join(self.output_dir, filename)

    @staticmethod
    def create_laplacian_kernel(neighborhood: int = 4) -> np.ndarray:
        """Create a Laplacian kernel with specified neighborhood connectivity."""
        if neighborhood == 4:
            return np.array([
                [0, 1, 0],
                [1, -4, 1],
                [0, 1, 0]
            ], dtype=np.float32)
        elif neighborhood == 8:
            return np.array([
                [1, 1, 1],
                [1, -8, 1],
                [1, 1, 1]
            ], dtype=np.float32)
        else:
            raise ValueError("Neighborhood must be either 4 or 8")

    @staticmethod
    def create_sobel_kernels() -> Tuple[np.ndarray, np.ndarray]:
        """Create Sobel operator kernels for edge detection."""
        sobel_x = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=np.float32)

        sobel_y = np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=np.float32)

        return sobel_x, sobel_y

    @staticmethod
    def create_averaging_kernel(size: int = 5) -> np.ndarray:
        """Create an averaging filter kernel of specified size."""
        return (1.0 / (size * size)) * np.ones((size, size), dtype=np.float32)


class Visualizer:
    """Class for visualizing image processing results."""

    @staticmethod
    def plot_filter_comparisons(
            original_image: np.ndarray,
            laplacian4_raw: np.ndarray,
            laplacian4_norm: np.ndarray,
            laplacian8_raw: np.ndarray,
            laplacian8_norm: np.ndarray,
            output_path: str,
            show_figure: bool = False
    ) -> None:
        """Create a comparison visualization with filtered images positioned closer to the original."""
        # Create a figure with optimized size
        fig = plt.figure(figsize=(18, 14))

        # Define a tighter grid layout with minimal spacing
        gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1])

        # Eliminate space between subplots
        gs.update(wspace=0.01, hspace=0.05)

        # Original image (spanning both rows in first column)
        ax0 = fig.add_subplot(gs[:, 0])
        ax0.imshow(original_image, cmap='gray')
        ax0.set_title('Original Image', fontsize=14, fontweight='bold', pad=10)
        ax0.axis('off')

        # 4-Neighbor Laplacian images (middle column)
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.imshow(laplacian4_raw, cmap='gray')
        ax1.set_title('4-Neighbor Laplacian (Raw)', fontsize=12, fontweight='bold', pad=8)
        ax1.axis('off')

        ax2 = fig.add_subplot(gs[1, 1])
        ax2.imshow(laplacian4_norm, cmap='gray')
        ax2.set_title('4-Neighbor Laplacian (Normalized)', fontsize=12, fontweight='bold', pad=8)
        ax2.axis('off')

        # 8-Neighbor Laplacian images (right column)
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(laplacian8_raw, cmap='gray')
        ax3.set_title('8-Neighbor Laplacian (Raw)', fontsize=12, fontweight='bold', pad=8)
        ax3.axis('off')

        ax4 = fig.add_subplot(gs[1, 2])
        ax4.imshow(laplacian8_norm, cmap='gray')
        ax4.set_title('8-Neighbor Laplacian (Normalized)', fontsize=12, fontweight='bold', pad=8)
        ax4.axis('off')

        fig.add_artist(
            plt.Line2D((0.36, 0.36), (0, 1), color='black', linestyle='-', linewidth=3,
                       transform=fig.transFigure, alpha=0.4))
        fig.add_artist(
            plt.Line2D((0.665, 0.665), (0, 1), color='black', linestyle='-', linewidth=3,
                       transform=fig.transFigure, alpha=0.4))

        # Extremely minimal borders - almost invisible
        for ax in [ax0, ax1, ax2, ax3, ax4]:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('whitesmoke')
                spine.set_linewidth(0.3)

        # Tight layout with minimal padding to bring images as close as possible
        plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, wspace=0, hspace=0.08)

        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        if show_figure:
            plt.show()
        else:
            plt.close(fig)

    @staticmethod
    def plot_sobel(x_sobel: np.ndarray, y_sobel: np.ndarray, sobel_magnitude: np.ndarray, filename: str) -> None:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(x_sobel, cmap="gray")
        axs[0].set_title('X-Sobel', fontsize=14, fontweight='bold', pad=10)

        axs[1].imshow(y_sobel, cmap="gray")
        axs[1].set_title('Y-Sobel', fontsize=14, fontweight='bold', pad=10)

        axs[2].imshow(sobel_magnitude, cmap="gray")
        axs[2].set_title('Sobel Magnitude', fontsize=14, fontweight='bold', pad=10)

        plt.tight_layout()
        plt.savefig(filename)
        plt.close(fig)

class ImageEnhancer:
    """Class for applying image enhancement operations."""

    @staticmethod
    def apply_power_law_transform(image: np.ndarray, gamma: float) -> np.ndarray:
        """Apply power-law (gamma) transformation to an image."""
        # Normalize to [0, 1] range
        normalized = image.astype(np.float32) / 255.0

        # Apply power-law transform
        transformed = np.power(normalized, gamma)

        # Convert back to uint8
        return (transformed * 255.0).astype(np.uint8)


def main():
    print("=" * 50)
    print(BColors.WARNING + BColors.BOLD + "INITIALIZATION" + BColors.ENDC + BColors.ENDC)

    # ======================================================
    # SECTION 1: DIRECTORY SETUP
    # ======================================================
    # Initialize directories
    base_dir = os.path.join("Images")
    os.makedirs(base_dir, exist_ok=True)

    # Subdirectories for different processing steps
    laplacian_dir = os.path.join(base_dir, "laplacian")
    edge_dir = os.path.join(base_dir, "edge_detection")
    enhanced_dir = os.path.join(base_dir, "enhanced")
    visualization_dir = os.path.join(base_dir, "visualizations")

    # Create directories
    for directory in [laplacian_dir, edge_dir, enhanced_dir, visualization_dir]:
        os.makedirs(directory, exist_ok=True)

    # ======================================================
    # SECTION 2: INITIALIZATION
    # ======================================================
    # Initialize filter and utility objects
    img_utils = ImageUtils()
    conv_processor = ConvolutionProcessor()
    img_filter = ImageFilter(base_dir)

    # Load original image
    original_image_path = "Images/skeleton_orig.tif"
    original_image = img_utils.load_image(original_image_path)
    print(f"Original Image Array:\n {original_image}")
    print(f"\nOriginal Image Shape: {original_image.shape}")

    # ======================================================
    # SECTION 3: LAPLACIAN FILTERING
    # ======================================================
    print("=" * 50)
    print(BColors.WARNING + BColors.BOLD + "APPLYING LAPLACIAN FILTERS" + BColors.ENDC + BColors.ENDC)

    lap4_kernel = img_filter.create_laplacian_kernel(neighborhood=4)
    lap8_kernel = img_filter.create_laplacian_kernel(neighborhood=8)

    pad_width, pad_height = conv_processor.calculate_padding_size(lap4_kernel)

    padded_image = conv_processor.apply_padding(original_image, pad_width, pad_height)
    print(f"Padded Image Shape: {padded_image.shape}")

    # Process with 4-neighbor Laplacian
    print("Processing with 4-neighbor Laplacian kernel...")
    lap4_result = conv_processor.apply_convolution(
        padded_image, lap4_kernel, pad_width, pad_height
    )

    # Save raw and normalized versions
    lap4_raw = np.clip(lap4_result, 0, 255).astype(np.uint8)
    lap4_norm = img_utils.normalize_image(lap4_result).astype(np.uint8)

    img_utils.save_image(
        lap4_raw,
        os.path.join(laplacian_dir, "laplacian4_raw.tif")
    )
    img_utils.save_image(
        lap4_norm,
        os.path.join(laplacian_dir, "(b)laplacian4_normalized.tif")
    )

    # Compute difference image (original - Laplacian)
    diff4_image = np.clip(
        original_image.astype(np.float32) - lap4_result,
        0, 255
    ).astype(np.uint8)

    img_utils.save_image(
        diff4_image,
        os.path.join(laplacian_dir, "(c)laplacian4_difference_with_original.tif")
    )

    # Process with 8-neighbor Laplacian
    print("Processing with 8-neighbor Laplacian kernel...")
    lap8_result = conv_processor.apply_convolution(
        padded_image, lap8_kernel, pad_width, pad_height
    )

    # Save raw and normalized versions
    lap8_raw = np.clip(lap8_result, 0, 255).astype(np.uint8)
    lap8_norm = img_utils.normalize_image(lap8_result).astype(np.uint8)

    img_utils.save_image(
        lap8_raw,
        os.path.join(laplacian_dir, "laplacian8_raw.tif")
    )
    img_utils.save_image(
        lap8_norm,
        os.path.join(laplacian_dir, "(b)laplacian8_normalized.tif")
    )

    # Compute difference image (original - Laplacian)
    diff8_image = np.clip(
        original_image.astype(np.float32) - lap8_result,
        0, 255
    ).astype(np.uint8)

    img_utils.save_image(
        diff8_image,
        os.path.join(laplacian_dir, "(c)laplacian8_difference_with_original.tif")
    )

    # ======================================================
    # SECTION 4: SOBEL EDGE DETECTION
    # ======================================================
    print("=" * 50)
    print(BColors.WARNING + BColors.BOLD + "APPLYING SOBEL EDGE DETECTION" + BColors.ENDC + BColors.ENDC)

    # Create Sobel kernels
    sobel_x_kernel, sobel_y_kernel = img_filter.create_sobel_kernels()

    # Apply Sobel operators
    print("Processing with x-sobel kernel...")
    sobel_x_result = conv_processor.apply_convolution(
        padded_image, sobel_x_kernel, pad_width, pad_height
    )

    print("Processing with y-sobel kernel...")
    sobel_y_result = conv_processor.apply_convolution(
        padded_image, sobel_y_kernel, pad_width, pad_height
    )

    print("Processing with sobel magnitude...")
    # Calculate gradient magnitude
    gradient_magnitude = np.sqrt(
        np.square(sobel_x_result) + np.square(sobel_y_result)
    )

    # Save normalized gradient magnitude
    sobel_magnitude = img_utils.normalize_image(gradient_magnitude).astype(np.uint8)
    img_utils.save_image(
        sobel_magnitude,
        os.path.join(edge_dir, "(d)sobel_magnitude.tif")
    )
    # ======================================================
    # SECTION 5: AVERAGING FILTER
    # ======================================================
    print("=" * 50)
    print(BColors.WARNING + BColors.BOLD + "APPLYING AVERAGING FILTER" + BColors.ENDC + BColors.ENDC)
    print("=" * 50)

    # Create averaging filter
    avg_kernel = img_filter.create_averaging_kernel(size=5)
    pad_width_avg, pad_height_avg = conv_processor.calculate_padding_size(avg_kernel)

    # Pad the Sobel magnitude image
    padded_sobel = conv_processor.apply_padding(
        sobel_magnitude, pad_width_avg, pad_height_avg
    )

    # Apply averaging filter
    avg_sobel_result = conv_processor.apply_convolution(
        padded_sobel, avg_kernel, pad_width_avg, pad_height_avg
    )

    # Save normalized result
    avg_sobel = img_utils.normalize_image(avg_sobel_result).astype(np.uint8)
    img_utils.save_image(
        avg_sobel,
        os.path.join(edge_dir, "(e)averaged_sobel.tif")
    )

    # ======================================================
    # SECTION 6: VISUALIZATION
    # ======================================================
    print(BColors.WARNING + BColors.BOLD + "CREATING VISUALIZATION" + BColors.ENDC + BColors.ENDC)

    print("Processing with Generating Laplacian Visualization...")
    Visualizer.plot_filter_comparisons(
        original_image=original_image,
        laplacian4_raw=lap4_raw,
        laplacian4_norm=lap4_norm,
        laplacian8_raw=lap8_raw,
        laplacian8_norm=lap8_norm,
        output_path=os.path.join(visualization_dir, "laplacian_filter_comparison.png")
    )

    print("Processing with Generating Sobel Visualization...")
    Visualizer.plot_sobel(
        np.clip(sobel_x_result, 0, 255),
        np.clip(sobel_y_result, 0, 255),
        sobel_magnitude,
        filename=os.path.join(visualization_dir, "sobel_comparison.png")
        )
    # ======================================================
    # SECTION 7: IMAGE ENHANCEMENT
    # ======================================================
    print("=" * 50)
    print(BColors.WARNING + BColors.BOLD + "CREATING MASKED AND ENHANCED IMAGES" + BColors.ENDC + BColors.ENDC)
    print("=" * 50)

    # Create masked image (multiply difference with averaged Sobel)
    masked_image_float = diff8_image.astype(np.float64) * avg_sobel.astype(np.float64)
    masked_image = img_utils.normalize_image(masked_image_float).astype(np.uint8)

    img_utils.save_image(
        masked_image,
        os.path.join(enhanced_dir, "(f)masked_edges.tif")
    )

    # Add masked image to original
    enhanced_image = np.clip(
        original_image.astype(np.float32) + masked_image.astype(np.float32),
        0, 255
    ).astype(np.uint8)

    img_utils.save_image(
        enhanced_image,
        os.path.join(enhanced_dir, "(g)enhanced_original.tif")
    )

    # Apply power-law transformation
    enhancer = ImageEnhancer()
    gamma_transformed = enhancer.apply_power_law_transform(enhanced_image, gamma=0.5)

    img_utils.save_image(
        gamma_transformed,
        os.path.join(enhanced_dir, "(h)gamma_enhanced.tif")
    )

    # ======================================================
    # SECTION 8: COMPLETION
    # ======================================================
    print(BColors.WARNING + BColors.BOLD + "PROCESSING COMPLETE" + BColors.ENDC + BColors.ENDC)
    print("=" * 50)
    print(BColors.OkGREEN + BColors.BOLD + f"All output files saved to: {base_dir}" + BColors.ENDC + BColors.ENDC)


if __name__ == '__main__':
    main()