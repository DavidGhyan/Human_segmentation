import numpy as np
from scipy.spatial import KDTree
import cv2
from skimage.filters import threshold_otsu


class BoundaryProcessor:
    """
    Class responsible for identifying boundary pixels in a binary image.
    """

    @staticmethod
    def find_boundary_pixels(image):
        """
        Finds the boundary pixels in a binary image by applying morphological operations.

        Args:
            image (numpy.ndarray): Input binary image where boundaries are marked as 1 and empty space as 0.

        Returns:
            numpy.ndarray: Array of coordinates of boundary pixels.
        """
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        eroded = cv2.erode(closed, kernel)

        return np.argwhere(closed - eroded == 1)


class Binarizer:
    """
    Class responsible for binarizing an array using either a specified threshold or Otsu's method.
    """

    @staticmethod
    def binarize_array(array, threshold=None, metod_otsu=False):
        """
        Binarizes an array of data.

        Args:
            array (numpy.ndarray): Input data array.
            threshold (float, optional): Threshold for binarization. If not specified, Otsu's method is used.
            metod_otsu (bool, optional): If True, uses Otsu's method for automatic threshold selection.

        Returns:
            numpy.ndarray: Binarized array.
        """
        if metod_otsu:
            thresh = threshold_otsu(array)
        elif threshold is not None:
            thresh = threshold
        else:
            raise ValueError("Threshold not specified or Otsu's method not selected.")

        return (array > thresh).astype(int)


class DistanceCalculator:
    """
    Class responsible for calculating minimum and second minimum distances from each empty pixel (0)
    to the nearest boundary pixels (1) in a batch of 3D images.
    """

    def __init__(self, input_array, output_array, binarizer=Binarizer):
        """
        Initializes the DistanceCalculator with input and output arrays.

        Parameters:
            input_array (numpy.ndarray): A 3D binary array of shape [batch, H, W] where 1 represents boundaries and 0 represents empty space.
            output_array (numpy.ndarray): A 3D array to be binarized for distance calculation.
            binarizer (class, optional): The class used to binarize the output array. Defaults to the Binarizer class.
        """
        self.input_array = input_array
        self.output_array = output_array
        self.output_array_binar = binarizer.binarize_array(output_array, metod_otsu=True)

    def _compute_distances(self, boundary_points, empty_points):
        """
        Computes the minimum and second minimum distances from each empty pixel to the nearest boundary pixel.

        Args:
            boundary_points (numpy.ndarray): Coordinates of the boundary pixels.
            empty_points (numpy.ndarray): Coordinates of the empty pixels.

        Returns:
            tuple: Minimum and second minimum distances for each empty point.
        """
        tree = KDTree(boundary_points)
        min_distances = np.zeros(len(empty_points))
        second_min_distances = np.zeros(len(empty_points))

        for idx, point in enumerate(empty_points):
            i, j = point
            nearest_distances, _ = tree.query((i, j), k=2)
            min_distances[idx] = nearest_distances[0]
            second_min_distances[idx] = nearest_distances[1]

        return min_distances, second_min_distances

    def compute_min_and_second_min_distance(self):
        """
        Computes the minimum and second minimum distances from each empty pixel to the nearest boundary pixels.

        Returns:
            tuple: Arrays of minimum and second minimum distances for each slice in the batch.
        """
        batch_size, height, width = self.input_array.shape
        min_distances = np.zeros((batch_size, height, width))
        second_min_distances = np.zeros((batch_size, height, width))

        for b in range(batch_size):
            boundary_points = np.argwhere(self.input_array[b] == 1)
            empty_points = np.argwhere(self.output_array_binar[b] == 0)

            if boundary_points.size == 0:
                continue  # Skip if no boundary points in the batch slice

            min_dist, sec_min_dist = self._compute_distances(boundary_points, empty_points)

            # Fill the calculated distances into the corresponding positions in the distance arrays
            for idx, point in enumerate(empty_points):
                i, j = point
                min_distances[b, i, j] = min_dist[idx]
                second_min_distances[b, i, j] = sec_min_dist[idx]

        return min_distances, second_min_distances
