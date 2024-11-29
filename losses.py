import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


class SobelEdgeWeightedCorrelationLoss(nn.Module):
    def __init__(self):
        super(SobelEdgeWeightedCorrelationLoss, self).__init__()

        # Sobel filters for computing gradients in X and Y directions
        self.sobel_x = torch.tensor([[-1, 0, 1],
                                     [-2, 0, 2],
                                     [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        self.sobel_y = torch.tensor([[-1, -2, -1],
                                     [0, 0, 0],
                                     [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def compute_edge_weight_map(self, targets):
        """Computes the edge-based weight map based on image gradients."""

        targets = targets.float().unsqueeze(1)  # Reshape to [batch, 1, H, W]

        # Apply Sobel filters to calculate horizontal and vertical gradients
        targets = F.pad(targets, (self.padding, self.padding, self.padding, self.padding), mode='reflect')
        grad_x = F.conv2d(targets, self.sobel_x.to(targets.device))
        grad_y = F.conv2d(targets, self.sobel_y.to(targets.device))

        # Compute gradient magnitude
        gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)

        # Normalize gradients to create the weight map in [0, 1]
        weight_map = gradient_magnitude / (gradient_magnitude.max() + 1e-6)
        return weight_map

    def forward(self, outputs, targets):
        # outputs [batch, C, H, W]
        # targets [batch, 1, H, W]
        # Convert logits to probabilities
        outputs = F.softmax(outputs, dim=1)

        # Squeeze the extra dimension in targets
        targets = targets.squeeze(1)  # Convert targets to [batch, H, W]

        # Calculate CrossEntropyLoss without reduction
        loss = F.cross_entropy(outputs, targets, reduction='none')

        # Compute edge-based weight map
        weight_map = self.compute_edge_weight_map(targets.unsqueeze(1))  # Add extra dimension back

        # Apply weight map to the loss
        weighted_loss = loss * weight_map.squeeze(1)  # Remove extra dimension

        return weighted_loss.mean()

# w_c нормально не реализован
class BoundaryAwareEdgeWeightedLoss(nn.Module):
    def __init__(self, w0=10, sigma=5):
        super(BoundaryAwareEdgeWeightedLoss, self).__init__()
        self.w0 = w0
        self.sigma = sigma
        self.distance_calculator = DistanceCalculator

    def compute_weight_map(self, d1, d2, wc):
        """
        Computes the weight map using distances d1 and d2, and the class frequency map wc.
        """
        return wc + self.w0 * torch.exp(-((d1 + d2) ** 2) / (2 * self.sigma ** 2))

    def forward(self, outputs, targets, wc):
        """
        Calculates the loss using weights and distances to object boundaries.

        :param outputs: Model logits [batch, num_classes, H, W]
        :param targets: Target labels [batch, H, W]
        :param wc: Weight map to compensate class frequency [batch, H, W]
        """

        # Generate d1 and d2 using DistanceCalculator based on outputs
        d1, d2 = self.distance_calculator(targets, outputs)

        # Compute the weight map
        weight_map = self.compute_weight_map(d1, d2, wc)

        # Convert logits to probabilities
        outputs = F.softmax(outputs, dim=1)

        # Calculate CrossEntropyLoss without averaging
        loss = F.cross_entropy(outputs, targets, reduction='none')

        # Apply the weight map to the loss
        weighted_loss = loss * weight_map

        return weighted_loss.mean()


class BoundaryLossCalculator(DistanceCalculator):
    """
    A subclass of DistanceCalculator to calculate boundary loss, which incorporates intersection and difference handling.
    """

    def _compute_min_and_second_min_for_loss(self):
        """
        Calculates minimum distances, considering intersections and differences between input and output arrays.
        """
        batch_size, height, width = self.input_array.shape
        min_distances = np.zeros((batch_size, height, width))

        for b in range(batch_size):
            intersection_coords = np.argwhere(np.logical_and(self.input_array[b] == 1, self.output_array_binar[b] == 1))
            difference_coords = np.argwhere(np.logical_and(self.output_array_binar[b] == 1, self.input_array[b] == 0))

            boundary_points = BoundaryProcessor.find_boundary_pixels(self.input_array[b])
            empty_points = np.argwhere(self.output_array_binar[b] == 1)

            if boundary_points.size == 0:
                continue

            min_dist, _ = self._compute_distances(boundary_points, empty_points)

            for idx, point in enumerate(empty_points):
                i, j = point
                # Assign negative distance for intersection and positive for difference
                if any(np.array_equal(point, coord) for coord in intersection_coords):
                    min_distances[b, i, j] = -1 * min_dist[idx]
                elif any(np.array_equal(point, coord) for coord in difference_coords):
                    min_distances[b, i, j] = min_dist[idx]

        return min_distances
    # STEGH PETQA LINI SUMMA @ST MI BATCHI
    def compute_boundary_loss(self):
        """
        Compute boundary loss by calculating minimum distances with intersection and difference handling.

        Returns:
            numpy.ndarray: Array of minimum distances with boundary loss adjustments.
        """
        return self.output_array * self._compute_min_and_second_min_for_loss()
