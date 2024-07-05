import torch
import numpy as np
import torch.nn.functional as F

def cross_entropy_loss(predicted, target):
    return F.cross_entropy(predicted, target)

def pixel_accuracy(predicted_image, mask):
    """
    Calculate the pixel accuracy between the predicted image and the ground truth mask.

    Args:
        predicted_image (torch.Tensor): Predicted image tensor of shape (N, C, H, W).
        mask (torch.Tensor): Ground truth mask tensor of shape (N, H, W).
        
    Returns:
        float: Pixel accuracy between the predicted image and the ground truth mask.
    """
    with torch.no_grad():
        # Convert predicted_image to class predictions
        predicted_image = torch.argmax(F.softmax(predicted_image, dim=1), dim=1)

        # Compare predicted_image with mask to get pixel-wise correctness
        correct = torch.eq(predicted_image, mask).int()

        # Calculate pixel accuracy
        accuracy = float(correct.sum()) / float(correct.numel())

    return accuracy


def mean_iou(predicted_label, label, eps=1e-10, num_classes=10):
    """
    Calculate the mean Intersection over Union (IoU) between the predicted labels and the ground truth labels.

    Args:
        predicted_label (torch.Tensor): Predicted label tensor of shape (N, C, H, W).
        label (torch.Tensor): Ground truth label tensor of shape (N, H, W).
        eps (float, optional): Epsilon value for numerical stability.
        num_classes (int, optional): Number of classes.

    Returns:
        float: Mean IoU value.

    """
    with torch.no_grad():
        # Convert predicted_label to class predictions
        predicted_label = F.softmax(predicted_label, dim=1)
        predicted_label = torch.argmax(predicted_label, dim=1)

        # Reshape predicted_label and label for easier computation
        predicted_label = predicted_label.contiguous().view(-1)
        label = label.contiguous().view(-1)

        iou_single_class = []
        for class_number in range(0, num_classes):
            true_predicted_class = predicted_label == class_number
            true_label = label == class_number

            if true_label.long().sum().item() == 0:
                iou_single_class.append(np.nan)
            else:
                # Calculate intersection and union
                intersection = (
                    torch.logical_and(true_predicted_class, true_label)
                    .sum()
                    .float()
                    .item())
                union = (
                    torch.logical_or(true_predicted_class, true_label)
                    .sum()
                    .float()
                    .item())

                # Calculate IoU for the current class
                iou = (intersection + eps) / (union + eps)
                iou_single_class.append(iou)

        # Calculate mean IoU across all classes
        return np.nanmean(iou_single_class)
