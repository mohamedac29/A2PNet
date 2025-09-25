import torch
import torch.nn.functional as F

def calculate_metrics(SR, GT, threshold=0.5, smooth=1e-5):
    """
    Calculates a suite of performance metrics for binary segmentation.
    All calculations are done in PyTorch for efficiency.

    Args:
        SR (torch.Tensor): The predicted segmentation map (usually model output before activation).
                           Shape: [B, C, H, W]
        GT (torch.Tensor): The ground truth segmentation map.
                           Should be binary (0s and 1s). Shape: [B, C, H, W]
        threshold (float): The threshold to binarize the prediction.
        smooth (float): A small epsilon value to prevent division by zero.

    Returns:
        dict: A dictionary containing all calculated metrics.
    """
    # Apply sigmoid and binarize the prediction
    SR = torch.sigmoid(SR)
    SR_binary = (SR > threshold).float()
    
    # Ensure GT is also binary (0 or 1)
    GT_binary = (GT > 0).float()

    # True Positive, False Positive, False Negative, True Negative
    TP = (SR_binary * GT_binary).sum()
    FP = SR_binary.sum() - TP
    FN = GT_binary.sum() - TP
    
    # Calculate total number of pixels to determine TN
    num_pixels = SR_binary.numel()
    TN = num_pixels - (TP + FP + FN)

    # --- Metric Calculations ---
    # Sensitivity (Recall)
    sensitivity = (TP + smooth) / (TP + FN + smooth)
    
    # Specificity
    specificity = (TN + smooth) / (TN + FP + smooth)
    
    # Precision
    precision = (TP + smooth) / (TP + FP + smooth)
    
    # F1 Score (Dice Coefficient on binary masks)
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity + smooth)
    
    # Accuracy
    accuracy = (TP + TN + smooth) / (TP + TN + FP + FN + smooth)
    
    # Intersection over Union (Jaccard Index)
    intersection = TP
    union = TP + FP + FN
    iou = (intersection + smooth) / (union + smooth)
    
    metrics = {
        'iou': iou.item(),
        'dice': f1_score.item(),
        'accuracy': accuracy.item(),
        'sensitivity': sensitivity.item(),
        'specificity': specificity.item(),
        'precision': precision.item(),
        'f1_score': f1_score.item()
    }
    
    return metrics


def dice_coef_soft(output, target, smooth=1e-5):
    """
    Calculates the 'soft' Dice coefficient, often used as a loss function.
    This version uses probabilities directly without thresholding.
    """
    output = torch.sigmoid(output).view(-1)
    target = target.view(-1)
    intersection = (output * target).sum()
    return (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)

if __name__ == '__main__':
    # --- Example Usage ---
    print("--- Example Metric Calculation ---")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create a dummy model output (raw logits) and ground truth
    dummy_output = torch.randn(2, 1, 256, 256).to(device) 
    dummy_gt = (torch.rand(2, 1, 256, 256) > 0.5).float().to(device) # Binary ground truth

    # Calculate metrics using the refactored function
    metrics = calculate_metrics(dummy_output, dummy_gt)

    print(f"Calculated Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # --- Example Soft Dice Calculation ---
    soft_dice = dice_coef_soft(dummy_output, dummy_gt)
    print(f"\nSoft Dice Coefficient: {soft_dice.item():.4f}")
