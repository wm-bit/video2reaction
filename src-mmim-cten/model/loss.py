import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from typing import Union
from torch import Tensor
import numpy as np
from torch.nn import CrossEntropyLoss


class CECADLoss(nn.Module):
    def __init__(self, weight=0.5):
        super().__init__()
        self.weight = weight
        self.ce = CrossEntropyLoss()
        self.cad = CumulativeAbsoluteDistanceLoss()

    def forward(self, pred_logit, gt):
        return self.ce(pred_logit, gt) + self.weight * self.cad(pred_logit, gt)
    

class CECJSLoss(nn.Module):
    def __init__(self, weight=0.5):
        super().__init__()
        self.weight = weight
        self.ce = CrossEntropyLoss()
        self.cjs = CJSLoss()

    def forward(self, pred_logit, gt):
        return self.ce(pred_logit, gt) + self.weight * self.cjs(pred_logit, gt)



class IIAMetric(nn.Module):
    def __init__(self, device=torch.device("cuda")):
        super(IIAMetric, self).__init__()
        self.device = device
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def get_image_embedding(self, image: Union[str, Image.Image]) -> torch.Tensor:
        if isinstance(image, str):
            image = Image.open(image)
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.clip_model.get_image_features(**inputs)
        return outputs
    
    def forward(self, original_image, edited_image):
        original_image_embedding = self.get_image_embedding(original_image)
        edited_image_embedding = self.get_image_embedding(edited_image)
        return torch.nn.functional.cosine_similarity(original_image_embedding, edited_image_embedding)


EPS = 1e-8

class CumulativeAbsoluteDistanceLoss(nn.Module):
    """
    Cumulative Absolute Distance (CAD) Loss for ordinal distributions.
    
    Computes the mean absolute difference between the CDFs of predicted and target distributions.
    Assumes inputs are distributions over ordinal labels sorted by valence.
    """

    def __init__(self, reduction: str = 'mean'):
        """
        Args:
            reduction (str): Specifies the reduction to apply to the output:
                             'none' | 'mean' | 'sum'. Default: 'mean'
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, logit_predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logit_predicted (Tensor): Predicted logit (not distribution), shape (B, C)
            target (Tensor): Target distribution, shape (B, C)
        Returns:
            Tensor: CAD loss
        """
        # Ensure inputs are probability distributions
        predicted = torch.softmax(logit_predicted, dim=1)
        predicted = torch.clamp(predicted, min=0.0)
        predicted = predicted / predicted.sum(dim=1, keepdim=True).clamp(min=1e-8)
        target = torch.clamp(target, min=0.0)
        target = target / target.sum(dim=1, keepdim=True).clamp(min=1e-8)

        # Compute cumulative sums (CDFs)
        cdf_pred = torch.cumsum(predicted, dim=1)
        cdf_target = torch.cumsum(target, dim=1)

        # Compute absolute difference and sum across labels
        cad = torch.sum(torch.abs(cdf_pred - cdf_target), dim=1)

        # Apply reduction
        if self.reduction == 'mean':
            return cad.mean()
        elif self.reduction == 'sum':
            return cad.sum()
        else:  # 'none'
            return cad
        
    def score(self, D: np.ndarray, D_pred: np.ndarray) -> float:
        """
        Compute the Cumulative Absolute Distance (CAD) loss score.
        
        Args:
            D (np.ndarray): True distribution, shape (B, C)
            D_pred (np.ndarray): Predicted distribution, shape (B, C)
        
        Returns:
            float: CAD loss score.
        """
        loss = self.forward(torch.tensor(D), torch.tensor(D_pred))
        return loss.item()
        

def compute_va_distance_matrix(class_va):
    """
    Computes pairwise Euclidean distances between classes in valence-arousal space.
    Args:
        class_va: Tensor of shape (C, 2)
    Returns:
        distance_matrix: Tensor of shape (C, C)
    """
    class_va = torch.tensor(class_va)
    diff = class_va.unsqueeze(1) - class_va.unsqueeze(0)  # (C, C, 2)
    dist = torch.norm(diff, dim=2)  # (C, C)
    return dist

def is_psd(mat):
    return bool((mat == mat.T).all() and (torch.linalg.eigvals(mat).real>=0).all())

class QFD2Loss(nn.Module):
    def __init__(self, distance_matrix=None):
        super(QFD2Loss, self).__init__()
        self.distance_matrix = distance_matrix
        
    def forward(self, D_pred_logit, D):
        D_pred = torch.softmax(D_pred_logit, dim=1)
        Q = D - D_pred
        L = D.shape[1]
        j = torch.arange(L).view(L, 1)
        k = torch.arange(L).view(1, L)
        if self.distance_matrix is None:
            A = 1 - torch.abs(j - k).float() / (L - 1)
        else:
            A = 1 - torch.abs(self.distance_matrix[j, k]) / ((self.distance_matrix.max() - self.distance_matrix.min()) + EPS)
        A = A.to(D.device)
        # Make sure A and D are the same dtype
        A = A.to(Q.dtype)
        # Check to see if A is positive semi-definite
        assert is_psd(A), "Distance matrix is not positive semi-definite"
        result = torch.matmul(torch.matmul(Q, A), Q.transpose(1, 0))
        return torch.mean(torch.diagonal(result))
    
    def score(self, D_pred: np.ndarray, D: np.ndarray) -> float:
        """
        Compute the QFD2 loss score.
        
        Args:
            D (np.ndarray): True distribution, shape (B, C)
            D_pred (np.ndarray): Predicted distribution, shape (B, C)
        
        Returns:
            float: QFD2 loss score.
        """
        loss = self.forward(torch.tensor(D), torch.tensor(D_pred))
        return loss.item()


class CJSLoss(nn.Module):
    def forward(self, D_pred_logit, D):
        D_pred = torch.softmax(D_pred_logit, dim=1)
        loss = 0.0
        for i in range(1, D.shape[1] + 1):
            D_i = D[:, :i]
            D_pred_i = D_pred[:, :i]
            m = 0.5 * (D_i + D_pred_i + EPS)
            js = 0.5 * (
                F.kl_div(m.log(), D_i, reduction='batchmean') +
                F.kl_div(m.log(), D_pred_i, reduction='batchmean')
            )
            loss += js
        return loss
    
    def score(self, D_pred: np.ndarray, D: np.ndarray) -> float:
        """
        Compute the CJSLoss score.
        
        Args:
            D (np.ndarray): True distribution, shape (B, C)
            D_pred (np.ndarray): Predicted distribution, shape (B, C)
        
        Returns:
            float: CJSLoss score.
        """
        loss = self.forward(torch.tensor(D), torch.tensor(D_pred))
        return loss.item()


def distance_weighted_kl(
    pred_dist: torch.Tensor,         # (B, C)
    target_dist: torch.Tensor,       # (B, C)
    A: torch.Tensor,                 # (C, C)
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Compute the distance-aware KL divergence between two distributions,
    weighted by distances in valence-arousal space.

    Args:
        pred_dist (torch.Tensor): Model output distribution, shape (B, C)
        target_dist (torch.Tensor): Ground-truth distribution, shape (B, C)
        A (torch.Tensor): Emotion distance matrix, shape (C, C)
        epsilon (float): Small value for numerical stability.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    p = pred_dist + epsilon  # Avoid log(0)
    q = target_dist + epsilon

    log_ratio = torch.log(q / p)             # (B, C)
    weighted = q.unsqueeze(2) * log_ratio.unsqueeze(2) * A.unsqueeze(0)  # (B, C, C)
    loss = weighted.sum(dim=(1, 2)).mean()   # scalar
    return loss

class DistanceAwareEmotionLoss(nn.Module):
    def __init__(self, class_va: torch.Tensor):
        """
        Args:
            class_va (torch.Tensor): Class-level valence-arousal coordinates, shape (C, 2)
        """
        super().__init__()
        A = compute_va_distance_matrix(class_va)  # (C, C)
        self.register_buffer("A", A)

    def forward(
        self,
        pred_dist: torch.Tensor,    # (B, C)
        target_dist: torch.Tensor   # (B, C)
    ) -> torch.Tensor:
        return distance_weighted_kl(pred_dist, target_dist, self.A)

import torch
import torch.nn.functional as F


# Source: Imbalanced Label Distribution Learning (AAAI-23) --> address imbaalnced ldl problem
class DistributionBalancedFocalLoss(torch.nn.Module):
    def __init__(self, num_classes, gamma=2, lambda_val=0.5, smooth_eps=1e-8):
        """
        Distribution-balanced focal loss for multi-label classification with negative-tolerant regularization.
        
        Args:
            num_classes (int): Number of classes (labels).
            gamma (float): Focal loss focusing parameter, typically > 0. Default is 2.
            lambda_val (float): Balance hyperparameter for mixing positive and negative labels. Default is 0.5.
            smooth_eps (float): Small value for smoothing.
        """
        super(DistributionBalancedFocalLoss, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.lambda_val = lambda_val
        self.smooth_eps = smooth_eps
        
    def forward(self, predictions, target, P_C, P_I):
        """
        Forward pass for Distribution-balanced Focal Loss.
        
        Args:
            predictions (torch.Tensor): The predicted logits, shape (batch_size, num_classes).
            target (torch.Tensor): The ground truth labels, shape (batch_size, num_classes), where each entry is 0 or 1.
            P_C (torch.Tensor): The expectation of label-level sampling frequency, shape (batch_size, num_classes).
            P_I (torch.Tensor): The expectation of instance-level sampling frequency, shape (batch_size, num_classes).
        
        Returns:
            torch.Tensor: The computed distribution-balanced focal loss.
        """
        
        # Compute re-balanced weights r_j_i
        r_j_i = P_C / P_I
        
        # Smooth the re-balancing weight
        r_j_i = torch.clamp(r_j_i, min=self.smooth_eps, max=1.0)
        
        # Compute the negative-tolerant regularization term (class-specific bias)
        logits = predictions  # Logits from the model
        q = torch.exp(logits - torch.mean(logits, dim=1, keepdim=True))  # NTR - class-specific bias
        
        # Normalize q to get q_ji (the predicted probability distribution)
        q = q / q.sum(dim=1, keepdim=True)
        
        # Compute focal loss components
        focal_loss = (1 - torch.sigmoid(logits)) ** self.gamma
        
        # Compute the log-term (cross-entropy part)
        log_term = torch.log((target + self.smooth_eps) / (q + self.smooth_eps))
        
        # Compute the final loss
        loss = r_j_i * focal_loss * (1 - (1 - 1 / self.lambda_val) * target + 1 / self.lambda_val) * log_term
        
        # Sum over the batch and classes
        loss = torch.sum(loss)
        
        return loss


import numpy as np
# from pyldl.metrics import score
from dataset.v2r_latent_set import REACTION_VA_MATRIX
from sklearn.metrics import mean_squared_error

# def compute_all_distribution_metrics(
#     pred_dist: np.ndarray,  # (N, C)
#     target_dist: np.ndarray,  # (B, C)
#     class_va: np.ndarray = REACTION_VA_MATRIX # (C, 2)
# ):
#     # Assume that distribution has not been sorted
#     # Get standard LDL metrics
#     metrics = score(pred_dist, target_dist, return_dict=True)
#     # Compute ordinal-based metrics
#     metrics["cad_ordinal_va"] = CumulativeAbsoluteDistanceLoss().score(pred_dist, target_dist)
#     metrics["qfd_ordinal_va"] = QFD2Loss().score(pred_dist, target_dist)
#     metrics["cjs_ordinal_va"] = CJSLoss().score(pred_dist, target_dist)
#     # Compute distance-aware metrics
#     metrics["qfd_sim_va"] = QFD2Loss(compute_va_distance_matrix(class_va)).score(pred_dist, target_dist)
#     metrics["mse"] = mean_squared_error(pred_dist, target_dist)

#     # TODO: Compute valence distribution metrics
#     return metrics

from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import top_k_accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, accuracy_score, roc_auc_score, balanced_accuracy_score
def compute_all_classification_metrics(
        pred_dist: np.ndarray,  # (N, C)
        target_dist: np.ndarray,  # (B, C),
):
    """
    Compute classification metrics for the predicted and target distributions.

    Args:
        pred_dist (np.ndarray): Predicted distribution, shape (N, C)
        target_dist (np.ndarray): Target distribution, shape (B, C)
        metrics (list): List of metrics to compute. Default is ["accuracy", "precision", "recall", "f1", "macro_f1", "micro_f1", "auc"].

    Returns:
        dict: Dictionary of computed metrics.
    """
    pred_classes = np.argmax(pred_dist, axis=1)
    target_classes = np.argmax(target_dist, axis=1)
    print(f"pred_classes: {pred_classes.shape}")
    print(f"target_classes: {target_classes.shape}")
    # TODO: set a threshold for positive class

    results = {}

    results["accuracy"] = accuracy_score(target_classes, pred_classes)
    results["precision"] = precision_score(target_classes, pred_classes, average='macro')
    results["recall"] = recall_score(target_classes, pred_classes, average='macro')
    results["f1"] = f1_score(target_classes, pred_classes, average='macro')
    results["macro_f1"] = f1_score(target_classes, pred_classes, average='macro')
    results["micro_f1"] = f1_score(target_classes, pred_classes, average='micro')
    results["balanced_accuracy"] = balanced_accuracy_score(target_classes, pred_classes)
    #results["auc"] = roc_auc_score(target_classes, pred_classes, average=None)
    return results