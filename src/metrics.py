#%%
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from typing import Union
from torch import Tensor
import numpy as np
#%%
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

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predicted (Tensor): Predicted distribution, shape (B, C)
            target (Tensor): Target distribution, shape (B, C)
        Returns:
            Tensor: CAD loss
        """
        # Ensure inputs are probability distributions
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

class SmoothCADLoss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        """
        Args:
            reduction (str): Specifies the reduction to apply to the output:
                             'none' | 'mean' | 'sum'. Default: 'mean'
        """
        super().__init__()
        self.reduction = reduction
        self.epsilon = 1e-6

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predicted (Tensor): Predicted distribution, shape (B, C)
            target (Tensor): Target distribution, shape (B, C)
        Returns:
            Tensor: CAD loss
        """
        # Ensure inputs are probability distributions
        predicted = torch.clamp(predicted, min=0.0)
        predicted = predicted / predicted.sum(dim=1, keepdim=True).clamp(min=1e-8)
        target = torch.clamp(target, min=0.0)
        target = target / target.sum(dim=1, keepdim=True).clamp(min=1e-8)

        # Compute cumulative sums (CDFs)
        cdf_pred = torch.cumsum(predicted, dim=1)
        cdf_target = torch.cumsum(target, dim=1)

        # Compute smooth MSE
        # TODO: Compute smooth CAD
        cad = torch.sqrt((cdf_pred - cdf_target) ** 2 + self.epsilon).sum(dim=1)

        # Apply reduction
        if self.reduction == 'mean':
            return cad.mean()
        elif self.reduction == 'sum':
            return cad.sum()
        else:  # 'none'
            return cad
        

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
        
    def forward(self, D, D_pred):
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
    
    def score(self, D: np.ndarray, D_pred: np.ndarray) -> float:
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
    def forward(self, D, D_pred):
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
    
    def score(self, D: np.ndarray, D_pred: np.ndarray) -> float:
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

#%%
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

#%%
import numpy as np
from src.dataset import REACTION_VA_MATRIX
from sklearn.metrics import mean_squared_error
#%%
############# for ldl ###################################################33
def euclidean(distribution_real, distribution_predict):
    height = distribution_real.shape[0]
    return np.sum(np.sqrt(np.sum((distribution_real - distribution_predict) ** 2, 1))) / height


def squared_chord(distribution_real, distribution_predict):
    height = distribution_real.shape[0]
    numerator = (np.sqrt(distribution_real) - np.sqrt(distribution_predict)) ** 2
    denominator = np.sum(numerator)
    return denominator / height

def sorensen(distribution_real, distribution_predict):
    height = distribution_real.shape[0]
    numerator = np.sum(np.abs(distribution_real - distribution_predict), 1)
    denominator = np.sum(distribution_real + distribution_predict, 1)
    return np.sum(numerator / denominator) / height

def squared_chi2(distribution_real, distribution_predict):
    height = distribution_real.shape[0]
    numerator = (distribution_real - distribution_predict) ** 2
    denominator = distribution_real + distribution_predict
    return np.sum(numerator / denominator) / height


def kl(distribution_real, distribution_predict):
    kl_divergence = 0
    for i in range(distribution_real.shape[0]):
        kl_divergence += np.sum(distribution_real[i] * np.log(distribution_real[i] / (distribution_predict[i] + 1e-10) + 1e-10))
    kl_divergence /= distribution_real.shape[0]
    return kl_divergence

def js(distribution_real, distribution_predict):
    height = distribution_real.shape[0]
    m = 0.5 * (distribution_real + distribution_predict)
    return 0.5 * (kl(distribution_real, m) + kl(distribution_predict, m)) / height


def intersection(distribution_real, distribution_predict):
    height, width = distribution_real.shape
    inter = 0.
    for i in range(height):
        for j in range(width):
            inter += np.min([distribution_real[i][j], distribution_predict[i][j]])
    return inter / height


def fidelity(distribution_real, distribution_predict):
    height = distribution_real.shape[0]
    return np.sum(np.sqrt(distribution_real * distribution_predict)) / height


def chebyshev(distribution_real, distribution_predict):
    height = distribution_real.shape[0]
    return np.sum(np.max(np.abs(distribution_real-distribution_predict), 1)) / height


def clark(distribution_real, distribution_predict):
    height = distribution_real.shape[0]
    return np.sum(np.sqrt(np.sum((distribution_real-distribution_predict)**2 / ((distribution_real+distribution_predict)**2 + EPS), 1))) / height


def canberra(distribution_real, distribution_predict):
    height = distribution_real.shape[0]
    return np.sum(np.abs(distribution_real-distribution_predict) / (distribution_real+distribution_predict)) / height


def cosine(distribution_real, distribution_predict):
    height = distribution_real.shape[0]
    return np.sum(np.sum(distribution_real*distribution_predict, 1) / (np.sqrt(np.sum(distribution_real**2, 1)) *\
           np.sqrt(np.sum(distribution_predict**2, 1)))) / height

def compute_all_distribution_metrics(
    pred_dist: np.ndarray,  # (N, C)
    target_dist: np.ndarray,  # (B, C)
    class_va: np.ndarray = REACTION_VA_MATRIX # (C, 2)
):
    # Assume that distribution has not been sorted
    # Get standard LDL metrics
    metrics = {}
    metrics["euclidean"] = euclidean(target_dist, pred_dist)
    metrics["squared_chord"] = squared_chord(target_dist, pred_dist)
    metrics["sorensen"] = sorensen(target_dist, pred_dist)
    metrics["squared_chi2"] = squared_chi2(target_dist, pred_dist)
    metrics["kl"] = kl(target_dist, pred_dist)
    metrics["intersection"] = intersection(target_dist, pred_dist)
    metrics["fidelity"] = fidelity(target_dist, pred_dist)
    metrics["chebyshev"] = chebyshev(target_dist, pred_dist)
    metrics["clark"] = clark(target_dist, pred_dist)
    metrics["canberra"] = canberra(target_dist, pred_dist)
    metrics["cosine"] = cosine(target_dist, pred_dist)
    # Compute Jensen-Shannon divergence
    metrics["jsd"] = js(target_dist, pred_dist)
    # Compute Pearson and Spearman correlation
    metrics["pearson"] = pearsonr(target_dist.flatten(), pred_dist.flatten())[0]
    metrics["spearman"] = spearmanr(target_dist.flatten(), pred_dist.flatten())[0]
    # Compute ordinal-based metrics
    metrics["cad_ordinal_va"] = CumulativeAbsoluteDistanceLoss().score(pred_dist, target_dist)
    metrics["qfd_ordinal_va"] = QFD2Loss().score(pred_dist, target_dist)
    metrics["cjs_ordinal_va"] = CJSLoss().score(pred_dist, target_dist)
    # Compute distance-aware metrics
    metrics["qfd_sim_va"] = QFD2Loss(compute_va_distance_matrix(class_va)).score(pred_dist, target_dist)
    metrics["mse"] = mean_squared_error(pred_dist, target_dist)
    

    # TODO: Compute valence distribution metrics
    return metrics

#%%
def compute_eec(pred_dist: np.ndarray, target_dist: np.ndarray, reaction_va_matrix: np.ndarray = REACTION_VA_MATRIX) -> float:
    """
    Compute the Emotional Consistency Coefficient (ECC) using hard label distributions and emotional distances.
    
    Parameters:
        pred_dist (np.ndarray): Predicted probability distributions (shape: N x num_classes).
        target_dist (np.ndarray): Target probability distributions (shape: N x num_classes).
        reaction_va_matrix (np.ndarray): A matrix of emotional distances between classes (shape: num_classes x num_classes).
        
    Returns:
        float: The ECC value.
    """
    
    
    # Ensure inputs are numpy arrays
    pred_dist = np.array(pred_dist)
    target_dist = np.array(target_dist)
    
    # Number of classes
    num_classes = pred_dist.shape[1]  # Shape: (N, num_classes)
    
    # Initialize the confusion matrix S_ij based on top labels (hard labels)
    S = np.zeros((num_classes, num_classes))
    
    # Iterate over each sample
    for i in range(pred_dist.shape[0]):
        # Get the top predicted label and the top ground truth label (hard labels)
        pred_label = np.argmax(pred_dist[i])  # Top predicted label
        true_label = np.argmax(target_dist[i])  # Top ground truth label
        
        # Update the confusion matrix: increment count from true class to predicted class
        S[true_label, pred_label] += 1
    
    # Initialize the emotional distance matrix W_ij
    W = compute_va_distance_matrix(reaction_va_matrix)  # Shape: (num_classes, num_classes)
    # TODO: Rescale W to [1, 22] range
    W = (W - W.min()) / (W.max() - W.min()) * 21 + 1  # Rescale to [1, 22]
    
    # # Compute emotional distances W_ij using the reaction_va_matrix
    # for i in range(num_classes):
    #     for j in range(num_classes):
    #         if i != j:
    #             # Use the reaction_va_matrix to get emotional distance between i and j
    #             W[i, j] = reaction_va_matrix[i, j]
    #         else:
    #             W[i, j] = EPS  # No emotional distance for the same class

    # Compute the EEC metric
    eec = 0
    N = np.sum(S)  # Total number of samples (sum of all confusion matrix entries)
    
    # Sum over all confusion matrix entries
    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                # Correctly classified samples (diagonal of the confusion matrix)
                eec += S[i, j] / N
            else:
                # Misclassified samples with emotional distance weight
                eec += (S[i, j] / N) * (1 / W[i, j])
    
    return eec.item()

import numpy as np

def compute_eec_multi_label_top_k(pred_dist: np.ndarray, target_dist: np.ndarray, k: int = 5, reaction_va_matrix: np.ndarray = REACTION_VA_MATRIX) -> float:
    """
    Compute the Emotional Consistency Coefficient (ECC) for multi-label top-k predictions using soft label distributions and emotional distances.
    
    Parameters:
        pred_dist (np.ndarray): Predicted probability distributions (shape: N x num_classes).
        target_dist (np.ndarray): Target probability distributions (shape: N x num_classes).
        k (int): The top-k number of predictions to consider (default is 5).
        reaction_va_matrix (np.ndarray): A matrix of emotional distances between classes (shape: num_classes x num_classes).
        
    Returns:
        float: The ECC value.
    """
    
    # Ensure inputs are numpy arrays
    pred_dist = np.array(pred_dist)
    target_dist = np.array(target_dist)
    
    # Number of classes
    num_classes = pred_dist.shape[1]  # Shape: (N, num_classes)
    
    # Initialize the confusion matrix S_ij based on top-k labels
    S = np.zeros((num_classes, num_classes))
    
    # Iterate over each sample
    for i in range(pred_dist.shape[0]):
        # Get the top-k predicted labels (indices of top-k probabilities)
        top_k_pred = np.argsort(pred_dist[i])[::-1][:k]
        
        # Get the top-k true labels (indices of top-k true labels)
        top_k_true = np.argsort(target_dist[i])[::-1][:k]
        
        # Update the confusion matrix: increment count from true class to predicted class
        for true_label in top_k_true:
            for pred_label in top_k_pred:
                S[true_label, pred_label] += 1
    
    # Initialize the emotional distance matrix W_ij using the reaction_va_matrix
    W = compute_va_distance_matrix(reaction_va_matrix)  # Shape: (num_classes, num_classes)
    # TODO: Rescale W to [1, 22] range
    W = (W - W.min()) / (W.max() - W.min()) * 21 + 1  # Rescale to [1, 22]
    
    # Compute the EEC metric
    eec = 0
    N = np.sum(S)  # Total number of samples (sum of all confusion matrix entries)
    
    # Sum over all confusion matrix entries
    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                # Correctly classified samples (diagonal of the confusion matrix)
                eec += S[i, j] / N
            else:
                # Misclassified samples with emotional distance weight
                eec += (S[i, j] / N) * (1 / W[i, j])
    
    return eec.item()


import numpy as np

def compute_mean_reciprocal_rank(pred_dist: np.ndarray, target_dist: np.ndarray) -> float:
    """
    Compute the Mean Reciprocal Rank (MRR) for the predicted and target distributions.

    Parameters:
    - pred_dist: A 2D numpy array of predicted probabilities, shape (num_samples, num_labels).
    - target_dist: A 2D numpy array of target probabilities (soft labels), shape (num_samples, num_labels).
    - k: Top k predictions to consider for computing the rank (optional, defaults to 5).

    Returns:
    - MRR: Mean Reciprocal Rank.
    """
    # Initialize a list to store reciprocal ranks
    reciprocal_ranks = []

    # Iterate over all samples
    for i in range(pred_dist.shape[0]):
        # Get the index of the correct label (the label with the highest probability in target_dist)
        correct_label = np.argmax(target_dist[i])  # Find the class with the highest probability in target_dist
        
        # Get the sorted indices of predicted probabilities (descending order)
        sorted_indices = np.argsort(pred_dist[i])[::-1]
        
        # Find the rank of the correct label in the sorted list
        rank = np.where(sorted_indices == correct_label)[0][0] + 1  # +1 for 1-based index
        
        # Compute the reciprocal rank
        reciprocal_ranks.append(1.0 / rank)
    
    # Compute the Mean Reciprocal Rank (MRR)
    mrr = np.mean(reciprocal_ranks)
    
    return mrr

def compute_mean_average_precision_at_k(
    pred_dist: np.ndarray,  # (N, C)
    target_dist: np.ndarray,  # (B, C)
    k: int = 5
) -> float:
    """
    Compute the Mean Average Precision at k (MAP@k) for the predicted and target distributions.

    Parameters:
    - pred_dist: A 2D numpy array of predicted probabilities, shape (num_samples, num_labels).
    - target_dist: A 2D numpy array of target probabilities (soft labels), shape (num_samples, num_labels).
    - k: Top k predictions to consider for computing MAP@k.

    Returns:
    - MAP@k: Mean Average Precision at k.
    """
    # Initialize a list to store average precisions
    average_precisions = []

    # Iterate over all samples
    for i in range(pred_dist.shape[0]):
        # Get the top k indices from the predicted distribution
        top_k_indices = np.argsort(pred_dist[i])[-k:]  # Get indices of top k predictions
        
        # Compute precision at k
        relevant_items = np.sum(target_dist[i][top_k_indices])
        precision_at_k = relevant_items / k
        
        # Append to the list of average precisions
        average_precisions.append(precision_at_k)
    
    # Compute the Mean Average Precision at k (MAP@k)
    map_at_k = np.mean(average_precisions)
    
    return map_at_k


from src.dataset import SENTIMENT_2_FINER_GRAINED_MAPPING, SENTIMENT_CLASSES, SENTIMENT_2_FINER_GRAINED_INDICES_MAPPING, REACTION_INDEX
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import top_k_accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, accuracy_score, roc_auc_score, balanced_accuracy_score
REACTION_MEAN_PROB = {'admiration': 0.2908085971668617,
 'disapproval': 0.23022106584989568,
 'disgust': 0.08605696391042585,
 'disappointment': 0.0808909355158524,
 'sadness': 0.09052509577053153,
 'fear': 0.13227519308570967,
 'confusion': 0.08328831825908062,
 'excitement': 0.06339880249341164,
 'amusement': 0.24901794791652737,
 'joy': 0.075733491816312,
 'surprise': 0.07232418992119678,
 'curiosity': 0.056463071168701126,
 'approval': 0.06933460000719317,
 'realization': 0.05068421403825185,
 'relief': 0.04932925897016657,
 'grief': 0.08131036191091369,
 'anger': 0.05679020983547485,
 'caring': 0.052975453835455454,
 'embarrassment': 0.047028451883238946,
 'annoyance': 0.04474940449968487,
 'nervousness': 0.052041157529265226
 }
from typing import Dict
def compute_all_classification_metrics(
        pred_dist: np.ndarray,  # (N, C)
        target_dist: np.ndarray,  # (B, C),
        classwise_threshold: Dict[str, float] = REACTION_MEAN_PROB,
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

    # Single-label metrics
    pred_classes = np.argmax(pred_dist, axis=1)
    target_classes = np.argmax(target_dist, axis=1)
    # TODO: set a threshold for positive class

    results = {}

    f1_classwise = f1_score(target_classes, pred_classes, average=None, labels=range(len(REACTION_INDEX)))
    results["f1_classwise"] = {reaction: f1_classwise[i] for reaction, i in REACTION_INDEX.items()}


    # Proposed metrics
    # Single label metric
    # MAE of predicted probability for groundtruth top 1 class
    results["mae_top_1_gt"] = np.mean(np.abs(pred_dist[np.arange(len(pred_dist)), target_classes] - target_dist[np.arange(len(target_dist)), target_classes]))
    # MAE of predicted probability of predicted top 1 class
    results["mae_top_1_pred"] = np.mean(np.abs(pred_dist[np.arange(len(pred_dist)), pred_classes] - target_dist[np.arange(len(target_dist)), pred_classes]))
    # EEC
    results["ecc"] = compute_eec(pred_dist, target_dist)
    # Multilabel metrics
    for p_threshold in np.arange(0.01, 0.3, 0.01):
        pred_classes = (pred_dist >= p_threshold).astype(int)
        target_classes = (target_dist >= p_threshold).astype(int)
        print(f"Threshold: {pred_classes.shape}")
        f1_classwise = f1_score(target_classes, pred_classes, average=None, labels=range(len(REACTION_INDEX)))
        results[f"f1_classwise_{p_threshold}"] = {reaction: f1_classwise[i] for reaction, i in REACTION_INDEX.items()}
        results[f"precision_{p_threshold}"] = precision_score(target_classes, pred_classes, average='macro', zero_division=np.nan)
        results[f"recall_{p_threshold}"] = recall_score(target_classes, pred_classes, average='macro', zero_division=np.nan)
        results[f"f1_macro_{p_threshold}"] = f1_score(target_classes, pred_classes, average='macro', zero_division=np.nan)
        results[f"f1_micro_{p_threshold}"] = f1_score(target_classes, pred_classes, average='micro', zero_division=np.nan)
        results[f"f1_weighted_{p_threshold}"] = f1_score(target_classes, pred_classes, average='weighted', zero_division=np.nan)
        # results[f"accuracy_{p_threshold}"] = accuracy_score(target_classes, pred_classes)
    
    # Use classwise threshold
    pred_classes = np.zeros_like(pred_dist)
    target_classes = np.zeros_like(target_dist)
    for reaction, index in REACTION_INDEX.items():
        pred_classes[:, index] = (pred_dist[:, index] >= classwise_threshold[reaction]).astype(int)
        target_classes[:, index] = (target_dist[:, index] >= classwise_threshold[reaction]).astype(int)
    f1_classwise = f1_score(target_classes, pred_classes, average=None, labels=range(len(REACTION_INDEX)))
    results["f1_classwise_expected_prob_threshold"] = {reaction: f1_classwise[i] for reaction, i in REACTION_INDEX.items()}
    results["f1_macro_expected_prob_threshold"] = f1_score(target_classes, pred_classes, average='macro', zero_division=np.nan)
    results["f1_micro_expected_prob_threshold"] = f1_score(target_classes, pred_classes, average='micro', zero_division=np.nan)
    results["f1_weighted_expected_prob_threshold"] = f1_score(target_classes, pred_classes, average='weighted', zero_division=np.nan)
    
    for top_k in [1, 2, 3, 4, 5]:
        # Filter based on lower p_threshold and then get top_k
        pred_classes = np.argsort(pred_dist, axis=1)[:, -top_k:]
        # Filter aft

        # Convert to one-hot encoding
        pred_classes = np.zeros_like(pred_dist)
        for i in range(pred_classes.shape[0]):
            pred_classes[i, np.argsort(pred_dist[i])[-top_k:]] = 1
            
        # Convert to one-hot encoding
        target_classes = np.zeros_like(target_dist)
        for i in range(target_classes.shape[0]):
            target_classes[i, np.argsort(target_dist[i])[-top_k:]] = 1
        results[f"f1_macro_top_{top_k}"] = f1_score(target_classes, pred_classes, average='macro')
        results[f"f1_micro_top_{top_k}"] = f1_score(target_classes, pred_classes, average='micro')
        results[f"f1_weighted_top_{top_k}"] = f1_score(target_classes, pred_classes, average='weighted')
        results[f"precision_weighted_top_{top_k}"] = precision_score(target_classes, pred_classes, average='weighted')
        results[f"recall_weighted_top_{top_k}"] = recall_score(target_classes, pred_classes, average='weighted')
        results[f"accuracy_top_{top_k}"] = accuracy_score(target_classes, pred_classes)
        results[f"ecc_top_{top_k}"] = compute_eec_multi_label_top_k(pred_dist, target_dist, k=top_k)
            

    for top_p in [0.5, 0.7, 0.9]:  # You can adjust these values
        # Sort pred_dist and target_dist in descending order to get the classes with the highest probability
        sorted_pred_indices = np.argsort(pred_dist, axis=1)[:, ::-1]  # Reverse sorting for descending order
        sorted_target_indices = np.argsort(target_dist, axis=1)[:, ::-1]

        # Get the sorted probabilities
        sorted_pred_values = np.take_along_axis(pred_dist, sorted_pred_indices, axis=1)
        sorted_target_values = np.take_along_axis(target_dist, sorted_target_indices, axis=1)

        # Compute the cumulative probability mass
        cumulative_pred_prob = np.cumsum(sorted_pred_values, axis=1)
        cumulative_target_prob = np.cumsum(sorted_target_values, axis=1)

        # Find the indices where the cumulative probability exceeds the top_p
        pred_mask = cumulative_pred_prob <= top_p
        target_mask = cumulative_target_prob <= top_p

        # Create a target_classes array that is 1 for the selected indices and 0 otherwise
        target_classes = np.zeros_like(target_dist, dtype=int)
        
        # Set target_classes to 1 for the indices selected by cumulative probability mass
        target_classes[np.arange(target_dist.shape[0])[:, None], sorted_target_indices] = target_mask

        # Create a pred_classes array that is 1 for the selected indices and 0 otherwise
        pred_classes = np.zeros_like(pred_dist, dtype=int)
        # Set pred_classes to 1 for the indices selected by cumulative probability mass
        pred_classes[np.arange(pred_dist.shape[0])[:, None], sorted_pred_indices] = pred_mask
        results[f"f1_macro_top_p_{top_p}"] = f1_score(target_classes, pred_classes, average='macro')
        results[f"f1_micro_top_p_{top_p}"] = f1_score(target_classes, pred_classes, average='micro')
        results[f"f1_weighted_top_p_{top_p}"] = f1_score(target_classes, pred_classes, average='weighted')
    

        
    # Ranking metric
    results["mrr"] = compute_mean_reciprocal_rank(pred_dist, target_dist)
    # MAP@k
    for top_k in [1, 2, 3]:
        results[f"map_at_{top_k}"] = compute_mean_average_precision_at_k(pred_dist, target_dist, k=top_k)
    
    # Get the upper bound of probabili
    #results["auc"] = roc_auc_score(target_classes, pred_classes, average=None)

    # Compute metrics for sentiment groupping
    # transform the target distribution to sentiment
    target_sentiment_dist = np.zeros((target_dist.shape[0], 3))
    for i in range(target_dist.shape[0]):
        for j, sentiment in enumerate(SENTIMENT_CLASSES):
            target_sentiment_dist[i, j] = np.sum(target_dist[i, SENTIMENT_2_FINER_GRAINED_INDICES_MAPPING[sentiment]])
    pred_sentiment_dist = np.zeros((pred_dist.shape[0], 3))
    for i in range(pred_dist.shape[0]):
        for j, sentiment in enumerate(SENTIMENT_CLASSES):
            pred_sentiment_dist[i, j] = np.sum(pred_dist[i, SENTIMENT_2_FINER_GRAINED_INDICES_MAPPING[sentiment]])
    
    
    # Compute metrics for sentiment groupping
    pred_sentiment_classes = np.argmax(pred_sentiment_dist, axis=1)
    target_sentiment_classes = np.argmax(target_sentiment_dist, axis=1)
    results["sentiment_accuracy"] = accuracy_score(target_sentiment_classes, pred_sentiment_classes)
    #results["sentiment_ecc"] = compute_eec(pred_sentiment_dist, target_sentiment_dist)
    sentiment_f1_classwise = f1_score(target_sentiment_classes, pred_sentiment_classes, average=None, labels=range(len(SENTIMENT_CLASSES)))
    print(sentiment_f1_classwise)
    results["sentiment_f1_classwise"] = {reaction: sentiment_f1_classwise[i] for i, reaction in enumerate(SENTIMENT_CLASSES)}
    results["sentiment_f1_macro"] = f1_score(target_sentiment_classes, pred_sentiment_classes, average='macro')
    results["sentiment_f1_micro"] = f1_score(target_sentiment_classes, pred_sentiment_classes, average='micro')
    results["sentiment_f1_weighted"] = f1_score(target_sentiment_classes, pred_sentiment_classes, average='weighted')
    results["sentiment_precision_macro"] = precision_score(target_sentiment_classes, pred_sentiment_classes, average='macro')
    results["sentiment_recall_macro"] = recall_score(target_sentiment_classes, pred_sentiment_classes, average='macro')
    results["sentiment_precision_micro"] = precision_score(target_sentiment_classes, pred_sentiment_classes, average='micro')
    results["sentiment_recall_micro"] = recall_score(target_sentiment_classes, pred_sentiment_classes, average='micro')
    results["sentiment_mae_top_1_gt"] = np.mean(np.abs(pred_sentiment_dist[np.arange(len(pred_sentiment_dist)), target_sentiment_classes] - target_sentiment_dist[np.arange(len(target_sentiment_dist)), target_sentiment_classes]))

    for p_threshold in np.arange(0.01, 0.3, 0.005):
        pred_sentiment_classes = (pred_sentiment_dist >= p_threshold).astype(int)
        target_sentiment_classes = (target_sentiment_dist >= p_threshold).astype(int)
        f1_classwise = f1_score(target_sentiment_classes, pred_sentiment_classes, average=None, labels=range(len(SENTIMENT_CLASSES)))
        results[f"sentiment_f1_classwise_{p_threshold}"] = {reaction: f1_classwise[i] for i, reaction in enumerate(SENTIMENT_CLASSES)}
        results[f"sentiment_precision_{p_threshold}"] = precision_score(target_sentiment_classes, pred_sentiment_classes, average='macro', zero_division=np.nan)
        results[f"sentiment_recall_{p_threshold}"] = recall_score(target_sentiment_classes, pred_sentiment_classes, average='macro', zero_division=np.nan)
        results[f"sentiment_f1_macro_{p_threshold}"] = f1_score(target_sentiment_classes, pred_sentiment_classes, average='macro', zero_division=np.nan)
        results[f"sentiment_f1_micro_{p_threshold}"] = f1_score(target_sentiment_classes, pred_sentiment_classes, average='micro', zero_division=np.nan)
        results[f"sentiment_f1_weighted_{p_threshold}"] = f1_score(target_sentiment_classes, pred_sentiment_classes, average='weighted', zero_division=np.nan)

    
    
    return results

    
    



