import numpy as np
import torch
from libs.datasets import to_frame_wise
from sklearn.metrics import auc
from typing import Tuple

def identity_error(
        segments, 
        video_length,
        pred_segment_action_prob,
        pred_segment_action,
        threshold = 0.5
    ):
    """Predict frame-wise error labels based on action probabilities and threshold.
    
    Args:
        segments: Action segment boundaries.
        video_length: Total length of the video in frames.
        pred_segment_action_prob: Predicted action probability for each segment.
        pred_segment_action: Predicted action class for each segment.
        threshold: Probability threshold to classify as normal (default: 0.5).
        
    Returns:
        Frame-wise error predictions where -1 indicates error and 1 indicates normal.
    """
    pred_segment_error = torch.full_like(pred_segment_action_prob, -1, dtype=torch.long)
    # leave all the background segments as normal since they don't produce action effect
    pred_segment_error[pred_segment_action == 0] = 1
    pred_segment_error[pred_segment_action_prob >= threshold] = 1

    pred_frame_error = \
        to_frame_wise(
            segments, 
            pred_segment_error.numpy(), 
            scores=None, 
            length=video_length
        )
    return pred_frame_error

def error_acc(pred: torch.Tensor, gt: torch.Tensor, gt_error: torch.Tensor) -> Tuple[int, int]:
    """
    Calculate the number of correctly detected error segments and the total number of segments.
    
    Parameters:
    - pred (torch.Tensor): Predicted frame-wise error labels. 
                           -1 indicates an error, 1 indicates no error.
    - gt (torch.Tensor): Ground truth frame-wise action labels (multi-class).
    - gt_error (torch.Tensor): Ground truth frame-wise error labels.
                               -1 indicates an error, 1 indicates no error.
    
    Returns:
    - Tuple[int, int]: A tuple containing:
        - num_correct (int): Number of correctly detected segments.
        - num_total (int): Total number of segments evaluated.
    """

    num_correct = 0
    num_total = 0
    
    gt_shifted = torch.cat((gt[:1], gt[:-1]))
    segment_changes = gt != gt_shifted 
    segment_changes[0] = True   
    segment_start_indices = torch.nonzero(segment_changes, as_tuple=False).squeeze(1)
    
    segment_end_indices = torch.cat((segment_start_indices[1:], torch.tensor([pred.numel()], device=pred.device)))

    for start, end in zip(segment_start_indices.tolist(), segment_end_indices.tolist()):
        segment_gt_error = gt_error[start].item()
        segment_pred = pred[start:end]
        error_count = torch.sum(segment_pred == -1).item()
        segment_length = end - start
        non_error_count = segment_length - error_count
        if segment_gt_error == -1:
            if error_count > non_error_count:
                num_correct += 1
        else:
            if error_count < non_error_count:
                num_correct += 1
        num_total += 1
    
    return num_correct, num_total


def roc_curve(all_preds, all_gts):
    """Calculate accuracy, TPR, FPR, and precision from predictions and ground truth.
    
    Args:
        all_preds: Predicted error labels where -1 is error and 1 is normal.
        all_gts: Ground truth error labels where -1 is error and 1 is normal.
        
    Returns:
        Tuple containing accuracy, TPR (recall), FPR, and precision.
    """
    # fpr = fp / (fp + tn)
    all_gt_normal = all_preds[all_gts == 1] # get predicted non-error items
    fp_tn = len(all_gt_normal) # number of total non-error items in the ground truth
    fp = len(all_gt_normal[all_gt_normal == -1]) # get FP
    
    # tpr = tp / (tp + fn)
    all_gt_error = all_preds[all_gts == -1] # get predicted error items
    tp_fn = len(all_gt_error) # number of total error items in the ground truth
    tp = len(all_gt_error[all_gt_error == -1]) # get TP

    # Calculate True Negatives
    tn = len(all_gt_normal[all_gt_normal == 1])  # get TN
    
    # Calculate False Negatives
    fn = len(all_gt_error[all_gt_error == 1])  # get FN

    # acc
    acc = torch.eq(torch.LongTensor(all_gts), torch.LongTensor(all_preds)).sum() / len(all_gts)

    # Calculate precision = tp / (tp + fp)
    if tp + fp == 0:
        precision = 1.0 if tp == 0 else 0.0
    else:
        precision = tp / (tp + fp)

    # Calculate recall (same as tpr) = tp / (tp + fn)
    if tp_fn == 0:
        if tp == 0:
            tpr = 1
        else:
            tpr = 0
    else:
        tpr = tp / tp_fn

    # Calculate fpr = fp / (fp + tn)
    if fp_tn == 0:
        if fp == 0:
            fpr = 0 
        else:
            fpr = 1
    else:
        fpr = fp / fp_tn
    
    return acc, tpr, fpr, precision


def evaluate_ed(results, threshold=0.5):
    """
    Compute overall error detection accuracy and micro framewise metrics
    from the provided results.

    Parameters:
        results (dict): A dictionary of results for each video.
        threshold (float): Threshold value for identity_error computation.

    Returns:
        dict: A dictionary with overall accuracy and micro metrics (acc, tpr, fpr, precision).
    """
    total_correct = 0
    total = 0
    preds = None
    gts = None

    for video_id, result in results.items():
        # Retrieve and prepare data for computation.
        segments = result['segments'].clone().numpy()
        pred_segment_action = result['segment_action_pred'].clone()
        segment_action_similarity = result['segment_action_sim'].clone()

        # Process ground truth error labels.
        gt_frame_error = result['frame_error_label'].clone()
        # Convert all error types to -1, and normal frames to 1.
        gt_frame_error[gt_frame_error > 0] = -1
        gt_frame_error[gt_frame_error == 0] = 1

        # Compute predicted frame error using the identity_error function.
        pred_frame_error = identity_error(
            segments, 
            gt_frame_error.size(0),
            segment_action_similarity, 
            pred_segment_action,
            threshold
        )
        
        # For overall accuracy, retrieve the frame action labels.
        gt_frame_action = result['frame_action_label'].clone()
        num_correct, num_total = error_acc(
            pred_frame_error, 
            gt_frame_action, 
            gt_frame_error
        )
        total_correct += num_correct
        total += num_total
        
        # For micro metrics, accumulate predictions and ground truths.
        if preds is None:
            preds = pred_frame_error
            gts = gt_frame_error
        else:
            preds = torch.cat((preds, pred_frame_error), dim=0)
            gts = torch.cat((gts, gt_frame_error), dim=0)
    
    # Compute overall accuracy.
    err_acc = total_correct / total if total > 0 else 0
    micro_acc, micro_tpr, micro_fpr, micro_precision = roc_curve(preds, gts)
    
    # Package the results in a dictionary.
    output = {
        'error_acc': err_acc,
        'micro': {
            'acc': micro_acc,
            'tpr': micro_tpr,
            'fpr': micro_fpr,
            'precision': micro_precision
        }
    }
    return output


def calculate_micro_auc(fprs, tprs):
    """Calculate micro-averaged AUC from FPR and TPR values.
    
    Args:
        fprs: List or array of false positive rates.
        tprs: List or array of true positive rates.
        
    Returns:
        Micro-averaged AUC score.
    """
    fprs = np.array(fprs).flatten()
    tprs = np.array(tprs).flatten()

    fpr_sorted = np.sort(fprs)
    tpr_sorted = np.sort(tprs)
    fpr_sorted = np.concatenate([fpr_sorted, np.array([1.0])], axis=0)
    tpr_sorted = np.concatenate([tpr_sorted, np.array([1.0])], axis=0)
    
    micro_auc_value = auc(fpr_sorted, tpr_sorted)
    return micro_auc_value

# -----------------------------
#  Single-video metric functions
# -----------------------------

def compute_accuracy_single_video(y_true, y_pred):
    """Frame-level accuracy for a single video.
    
    Args:
        y_true: Ground truth frame-wise labels.
        y_pred: Predicted frame-wise labels.
        
    Returns:
        Tuple containing number of correct predictions and total number of frames.
    """
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    return np.sum(y_true == y_pred), len(y_true)

def compute_mIoU_single_video(y_true, y_pred):
    """Mean Intersection over Union (mIoU) for a single video.
    
    Args:
        y_true: Ground truth frame-wise labels.
        y_pred: Predicted frame-wise labels.
        
    Returns:
        Mean IoU averaged across all classes present in the video.
    """

    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    
    all_classes = np.unique(np.concatenate((y_true, y_pred)))
    intersection_sum, valid_classes = 0.0, 0
    
    for cls in all_classes:
        intersection = np.sum((y_true == cls) & (y_pred == cls))
        union        = np.sum((y_true == cls) | (y_pred == cls))
        if union > 0:
            intersection_sum += intersection / union
            valid_classes += 1
    
    if valid_classes == 0:
        return 0.0
    return intersection_sum / valid_classes

def framewise_to_segments(frame_labels):
    """Convert frame-wise labels into a list of segments.
    
    Args:
        frame_labels: Array of frame-wise class labels.
        
    Returns:
        List of tuples (label, start_index, end_index) representing contiguous segments.
    """
    if len(frame_labels) == 0:
        return []
    
    segments = []
    start = 0
    current_label = frame_labels[0]
    
    for i in range(1, len(frame_labels)):
        if frame_labels[i] != current_label:
            segments.append((current_label, start, i - 1))
            current_label = frame_labels[i]
            start = i
    # Append the final segment
    segments.append((current_label, start, len(frame_labels) - 1))
    
    return segments

def compute_edit_score_single_video(y_true, y_pred):
    """Segment-level Edit Score using Levenshtein edit distance on label sequences.
    
    Args:
        y_true: Ground truth frame-wise labels.
        y_pred: Predicted frame-wise labels.
        
    Returns:
        Normalized edit score in range [0, 1], where 1 is perfect match.
    """
    true_segments = framewise_to_segments(y_true)
    pred_segments = framewise_to_segments(y_pred)
    
    # Convert segments -> label sequences
    true_seq = [s[0] for s in true_segments]
    pred_seq = [s[0] for s in pred_segments]
    
    len_t = len(true_seq)
    len_p = len(pred_seq)
    if len_t == 0 and len_p == 0:
        return 1.0  # Both empty => perfect
    
    # DP table for edit distance
    D = np.zeros((len_p+1, len_t+1), dtype=int)
    for i in range(len_p+1):
        D[i, 0] = i
    for j in range(len_t+1):
        D[0, j] = j
    
    for i in range(1, len_p+1):
        for j in range(1, len_t+1):
            if pred_seq[i-1] == true_seq[j-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = 1 + min(D[i-1, j],    # delete
                                  D[i, j-1],    # insert
                                  D[i-1, j-1])  # substitute
    
    edit_distance = D[len_p, len_t]
    max_len = max(len_p, len_t)
    return 1.0 - edit_distance / max_len

def compute_f1_0_5_single_video(y_true, y_pred):
    """Segment-level F1@0.5 for a single video.
    
    Args:
        y_true: Ground truth frame-wise labels.
        y_pred: Predicted frame-wise labels.
        
    Returns:
        F1 score where a predicted segment matches ground truth if labels match and IoU >= 0.5.
    """
    gt_segments = framewise_to_segments(y_true)
    pred_segments = framewise_to_segments(y_pred)

    def segment_to_frame_set(seg):
        # seg = (label, start, end)
        return set(range(seg[1], seg[2] + 1))

    matched_gt = set()
    tp = 0
    for pseg_idx, pseg in enumerate(pred_segments):
        p_label, _, _ = pseg
        p_frames = segment_to_frame_set(pseg)
        best_iou = 0.0
        best_gt_idx = None
        for g_idx, gseg in enumerate(gt_segments):
            if g_idx in matched_gt:
                continue
            g_label, _, _ = gseg
            if g_label != p_label:
                continue
            g_frames = segment_to_frame_set(gseg)
            
            inter = len(p_frames.intersection(g_frames))
            uni = len(p_frames.union(g_frames))
            iou = inter / uni if uni > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx
        
        if best_iou >= 0.5 and best_gt_idx is not None:
            tp += 1
            matched_gt.add(best_gt_idx)
    
    fp = len(pred_segments) - tp
    fn = len(gt_segments) - len(matched_gt)
    
    if (tp + fp) == 0 or (tp + fn) == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def evaluate_seg(list_of_y_true, list_of_y_pred):
    """Evaluate segmentation metrics across multiple videos.
    
    Args:
        list_of_y_true: List of ground truth frame-wise label arrays, one per video.
        list_of_y_pred: List of predicted frame-wise label arrays, one per video.
        
    Returns:
        Dictionary containing 'Accuracy' (micro-averaged), 'mIoU', 'Edit', and 'F1@0.5' (macro-averaged).
    """
    # Check lengths
    assert len(list_of_y_true) == len(list_of_y_pred), "Mismatch in number of videos."
    
    # For micro accuracy
    total_correct = 0
    total_frames = 0
    
    # For macro metrics
    ious = []
    edits = []
    f1s = []
    
    # -- Loop over each video --
    for y_true, y_pred in zip(list_of_y_true, list_of_y_pred):
        # 1) Accumulate micro-accuracy stats
        correct_count, frame_count = compute_accuracy_single_video(y_true, y_pred)
        total_correct += correct_count
        total_frames += frame_count
        
        # 2) Compute per-video mIoU, Edit, F1@0.5
        ious.append(compute_mIoU_single_video(y_true, y_pred))
        edits.append(compute_edit_score_single_video(y_true, y_pred))
        f1s.append(compute_f1_0_5_single_video(y_true, y_pred))
    
    # Compute final micro accuracy
    micro_accuracy = total_correct / total_frames if total_frames > 0 else 0.0
    
    # Compute final macro average of the other metrics
    mIoU = np.mean(ious)
    edit = np.mean(edits)
    f1_05 = np.mean(f1s)
    
    return {
        "Accuracy": micro_accuracy,
        "mIoU": mIoU,
        "Edit": edit,
        "F1@0.5": f1_05
    }

def topk_accuracy(target, output, topk=(1,), name=None):
    """Computes the accuracy over the k top predictions for the specified values of k.
    
    Args:
        target: Ground truth labels tensor of shape (batch_size,).
        output: Model output logits tensor of shape (batch_size, num_classes).
        topk: Tuple of k values to compute top-k accuracy for (default: (1,)).
        name: Optional name prefix for the accuracy keys in returned dictionary.
        
    Returns:
        Dictionary with keys '{name}_top{k}_acc' containing top-k accuracy percentages for each k.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        # Dictionary to store results for each k
        accuracies = {}
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            accuracies[f"{name}_top{k}_acc"] = correct_k.mul_(100.0 / batch_size)
        return accuracies