import torch
from options import *
import numpy as np
from dataset_loader import *
from sklearn.metrics import roc_curve,auc,precision_recall_curve, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings("ignore")

def get_video_scores_and_labels(test_loader, net):
    video_scores_all = []
    video_labels_all = []
    for _data, _labels_batch in test_loader:
        _data = _data.cuda()
        res = net(_data)
        if res.ndim == 3 and res.shape[2] == 1:
            video_scores_batch = res.max(dim=1)[0].squeeze(dim=1)
        elif res.ndim == 2 and res.shape[1] == 1:
            video_scores_batch = res.squeeze(dim=1)
        elif res.ndim == 2 and res.shape[1] > 1:
             print(f"Warning: Model output shape is (batch_size, {res.shape[1]}). Taking max over the second dimension for video score.")
             video_scores_batch = res.max(dim=1)[0]
        else:
            print(f"Error: Unexpected prediction shape from model: {res.shape}. Cannot reliably extract video scores.")
            if res.shape[0] == 1 and res.numel() > 0 :
                 video_scores_batch = res.flatten()[0].unsqueeze(0)
            else:
                 print(f"Skipping batch due to unhandled prediction shape: {res.shape}")
                 continue
        video_scores_all.append(video_scores_batch.cpu().detach().numpy())
        video_labels_all.append(_labels_batch.cpu().numpy())
    if not video_scores_all or not video_labels_all:
        return np.array([]), np.array([])
    video_scores_all = np.concatenate(video_scores_all)
    video_labels_all = np.concatenate(video_labels_all)
    return video_scores_all, video_labels_all

def get_metrics(video_predictions, video_labels):
    metrics = {}
    if video_labels.size == 0 or video_predictions.size == 0:
        print("Warning: Empty labels or predictions, returning zero for all metrics.")
        metrics['AUC'] = 0.0
        metrics['AP'] = 0.0
        metrics['Precision'] = 0.0
        metrics['Recall'] = 0.0
        metrics['F1'] = 0.0
        return metrics
    fpr, tpr, _ = roc_curve(video_labels, video_predictions)
    metrics['AUC'] = auc(fpr, tpr)
    precision_vals, recall_vals, _ = precision_recall_curve(video_labels, video_predictions)
    metrics['AP'] = auc(recall_vals, precision_vals)
    threshold = 0.5
    binary_predictions = (video_predictions >= threshold).astype(int)
    metrics['Precision'] = precision_score(video_labels, binary_predictions, zero_division=0)
    metrics['Recall'] = recall_score(video_labels, binary_predictions, zero_division=0)
    metrics['F1'] = f1_score(video_labels, binary_predictions, zero_division=0)
    return metrics

def test(net, test_loader, test_info, step, model_file=None):
    with torch.no_grad():
        net.eval()
        net.flag = "Test"
        if model_file is not None:
            try:
                net.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))
            except FileNotFoundError:
                print(f"Error: Model file not found at {model_file}")
                test_info['step'].append(step)
                test_info.setdefault('AUC', []).append(0)
                test_info.setdefault('AP', []).append(0)
                test_info.setdefault('Precision', []).append(0)
                test_info.setdefault('Recall', []).append(0)
                test_info.setdefault('F1', []).append(0)
                return {"AUC": 0, "AP": 0, "Precision": 0, "Recall": 0, "F1": 0}
            except Exception as e:
                print(f"Error loading model state_dict: {e}")
                test_info['step'].append(step)
                test_info.setdefault('AUC', []).append(0)
                test_info.setdefault('AP', []).append(0)
                test_info.setdefault('Precision', []).append(0)
                test_info.setdefault('Recall', []).append(0)
                test_info.setdefault('F1', []).append(0)
                return {"AUC": 0, "AP": 0, "Precision": 0, "Recall": 0, "F1": 0}
        video_predictions, video_labels = get_video_scores_and_labels(test_loader, net)
        if video_predictions.size == 0 or video_labels.size == 0:
            print("Warning: No predictions or labels obtained from test_loader. Skipping metrics calculation for this step.")
            current_metrics = {"AUC": 0, "AP": 0, "Precision": 0, "Recall": 0, "F1": 0}
        else:
            current_metrics = get_metrics(video_predictions, video_labels)
        test_info['step'].append(step)
        for metric_name in ['AUC', 'AP', 'Precision', 'Recall', 'F1']:
            test_info.setdefault(metric_name, [])
        scaled_metrics_for_return = {}
        for score_name, score_value in current_metrics.items():
            scaled_value = score_value * 100
            test_info[score_name].append(scaled_value)
            scaled_metrics_for_return[score_name] = scaled_value
        return scaled_metrics_for_return
