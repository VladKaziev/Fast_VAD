import torch
from options import *
import numpy as np
from dataset_loader import *
from sklearn.metrics import roc_curve,auc,precision_recall_curve, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings("ignore")

def get_video_scores_and_labels(test_loader, net):
    """
    Получает предсказания на уровне видео и истинные метки из загрузчика.
    Предполагается, что test_loader возвращает (_data, _video_label_batch).
    Предполагается, что net(_data) возвращает скоры сегментов (batch_size, num_segments, 1),
    и мы берем максимум по сегментам для получения видео-уровневого скора.
    """
    video_scores_all = []
    video_labels_all = []

    for _data, _labels_batch in test_loader: # Итерация по загрузчику
        _data = _data.cuda()
        # _labels_batch уже должны быть на CPU, если загрузчик не перемещает их на CUDA
        
        res = net(_data)  # Ожидаемый выход: (batch_size, num_segments, 1) - скоры для каждого сегмента

        # Агрегируем скоры сегментов для получения видео-уровневых скоров
        # Используем максимум по сегментам (dim=1)
        if res.ndim == 3 and res.shape[2] == 1:  # (batch_size, num_segments, 1)
            # Берем максимальный скор по всем сегментам для каждого видео в батче
            video_scores_batch = res.max(dim=1)[0].squeeze(dim=1)  # -> (batch_size)
        elif res.ndim == 2 and res.shape[1] == 1:  # Уже (batch_size, 1) - предсказанный видео-уровневый скор
            video_scores_batch = res.squeeze(dim=1)  # -> (batch_size)
        elif res.ndim == 2 and res.shape[1] > 1: # (batch_size, num_features_possibly_not_segments)
             print(f"Warning: Model output shape is (batch_size, {res.shape[1]}). Taking max over the second dimension for video score.")
             video_scores_batch = res.max(dim=1)[0] # -> (batch_size)
        else:
            print(f"Error: Unexpected prediction shape from model: {res.shape}. Cannot reliably extract video scores.")
            # В случае ошибки, можно вернуть пустые массивы или обработать иначе
            # Для простоты, если один элемент в батче и форма res (1, X), берем res[0,0]
            if res.shape[0] == 1 and res.numel() > 0 :
                 video_scores_batch = res.flatten()[0].unsqueeze(0) # Делаем его (1,)
            else: # Пропускаем этот батч или обрабатываем ошибку
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
    """Вычисляет метрики AUC и AP для видео-уровневых предсказаний и меток."""
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
    
    # Для PR-кривой и AP используются вероятности
    precision_vals, recall_vals, _ = precision_recall_curve(video_labels, video_predictions)
    metrics['AP'] = auc(recall_vals, precision_vals)

    # Для Precision, Recall, F1 нужны бинарные предсказания. Используем порог 0.5.
    # Вы можете захотеть найти оптимальный порог или использовать другой.
    threshold = 0.5
    binary_predictions = (video_predictions >= threshold).astype(int)
    
    # zero_division=0 означает, что если знаменатель 0 (например, нет True Positives), метрика будет 0.
    metrics['Precision'] = precision_score(video_labels, binary_predictions, zero_division=0)
    metrics['Recall'] = recall_score(video_labels, binary_predictions, zero_division=0)
    metrics['F1'] = f1_score(video_labels, binary_predictions, zero_division=0)
    
    return metrics

def test(net, test_loader, test_info, step, model_file=None):
    with torch.no_grad():
        net.eval()
        net.flag = "Test"  # Этот флаг может влиять на поведение модели
        if model_file is not None:
            try:
                # Загружаем модель на то же устройство, где она была сохранена, или на CPU
                net.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))
            except FileNotFoundError:
                print(f"Error: Model file not found at {model_file}")
                # Заполняем test_info пустыми значениями или нулями
                test_info['step'].append(step)
                test_info.setdefault('AUC', []).append(0)
                test_info.setdefault('AP', []).append(0)
                test_info.setdefault('Precision', []).append(0)
                test_info.setdefault('Recall', []).append(0)
                test_info.setdefault('F1', []).append(0)
                return {"AUC": 0, "AP": 0, "Precision": 0, "Recall": 0, "F1": 0} # Возвращаем нулевые метрики
            except Exception as e:
                print(f"Error loading model state_dict: {e}")
                test_info['step'].append(step)
                test_info.setdefault('AUC', []).append(0)
                test_info.setdefault('AP', []).append(0)
                test_info.setdefault('Precision', []).append(0)
                test_info.setdefault('Recall', []).append(0)
                test_info.setdefault('F1', []).append(0)
                return {"AUC": 0, "AP": 0, "Precision": 0, "Recall": 0, "F1": 0}

        # Больше не загружаем frame_gt из файла
        # frame_gt = np.load("frame_label/xd_gt.npy")
        
        video_predictions, video_labels = get_video_scores_and_labels(test_loader, net)

        if video_predictions.size == 0 or video_labels.size == 0:
            print("Warning: No predictions or labels obtained from test_loader. Skipping metrics calculation for this step.")
            current_metrics = {"AUC": 0, "AP": 0, "Precision": 0, "Recall": 0, "F1": 0}
            # if test_info['step'] and len(test_info['AUC']) > 0 : # Это условие не нужно, просто логируем 0
            #      pass
        else:
            current_metrics = get_metrics(video_predictions, video_labels)
        
        test_info['step'].append(step)
        # Гарантируем, что ключи существуют в test_info для всех метрик
        for metric_name in ['AUC', 'AP', 'Precision', 'Recall', 'F1']:
            test_info.setdefault(metric_name, [])

        scaled_metrics_for_return = {}
        for score_name, score_value in current_metrics.items():
            scaled_value = score_value * 100 # Метрики хранятся в процентах
            test_info[score_name].append(scaled_value)
            scaled_metrics_for_return[score_name] = scaled_value
            
        return scaled_metrics_for_return
