import pdb
import numpy as np
import torch.utils.data as data
import utils
import time
import wandb
import os
import torch

from options import parse_args # Используем существующие options

from train import train # Используем существующий train.py
from losses import LossComputer # Используем существующий LossComputer
from test import test # Используем существующий test.py
from models.wsad_performer import WSADPerformer # Импортируем новую модель

from dataset_loader import Video_dataset # Используем существующий dataset_loader
from tqdm import tqdm

localtime = time.localtime()
time_ymd = time.strftime("%Y-%m-%d", localtime)

if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        pdb.set_trace()

    # Модифицируем пути для логов и моделей, чтобы не перезаписывать старые
    run_version = args.version + "_performer"
    args.log_path = os.path.join(args.log_path, time_ymd, 'xd', run_version)
    args.model_path = os.path.join(args.model_path, time_ymd, 'xd', run_version)
    
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    wandb.init(
        project="BN-WVAD-Performer", # Можно изменить имя проекта для wandb
        name=run_version,
        config=vars(args), # Логируем все аргументы
        settings=wandb.Settings(code_dir=os.path.dirname(os.path.abspath(__file__))),
        save_code=True,
        mode="offline" if getattr(args, 'wandb_offline', True) else "online"
    )

    # Определяем worker_init_fn здесь, как в main.py
    worker_init_fn = None
    if args.seed >= 0:
        utils.set_seed(args.seed)
        # Более корректное определение worker_init_fn для DataLoader
        worker_init_fn = lambda worker_id: np.random.seed(args.seed + worker_id)
    
    # Используем WSADPerformer
    net = WSADPerformer(args.len_feature, flag="Train", args=args)
    net = net.cuda()

    # Загрузка данных (аналогично main.py)
    normal_dataset = Video_dataset(root_dir=args.root_dir, list_file_name=args.list_file_train, mode='Train',
                             num_segments=args.num_segments, len_feature=args.len_feature, is_normal_filter=True, seed=args.seed)
    if len(normal_dataset) == 0:
        actual_train_list_path = os.path.join('list', args.list_file_train) # Упрощено для единообразия
        print(f"Error: Normal training dataset is empty. Check list file: '{actual_train_list_path}' and --root_dir: '{args.root_dir}'.")
        exit(1)
    normal_train_loader = data.DataLoader(normal_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=True, num_workers=args.num_workers,
                                          worker_init_fn=worker_init_fn, drop_last=True)
    if len(normal_train_loader) == 0:
        print(f"Error: normal_train_loader is empty (0 batches). Samples: {len(normal_dataset)}, batch_size: {args.batch_size}. Adjust batch_size or add samples.")
        exit(1)

    abnormal_dataset = Video_dataset(root_dir=args.root_dir, list_file_name=args.list_file_train, mode='Train',
                               num_segments=args.num_segments, len_feature=args.len_feature, is_normal_filter=False, seed=args.seed)
    if len(abnormal_dataset) == 0:
        actual_train_list_path = os.path.join('list', args.list_file_train)
        print(f"Error: Abnormal training dataset is empty. Check list file: '{actual_train_list_path}' and --root_dir: '{args.root_dir}'.")
        exit(1)
    abnormal_train_loader = data.DataLoader(abnormal_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=True, num_workers=args.num_workers,
                                            worker_init_fn=worker_init_fn, drop_last=True)
    if len(abnormal_train_loader) == 0:
        print(f"Error: abnormal_train_loader is empty (0 batches). Samples: {len(abnormal_dataset)}, batch_size: {args.batch_size}. Adjust batch_size or add samples.")
        exit(1)
        
    test_dataset = Video_dataset(root_dir=args.root_dir, list_file_name=args.list_file_test, mode='Test',
                           num_segments=args.num_segments, len_feature=args.len_feature, seed=args.seed)
    if len(test_dataset) == 0:
        actual_test_list_path = os.path.join('list', args.list_file_test)
        print(f"Warning: Test dataset is empty. Check list file: '{actual_test_list_path}' and --root_dir: '{args.root_dir}'. Testing might be skipped.")
        if args.num_iters == 0: exit(1)
             
    test_loader = data.DataLoader(test_dataset,
                                  batch_size=args.batch_size, 
                                  shuffle=False, num_workers=args.num_workers,
                                  worker_init_fn=worker_init_fn, drop_last=False)
    if len(test_dataset) > 0 and len(test_loader) == 0:
        print(f"Warning: test_loader is empty (0 batches) despite test_dataset having {len(test_dataset)} samples. Adjust batch_size.")

    # test_info и best_scores остаются такими же
    test_info = {'step': [], 'AUC': [], 'AP': [], 'Precision': [], 'Recall': [], 'F1': []}
    criterion = LossComputer()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr[0],
                                 betas=(0.9, 0.999), weight_decay=args.weight_decay)

    best_scores = {
        'best_AUC': -1, 'best_AP': -1, 'best_Precision': -1,
        'best_Recall': -1, 'best_F1': -1,
    }

    # Начальное тестирование
    if len(test_loader) > 0:
        metric = test(net, test_loader, test_info, 0) # Убираем args, если test его не ожидает
    else:
        print("Warning: test_loader is empty. Initial test metrics will be all zeros.")
        metric = {"AUC": 0, "AP": 0, "Precision": 0, "Recall": 0, "F1": 0}
        test_info['step'].append(0)
        for m_name in metric.keys(): test_info[m_name].append(0)
        for k in best_scores.keys(): best_scores[k] = 0

    wandb.log({**metric, **best_scores}, step=0)

    # Цикл обучения
    for step in tqdm(range(1, args.num_iters + 1), total=args.num_iters, dynamic_ncols=True):
        if step > 1 and args.lr[step - 1] != args.lr[step - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr[step - 1]
        
        if (step - 1) % len(normal_train_loader) == 0:
            normal_loader_iter = iter(normal_train_loader)
        if (step - 1) % len(abnormal_train_loader) == 0:
            abnormal_loader_iter = iter(abnormal_train_loader)
        
        losses = train(net, normal_loader_iter, abnormal_loader_iter, optimizer, criterion) # Убираем args
        log_data = {**losses}

        if step % args.plot_freq == 0 and step > 0:
            current_metrics = {}
            if len(test_loader) > 0:
                current_metrics = test(net, test_loader, test_info, step) # Убираем args, если test его не ожидает
            else:
                current_metrics = {"AUC": 0, "AP": 0, "Precision": 0, "Recall": 0, "F1": 0}
                test_info['step'].append(step)
                for m_name in current_metrics.keys(): test_info[m_name].append(0)
            
            log_data.update(current_metrics)

            # Обновление best_scores и сохранение модели
            # Сохраняем по AP, как в оригинальном main.py
            if test_info.get("AP") and len(test_info["AP"]) > 0 and test_info["AP"][-1] > best_scores['best_AP']:
                best_scores['best_AP'] = test_info["AP"][-1] # Обновляем лучший AP явно
                # Обновляем и другие лучшие метрики, соответствующие этому лучшему AP
                for n_metric in ['AUC', 'Precision', 'Recall', 'F1']:
                    if test_info.get(n_metric) and len(test_info[n_metric]) > 0:
                         best_scores['best_' + n_metric] = test_info[n_metric][-1]
                
                utils.save_best_record(test_info, os.path.join(args.log_path, f"xd_performer_best_record_seed{args.seed}.txt"))
                torch.save(net.state_dict(), os.path.join(args.model_path, f"xd_performer_best_seed{args.seed}.pkl"))
            
            # Обновляем best_scores для всех метрик независимо (если какая-то другая метрика стала лучше, не только AP)
            for n, v in current_metrics.items():
                best_name = 'best_' + n
                if best_name in best_scores:
                    if v > best_scores[best_name]:
                        best_scores[best_name] = v
            log_data.update(best_scores)        
        
        wandb.log(log_data, step=step)

    # Сохранение финальной модели
    torch.save(net.state_dict(), os.path.join(args.model_path, f"xd_performer_final_seed{args.seed}.pkl"))
    wandb.finish()
    print(f"Training finished. Logs and models saved in: {args.log_path} and {args.model_path}") 