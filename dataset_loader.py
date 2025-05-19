import torch
import torch.utils.data as data
import os
import numpy as np
import utils 

class XDVideo(data.DataLoader):
    def __init__(self, root_dir, list_file_name, mode, num_segments, len_feature, seed=-1, is_normal_filter=None):
        if seed >= 0:
            utils.set_seed(seed)
        self.data_path = root_dir # Например, 'root_dir'
        self.mode = mode
        self.num_segments = num_segments
        self.len_feature = len_feature
        
        # self.feature_path больше не используется отдельно, т.к. vid_name теперь относительный путь
        split_path = os.path.join("list", list_file_name)
        
        vid_list_all = []
        try:
            with open(split_path, 'r', encoding="utf-8") as split_file:
                for line in split_file:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        vid_list_all.append((parts[0], int(parts[1]))) # (относительный_путь, метка)
                    elif len(parts) == 1: # Для старых файлов XD, где метка не указана
                        vid_list_all.append((parts[0], -1)) # Используем -1 как плейсхолдер для метки
        except FileNotFoundError:
            print(f"Error: List file not found at {split_path}")
            self.vid_list = []
            return

        self.vid_list = []
        if self.mode == "Train":
            if is_normal_filter is True:
                self.vid_list = [item for item in vid_list_all if item[1] == 0]
            elif is_normal_filter is False:
                self.vid_list = [item for item in vid_list_all if item[1] == 1]
            else: # Если is_normal_filter не задан, но режим Train, берем все (если нужно смешанное обучение из одного файла)
                self.vid_list = vid_list_all # или можно вызвать ошибку, если всегда ожидается фильтр
        else: # Для 'Test' или других режимов
            self.vid_list = vid_list_all
        
        if not self.vid_list and vid_list_all: # Если список пуст из-за фильтра, но исходный файл был не пуст
            print(f"Warning: Video list became empty after filtering for mode={self.mode}, list_file_name={list_file_name}, is_normal_filter={is_normal_filter}")
        elif not vid_list_all and not self.vid_list : # Если исходный файл был пуст или не найден
             pass # Ошибка уже выведена выше
        
    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):
        data, label = self.get_data(index)
        return data, label

    def get_data(self, index):
        vid_path_relative, label_from_list = self.vid_list[index]
        
        # Определяем метку: если в списке была -1 (старый формат XD), используем старую логику, иначе берем из списка
        if label_from_list == -1: # Обработка для старых XD файлов без явной метки
            label = 0
            if "_label_A" not in vid_path_relative: # Старая логика определения метки для XD
                label = 1
        else:
            label = label_from_list

        video_feature_full_path = os.path.join(self.data_path, vid_path_relative)
        
        try:
            video_feature = np.load(video_feature_full_path).astype(np.float32)
        except FileNotFoundError:
            print(f"Error: Feature file not found at {video_feature_full_path}")
            return torch.zeros((self.num_segments, self.len_feature), dtype=torch.float32), torch.tensor(-1, dtype=torch.long)
        except Exception as e:
            print(f"Error loading feature file {video_feature_full_path}: {e}")
            return torch.zeros((self.num_segments, self.len_feature), dtype=torch.float32), torch.tensor(-1, dtype=torch.long)

        # Проверка на пустой файл после загрузки
        if video_feature.size == 0: # Проверяем общее количество элементов
            print(f"Warning: Empty feature file (or all zeros) {video_feature_full_path}")
            return torch.zeros((self.num_segments, self.len_feature), dtype=torch.float32), torch.tensor(label, dtype=torch.long)

        # Если признаки имеют форму (1, num_segments, len_feature), извлекаем нужную часть
        if video_feature.ndim == 3 and video_feature.shape[0] == 1:
            video_feature = video_feature[0] # Теперь video_feature имеет форму (num_segments, len_feature)
        
        # Проверка соответствия размерностей после возможного извлечения
        if video_feature.shape[0] != self.num_segments or video_feature.shape[1] != self.len_feature:
            # Если режим 'Train' и количество сегментов не совпадает, применяем семплирование/усреднение
            # (Эта логика взята из оригинального кода и может быть нужна, если некоторые файлы все же имеют другую длительность)
            if self.mode == "Train" and video_feature.shape[0] > 0 :
                # Проверяем, что len_feature совпадает, иначе семплирование по времени не поможет
                if video_feature.shape[1] != self.len_feature:
                    print(f"Error: Feature dimension mismatch in {video_feature_full_path}. Expected {self.len_feature}, got {video_feature.shape[1]}. Cannot proceed.")
                    return torch.zeros((self.num_segments, self.len_feature), dtype=torch.float32), torch.tensor(label, dtype=torch.long)

                print(f"Warning: Mismatch in segment count for {video_feature_full_path}. Expected {self.num_segments}, got {video_feature.shape[0]}. Resampling for Train mode.")
                processed_feature = np.zeros((self.num_segments, self.len_feature), dtype=np.float32)
                sample_indices = np.linspace(0, video_feature.shape[0], self.num_segments + 1, dtype=np.uint16)
                for i in range(len(sample_indices) - 1):
                    start_idx = sample_indices[i]
                    end_idx = sample_indices[i+1]
                    if start_idx == end_idx:
                        actual_idx = min(start_idx, video_feature.shape[0] - 1)
                        processed_feature[i,:] = video_feature[actual_idx,:]
                    else:
                        actual_end_idx = min(end_idx, video_feature.shape[0])
                        segment_features = video_feature[start_idx:actual_end_idx,:]
                        if segment_features.shape[0] > 0:
                             processed_feature[i,:] = segment_features.mean(0)
                        elif video_feature.shape[0] > 0:
                             processed_feature[i,:] = video_feature[-1,:]
                video_feature = processed_feature
            elif video_feature.shape[0] != self.num_segments : # Для Test режима или если в Train len_feature не совпал
                 print(f"Error: Dimension mismatch in {video_feature_full_path}. Expected ({self.num_segments}, {self.len_feature}), got {video_feature.shape}. Cannot proceed.")
                 return torch.zeros((self.num_segments, self.len_feature), dtype=torch.float32), torch.tensor(label, dtype=torch.long)


        # Если мы дошли сюда, video_feature должен иметь форму (self.num_segments, self.len_feature)
        return video_feature, torch.tensor(label, dtype=torch.long)
