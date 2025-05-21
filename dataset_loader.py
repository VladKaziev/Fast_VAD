import torch
import torch.utils.data as data
import os
import numpy as np
import utils 

class Video_dataset(data.DataLoader):
    def __init__(self, root_dir, list_file_name, mode, num_segments, len_feature, seed=-1, is_normal_filter=None):
        if seed >= 0:
            utils.set_seed(seed)
        self.data_path = root_dir 
        self.mode = mode
        self.num_segments = num_segments
        self.len_feature = len_feature
        split_path = os.path.join("list", list_file_name)
        vid_list_all = []
        try:
            with open(split_path, 'r', encoding="utf-8") as split_file:
                for line in split_file:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        vid_list_all.append((parts[0], int(parts[1]))) 
                    elif len(parts) == 1: 
                        vid_list_all.append((parts[0], -1)) 
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
            else: 
                self.vid_list = vid_list_all 
        else: 
            self.vid_list = vid_list_all
        if not self.vid_list and vid_list_all: 
            print(f"Warning: Video list became empty after filtering for mode={self.mode}, list_file_name={list_file_name}, is_normal_filter={is_normal_filter}")
        elif not vid_list_all and not self.vid_list : 
             pass 
    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):
        data, label = self.get_data(index)
        return data, label

    def get_data(self, index):
        vid_path_relative, label_from_list = self.vid_list[index]
        if label_from_list == -1: 
            label = 0
            if "_label_A" not in vid_path_relative: 
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
        if video_feature.size == 0: 
            print(f"Warning: Empty feature file (or all zeros) {video_feature_full_path}")
            return torch.zeros((self.num_segments, self.len_feature), dtype=torch.float32), torch.tensor(label, dtype=torch.long)
        if video_feature.ndim == 3 and video_feature.shape[0] == 1:
            video_feature = video_feature[0] 
        if video_feature.shape[0] != self.num_segments or video_feature.shape[1] != self.len_feature:
            if self.mode == "Train" and video_feature.shape[0] > 0 :
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
            elif video_feature.shape[0] != self.num_segments : 
                 print(f"Error: Dimension mismatch in {video_feature_full_path}. Expected ({self.num_segments}, {self.len_feature}), got {video_feature.shape}. Cannot proceed.")
                 return torch.zeros((self.num_segments, self.len_feature), dtype=torch.float32), torch.tensor(label, dtype=torch.long)
        return video_feature, torch.tensor(label, dtype=torch.long)
