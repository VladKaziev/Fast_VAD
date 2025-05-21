import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import Compose
import numpy as np
import os
from PIL import Image
import math
import argparse
from tqdm import tqdm
import av
try:
    from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample
    from pytorchvideo.models import x3d
except ImportError:
    print("PyTorchVideo not found. Please install it: pip install pytorchvideo")
    raise

def load_x3d_model(model_name='x3d_l', device='cuda'):
    print(f"Загрузка модели {model_name}...")
    model = x3d.create_x3d(
        input_channel=3,
        input_clip_length=16,
        input_crop_size=320,
        model_num_class=400,
        norm=torch.nn.BatchNorm3d,
        activation=torch.nn.ReLU,
        head_activation=None
    )
    model = model.to(device)
    model.eval()
    print(f"Модель {model_name} загружена на {device}.")
    return model

def get_spatial_crops(clip_tensor_c_t_h_w, target_crop_size, side_size):
    c, t, h_scaled, w_scaled = clip_tensor_c_t_h_w.shape
    if not (h_scaled >= target_crop_size and w_scaled >= target_crop_size):
        print(f"Warning: Scaled size ({h_scaled}x{w_scaled}) is smaller than target crop ({target_crop_size}x{target_crop_size}). Using center crop or resizing.")
        if h_scaled != target_crop_size or w_scaled != target_crop_size:
             clip_tensor_c_t_h_w = TF.resize(clip_tensor_c_t_h_w, [target_crop_size, target_crop_size], antialias=True)
             h_scaled, w_scaled = target_crop_size, target_crop_size
    crop_coords = [
        (0, 0),
        (0, w_scaled - target_crop_size),
        (h_scaled - target_crop_size, 0),
        (h_scaled - target_crop_size, w_scaled - target_crop_size),
        ((h_scaled - target_crop_size) // 2, (w_scaled - target_crop_size) // 2)
    ]
    cropped_clips = []
    for y_start, x_start in crop_coords:
        y_start = max(0, y_start)
        x_start = max(0, x_start)
        y_end = min(h_scaled, y_start + target_crop_size)
        x_end = min(w_scaled, x_start + target_crop_size)
        crop = clip_tensor_c_t_h_w[..., y_start:y_end, x_start:x_end]
        if crop.shape[-2] != target_crop_size or crop.shape[-1] != target_crop_size:
            crop = TF.resize(crop, [target_crop_size, target_crop_size], antialias=True)
        cropped_clips.append(crop)
    return cropped_clips

def extract_features_for_video(
    video_path,
    model,
    x3d_params,
    num_target_segments,
    device,
    temp_batch_size=10
    ):
    video_fps = 30
    try:
        container = av.open(video_path)
        video_fps = container.streams.video[0].average_rate
        if not video_fps or video_fps == 0:
            print(f"Warning: Could not determine FPS for {video_path}. Using default 30 FPS.")
            video_fps = 30
        total_frames = container.streams.video[0].frames
        if total_frames == 0:
             total_frames = sum(1 for _ in container.decode(video=0))
             container.seek(0)
        if total_frames == 0:
            print(f"Error: Could not read frames from {video_path}. Skipping.")
            return None
    except Exception as e:
        print(f"Error opening or reading video {video_path}: {e}. Skipping.")
        return None
    x3d_num_frames = x3d_params['num_frames']
    x3d_sampling_rate = x3d_params['sampling_rate']
    x3d_crop_size = x3d_params['crop_size']
    x3d_side_size = x3d_params['side_size']
    window_original_frames = x3d_num_frames * x3d_sampling_rate
    mean = torch.tensor([0.45, 0.45, 0.45], dtype=torch.float32).view(3, 1, 1, 1)
    std = torch.tensor([0.225, 0.225, 0.225], dtype=torch.float32).view(3, 1, 1, 1)
    transform_pipeline = Compose([
        T.Lambda(lambda x: x / 255.0),
        T.Normalize(mean=mean, std=std),
        ShortSideScale(size=x3d_side_size),
    ])
    segment_start_original_frames = np.linspace(
        0,
        max(0, total_frames - window_original_frames),
        num_target_segments,
        dtype=int
    )
    video_aggregated_features = []
    model_blocks = list(model.blocks.children())
    feature_extractor = torch.nn.Sequential(*model_blocks[:-1])
    pooling_layer = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
    for start_original_frame_idx in segment_start_original_frames:
        current_window_pil_images = []
        container.seek(int(start_original_frame_idx), stream=container.streams.video[0])
        frames_collected = 0
        for frame in container.decode(video=0):
            if frames_collected < window_original_frames:
                current_window_pil_images.append(frame.to_image())
                frames_collected += 1
            else:
                break
        if frames_collected < window_original_frames:
            if not current_window_pil_images:
                 print(f"Error: Failed to collect any frames for a window in {video_path} at start_idx {start_original_frame_idx}. Skipping this segment.")
                 video_aggregated_features.append(np.zeros((192,), dtype=np.float32) if video_aggregated_features else np.zeros((192,), dtype=np.float32))
                 continue
            print(f"Warning: Not enough frames for a full window in {video_path} at start_idx {start_original_frame_idx}. Padding with last frame.")
            current_window_pil_images.extend([current_window_pil_images[-1]] * (window_original_frames - frames_collected))
        clip_tensor_c_t_h_w = torch.stack([TF.to_tensor(img) for img in current_window_pil_images], dim=1)
        subsampled_clip = UniformTemporalSubsample(x3d_num_frames)(clip_tensor_c_t_h_w)
        transformed_subsampled_clip_original = transform_pipeline(subsampled_clip.clone())
        flipped_subsampled_clip = subsampled_clip.flip(dims=(-1,))
        transformed_subsampled_clip_flipped = transform_pipeline(flipped_subsampled_clip)
        clips_to_process_for_segment = []
        original_spatial_crops = get_spatial_crops(transformed_subsampled_clip_original, x3d_crop_size, x3d_side_size)
        clips_to_process_for_segment.extend(original_spatial_crops)
        flipped_spatial_crops = get_spatial_crops(transformed_subsampled_clip_flipped, x3d_crop_size, x3d_side_size)
        clips_to_process_for_segment.extend(flipped_spatial_crops)
        if not clips_to_process_for_segment:
            print(f"Warning: No clips generated for segment (after 10-crop) starting at {start_original_frame_idx} in {video_path}. Skipping this segment.")
            video_aggregated_features.append(np.zeros((192,), dtype=np.float32) if video_aggregated_features else np.zeros((192,), dtype=np.float32))
            continue
        segment_features_list_tensors = []
        with torch.no_grad():
            for i in range(0, len(clips_to_process_for_segment), temp_batch_size):
                batch_clips_tensor = torch.stack(clips_to_process_for_segment[i:i+temp_batch_size]).to(device)
                features_out = feature_extractor(batch_clips_tensor)
                pooled_features_out = pooling_layer(features_out)
                flattened_features_out = torch.flatten(pooled_features_out, start_dim=1)
                segment_features_list_tensors.append(flattened_features_out)
        if not segment_features_list_tensors:
            print(f"Warning: No features extracted for segment (after model) starting at {start_original_frame_idx} in {video_path}.")
            video_aggregated_features.append(np.zeros((192,), dtype=np.float32) if video_aggregated_features else np.zeros((192,), dtype=np.float32))
            continue
        all_segment_features_tensor = torch.cat(segment_features_list_tensors, dim=0)
        averaged_segment_features_tensor = torch.mean(all_segment_features_tensor, dim=0)
        video_aggregated_features.append(averaged_segment_features_tensor.cpu().numpy())
    container.close()
    if not video_aggregated_features:
        print(f"Error: No features aggregated for {video_path}. Skipping.")
        return None
    if len(video_aggregated_features) != num_target_segments:
        print(f"Warning: Expected {num_target_segments} segments, but got {len(video_aggregated_features)} for {video_path}.")
        if video_aggregated_features:
            last_known_feature = video_aggregated_features[-1]
            feature_dim_actual = last_known_feature.shape[0]
            while len(video_aggregated_features) < num_target_segments:
                video_aggregated_features.append(last_known_feature)
        else:
            actual_feature_dim = 192
            print(f"Critical: No features extracted, padding with zeros to shape ({num_target_segments}, {actual_feature_dim})")
            for _ in range(num_target_segments):
                video_aggregated_features.append(np.zeros((actual_feature_dim,), dtype=np.float32))
    final_features_np = np.array(video_aggregated_features, dtype=np.float32)
    return final_features_np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Извлечение признаков X3D для видео.")
    parser.add_argument('--input_dir', type=str, required=True, help="Директория с видеофайлами.")
    parser.add_argument('--output_dir', type=str, required=True, help="Директория для сохранения .npy файлов с признаками.")
    parser.add_argument('--x3d_model_name', type=str, default='x3d_l', choices=['x3d_xs', 'x3d_s', 'x3d_m', 'x3d_l'], help="Версия модели X3D.")
    parser.add_argument('--num_segments', type=int, default=16, help="Целевое количество сегментов на видео.")
    parser.add_argument('--batch_size_clips', type=int, default=10, help="Размер батча для обработки клипов через модель (влияет на VRAM).")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Устройство для вычислений (cuda/cpu).")
    parser.add_argument('--num_workers', type=int, default=0, help="Количество параллельных процессов для обработки видео (0 - без параллелизма).")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    x3d_params_dict = {
        'x3d_xs': {'num_frames': 4, 'sampling_rate': 12, 'crop_size': 182, 'side_size': 182, 'feature_dim': 2048},
        'x3d_s':  {'num_frames': 13, 'sampling_rate': 6, 'crop_size': 182, 'side_size': 182, 'feature_dim': 2048},
        'x3d_m':  {'num_frames': 16, 'sampling_rate': 5, 'crop_size': 256, 'side_size': 256, 'feature_dim': 2048},
        'x3d_l':  {'num_frames': 16, 'sampling_rate': 5, 'crop_size': 320, 'side_size': 320, 'feature_dim': 2048}
    }
    current_x3d_params = x3d_params_dict[args.x3d_model_name]
    x3d_model = load_x3d_model(args.x3d_model_name, args.device)
    video_files = []
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_files.append(os.path.join(root, file))
    print(f"Найдено {len(video_files)} видео для обработки.")
    for video_path in tqdm(video_files, desc="Обработка видео"):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_filename = f"{video_name}_x3d_{args.x3d_model_name}.npy"
        output_filepath = os.path.join(args.output_dir, output_filename)
        if os.path.exists(output_filepath):
            print(f"Файл {output_filename} уже существует. Пропуск.")
            continue
        print(f"Обработка: {video_path}")
        features = extract_features_for_video(
            video_path,
            x3d_model,
            current_x3d_params,
            args.num_segments,
            args.device,
            args.batch_size_clips
        )
        if features is not None:
            features_with_batch_dim = np.expand_dims(features, axis=0)
            np.save(output_filepath, features_with_batch_dim)
            print(f"Признаки сохранены в: {output_filepath}, форма: {features_with_batch_dim.shape}")
        else:
            print(f"Не удалось извлечь признаки для {video_path}")
    print("Извлечение признаков завершено.") 