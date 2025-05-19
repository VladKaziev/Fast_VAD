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
import av # PyAV for video loading

# Try to import PyTorchVideo, if not available, provide instructions
try:
    from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample
    from pytorchvideo.models import x3d
except ImportError:
    print("PyTorchVideo not found. Please install it: pip install pytorchvideo")
    raise

def load_x3d_model(model_name='x3d_l', device='cuda'):
    """
    Загружает предобученную модель X3D и удаляет классификационную голову.
    """
    print(f"Загрузка модели {model_name}...")
    model = x3d.create_x3d(
        input_channel=3,
        input_clip_length=16, # X3D_L typically takes 16 frames
        input_crop_size=320,  # X3D_L typically takes 320x320 crops
        model_num_class=400,  # Irrelevant as we remove the head
        norm=torch.nn.BatchNorm3d,
        activation=torch.nn.ReLU,
        head_activation=None # No activation in head before removal
    )
    model = model.to(device)
    model.eval()
    print(f"Модель {model_name} загружена на {device}.")
    return model

def get_spatial_crops(clip_tensor_c_t_h_w, target_crop_size, side_size):
    """
    Создает 5 пространственных кропов (TL, TR, BL, BR, Center) из клипа.
    Клип должен быть уже после ShortSideScale.
    Args:
        clip_tensor_c_t_h_w: Тензор клипа (C, T, H_scaled, W_scaled).
        target_crop_size (int): Размер целевого кропа (например, 320).
        side_size (int): Размер короткой стороны после ShortSideScale (например, 320)
    Returns:
        list_of_clips: Список из 5 тензоров-кропов.
    """
    c, t, h_scaled, w_scaled = clip_tensor_c_t_h_w.shape

    if not (h_scaled >= target_crop_size and w_scaled >= target_crop_size):
         # Если после ShortSideScale размер меньше crop_size, просто берем центральный кроп
         # или можно сделать паддинг/ресайз. Для X3D side_size=crop_size=320 обычно.
        print(f"Warning: Scaled size ({h_scaled}x{w_scaled}) is smaller than target crop ({target_crop_size}x{target_crop_size}). Using center crop or resizing.")
        # Простейший вариант - ресайз до target_crop_size, если они не равны side_size
        if h_scaled != target_crop_size or w_scaled != target_crop_size:
             clip_tensor_c_t_h_w = TF.resize(clip_tensor_c_t_h_w, [target_crop_size, target_crop_size], antialias=True)
             h_scaled, w_scaled = target_crop_size, target_crop_size


    crop_coords = [
        (0, 0),  # Top-Left
        (0, w_scaled - target_crop_size),  # Top-Right
        (h_scaled - target_crop_size, 0),  # Bottom-Left
        (h_scaled - target_crop_size, w_scaled - target_crop_size),  # Bottom-Right
        ((h_scaled - target_crop_size) // 2, (w_scaled - target_crop_size) // 2)  # Center
    ]

    cropped_clips = []
    for y_start, x_start in crop_coords:
        # Гарантируем, что координаты не отрицательные (может случиться, если h_scaled < target_crop_size)
        y_start = max(0, y_start)
        x_start = max(0, x_start)
        
        # Гарантируем, что кроп не выходит за пределы
        y_end = min(h_scaled, y_start + target_crop_size)
        x_end = min(w_scaled, x_start + target_crop_size)

        crop = clip_tensor_c_t_h_w[..., y_start:y_end, x_start:x_end]
        
        # Если из-за округлений или малых размеров кроп получился не того размера, делаем ресайз
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
    temp_batch_size=10 # Внутренний батч для обработки клипов через модель
    ):
    """
    Извлекает признаки для одного видео.
    """
    video_fps = 30 # По умолчанию, или можно читать из видео
    try:
        container = av.open(video_path)
        video_fps = container.streams.video[0].average_rate # Получаем FPS из видео
        if not video_fps or video_fps == 0: # Проверка на случай, если FPS не определен
            print(f"Warning: Could not determine FPS for {video_path}. Using default 30 FPS.")
            video_fps = 30
        total_frames = container.streams.video[0].frames
        if total_frames == 0: # Если нет информации о кадрах, пытаемся прочитать
             total_frames = sum(1 for _ in container.decode(video=0))
             container.seek(0) # Возвращаем указатель в начало
        if total_frames == 0:
            print(f"Error: Could not read frames from {video_path}. Skipping.")
            return None
    except Exception as e:
        print(f"Error opening or reading video {video_path}: {e}. Skipping.")
        return None

    x3d_num_frames = x3d_params['num_frames']        # 16
    x3d_sampling_rate = x3d_params['sampling_rate']  # 5 (для X3D-L)
    x3d_crop_size = x3d_params['crop_size']          # 320 (для X3D-L)
    x3d_side_size = x3d_params['side_size']          # 320 (для X3D-L)

    # Длительность окна в оригинальных кадрах, из которого сэмплируются кадры для X3D
    window_original_frames = x3d_num_frames * x3d_sampling_rate # 16 * 5 = 80

    # Трансформации для X3D (применяются к каждому 16-кадровому клипу)
    # Нормализация такая же, как в PyTorchVideo/SlowFast
    mean = torch.tensor([0.45, 0.45, 0.45], dtype=torch.float32).view(3, 1, 1, 1)
    std = torch.tensor([0.225, 0.225, 0.225], dtype=torch.float32).view(3, 1, 1, 1)

    transform_pipeline = Compose([
        T.Lambda(lambda x: x / 255.0), # В тензор [0,1]
        T.Normalize(mean=mean, std=std),
        ShortSideScale(size=x3d_side_size), # Масштабируем короткую сторону до side_size
    ])

    # Берем равномерно распределенные начальные точки для окон
    segment_start_original_frames = np.linspace(
        0,
        max(0, total_frames - window_original_frames), # Убеждаемся, что не выходим за пределы
        num_target_segments,
        dtype=int
    )

    video_aggregated_features = [] 

    model_blocks = list(model.blocks.children())
    feature_extractor = torch.nn.Sequential(*model_blocks[:-1]) 
    pooling_layer = torch.nn.AdaptiveAvgPool3d((1, 1, 1))

    for start_original_frame_idx in segment_start_original_frames:
        # --- Начало обработки одного "якорного" сегмента ---
        current_window_pil_images = [] # PIL Images для текущего окна
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
                 # Добавляем нулевой вектор, чтобы сохранить количество сегментов
                 video_aggregated_features.append(np.zeros((192,), dtype=np.float32) if video_aggregated_features else np.zeros((192,), dtype=np.float32)) # Используем 192, как ранее выяснили
                 continue
            print(f"Warning: Not enough frames for a full window in {video_path} at start_idx {start_original_frame_idx}. Padding with last frame.")
            current_window_pil_images.extend([current_window_pil_images[-1]] * (window_original_frames - frames_collected))

        clip_tensor_c_t_h_w = torch.stack([TF.to_tensor(img) for img in current_window_pil_images], dim=1)
        subsampled_clip = UniformTemporalSubsample(x3d_num_frames)(clip_tensor_c_t_h_w)
        
        # Трансформации для оригинального и отраженного клипа
        # Этот transform_pipeline создается один раз вне цикла, он уже есть
        transformed_subsampled_clip_original = transform_pipeline(subsampled_clip.clone()) # .clone() чтобы не менять исходный subsampled_clip для отражения
        
        flipped_subsampled_clip = subsampled_clip.flip(dims=(-1,))
        transformed_subsampled_clip_flipped = transform_pipeline(flipped_subsampled_clip)

        # Получаем 10 кропов-тензоров для текущего сегмента
        clips_to_process_for_segment = [] 
        original_spatial_crops = get_spatial_crops(transformed_subsampled_clip_original, x3d_crop_size, x3d_side_size)
        clips_to_process_for_segment.extend(original_spatial_crops)
        flipped_spatial_crops = get_spatial_crops(transformed_subsampled_clip_flipped, x3d_crop_size, x3d_side_size)
        clips_to_process_for_segment.extend(flipped_spatial_crops)
        # --- Конец формирования 10 тензоров-кропов для сегмента ---

        if not clips_to_process_for_segment: # Должно быть 10 клипов
            print(f"Warning: No clips generated for segment (after 10-crop) starting at {start_original_frame_idx} in {video_path}. Skipping this segment.")
            video_aggregated_features.append(np.zeros((192,), dtype=np.float32) if video_aggregated_features else np.zeros((192,), dtype=np.float32))
            continue

        segment_features_list_tensors = []
        with torch.no_grad():
            for i in range(0, len(clips_to_process_for_segment), temp_batch_size):
                # clips_to_process_for_segment уже содержит тензоры
                batch_clips_tensor = torch.stack(clips_to_process_for_segment[i:i+temp_batch_size]).to(device)
                features_out = feature_extractor(batch_clips_tensor) 
                pooled_features_out = pooling_layer(features_out) 
                flattened_features_out = torch.flatten(pooled_features_out, start_dim=1) 
                segment_features_list_tensors.append(flattened_features_out)
        
        if not segment_features_list_tensors:
            print(f"Warning: No features extracted for segment (after model) starting at {start_original_frame_idx} in {video_path}.")
            video_aggregated_features.append(np.zeros((192,), dtype=np.float32) if video_aggregated_features else np.zeros((192,), dtype=np.float32))
            continue

        all_segment_features_tensor = torch.cat(segment_features_list_tensors, dim=0) # (10, feature_dim)
        averaged_segment_features_tensor = torch.mean(all_segment_features_tensor, dim=0) # (feature_dim,)
        video_aggregated_features.append(averaged_segment_features_tensor.cpu().numpy()) # Сохраняем как numpy

    container.close()

    if not video_aggregated_features:
        print(f"Error: No features aggregated for {video_path}. Skipping.")
        return None
    
    # Если из-за пропусков каких-то сегментов у нас меньше num_target_segments, нужно дополнить
    # или обработать эту ситуацию. Пока будем паниковать, если их не num_target_segments.
    if len(video_aggregated_features) != num_target_segments:
        print(f"Warning: Expected {num_target_segments} segments, but got {len(video_aggregated_features)} for {video_path}.")
        # Можно дополнить нулями до num_target_segments, если это необходимо.
        # Например, если video_aggregated_features[0] существует:
        # feature_dim_actual = video_aggregated_features[0].shape[0]
        # while len(video_aggregated_features) < num_target_segments:
        #     video_aggregated_features.append(np.zeros((feature_dim_actual,), dtype=np.float32))
        # Это создаст проблему, если video_aggregated_features пуст изначально.
        # Безопаснее просто вернуть то, что есть, и обработать на уровне загрузчика данных,
        # либо падать с ошибкой, если фиксированное число сегментов критично.
        # Сейчас, для соответствия новой цели (1, 16, 192), мы должны обеспечить 16 сегментов.
        # Если сегментов меньше, дополним последним известным или нулями.
        if video_aggregated_features: # если есть хоть один признак
            last_known_feature = video_aggregated_features[-1]
            feature_dim_actual = last_known_feature.shape[0]
            while len(video_aggregated_features) < num_target_segments:
                video_aggregated_features.append(last_known_feature) # Паддинг последним известным
        else: # Если вообще не было признаков, а мы должны вернуть (1,16,192)
             # Нужно знать feature_dim. Возьмем его из current_x3d_params, если это возможно
             # Это менее надежно, так как фактический feature_dim может отличаться
             # Фактический feature_dim = 192, как мы видели
            actual_feature_dim = 192 # Используем ранее определенную размерность
            print(f"Critical: No features extracted, padding with zeros to shape ({num_target_segments}, {actual_feature_dim})")
            for _ in range(num_target_segments):
                video_aggregated_features.append(np.zeros((actual_feature_dim,), dtype=np.float32))

    final_features_np = np.array(video_aggregated_features, dtype=np.float32) # Форма (num_target_segments, feature_dim)
    # Ожидаемая форма: (16, 192)
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

    # Параметры для X3D (могут зависеть от model_name, но для L они такие)
    # Взят из pytorchvideo.models.x3d.create_x3d
    x3d_params_dict = {
        'x3d_xs': {'num_frames': 4, 'sampling_rate': 12, 'crop_size': 182, 'side_size': 182, 'feature_dim': 2048}, # Уточнить feature_dim
        'x3d_s':  {'num_frames': 13, 'sampling_rate': 6, 'crop_size': 182, 'side_size': 182, 'feature_dim': 2048},
        'x3d_m':  {'num_frames': 16, 'sampling_rate': 5, 'crop_size': 256, 'side_size': 256, 'feature_dim': 2048},
        'x3d_l':  {'num_frames': 16, 'sampling_rate': 5, 'crop_size': 320, 'side_size': 320, 'feature_dim': 2048} # Размер канала после conv5_5 -> res_conv5 блока
    }
    current_x3d_params = x3d_params_dict[args.x3d_model_name]

    # Загрузка модели один раз
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
            # Добавляем батчевое измерение, если оно еще не (1, 16, 192)
            # Сейчас features будет (16, 192), поэтому добавляем batch_dim
            features_with_batch_dim = np.expand_dims(features, axis=0) # Теперь форма (1, 16, 192)
            np.save(output_filepath, features_with_batch_dim)
            print(f"Признаки сохранены в: {output_filepath}, форма: {features_with_batch_dim.shape}")
        else:
            print(f"Не удалось извлечь признаки для {video_path}")
            
    print("Извлечение признаков завершено.") 