# Fast_VAD

## Описание
Fast_VAD — это система для обнаружения аномалий в видео с помощью weakly-supervised anomaly detection (WSAD) и модели Performer. Проект реализован на Python с использованием PyTorch и поддерживает обучение и тестирование на пользовательских датасетах.

## Структура проекта
- `dataset_loader.py` — загрузка и подготовка датасета (класс Video_dataset)
- `main_performer.py` — основной скрипт для обучения и тестирования Performer
- `models/` — модели для WSAD
- `train.py`, `test.py` — функции для обучения и тестирования
- `utils.py` — вспомогательные функции
- `list/` — списки файлов для обучения и теста

## Установка зависимостей
Рекомендуется использовать виртуальное окружение:

```bash
python -m venv venv
source venv/bin/activate  # или venv\Scripts\activate для Windows
pip install -r requirements.txt
```

Минимальные зависимости:
- torch
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- wandb
- tqdm

## Быстрый старт
1. Подготовьте списки файлов для обучения и теста в папке `list/` (пример формата: `relative_path label`).
2. Запустите обучение Performer:

```bash
python main_performer.py --root_dir <путь_к_фичам> --list_file_train <train_list.txt> --list_file_test <test_list.txt> --num_segments 32 --len_feature 1024 --batch_size 16 --num_workers 4 --version exp1
```

Параметры можно посмотреть в `options.py`.

## Пример структуры данных
```
root_dir/
    video1.npy
    video2.npy
    ...
list/
    train_list.txt
    test_list.txt
```

## Визуализация и логирование
- Используется [Weights & Biases (wandb)](https://wandb.ai/) для логирования метрик. 