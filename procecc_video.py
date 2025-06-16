import json
from datetime import datetime
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import json
import random
import numpy as np



def convert_time(t):
    dt = datetime.strptime(t, "%H:%M:%S")
    return dt.hour * 3600 + dt.minute * 60 + dt.second

def calculate_diff_time(start, end):
    start_seconds = convert_time(start)
    end_seconds = convert_time(end)
    diff_seconds = end_seconds - start_seconds
    return diff_seconds


with open("labels_json/train_labels.json", "r", encoding='utf-8') as f:
    train_labels = json.load(f)

# поиск ошибок в разметке когда время начала больше времени конца
for video, info in train_labels.items():
    diff_time = calculate_diff_time(info['start'], info['end'])
    if diff_time < 0:
        print(f"{video} - {info['name']}")


class IntroClipDataset(Dataset):
    def __init__(self, video_dir, annotation_path, clip_len=16, resize=(480, 480), stride=8):
        self.video_dir = video_dir
        self.clip_len = clip_len
        self.resize = resize
        self.stride = stride  # кадры между клипами
        self.clips = []

        with open(annotation_path, "r", encoding='utf-8') as f:
            self.annotations = json.load(f)


        self._prepare_clips()

    def _prepare_clips(self):
    
        for video, info in self.annotations.items():
            if video != "-220020068_456255339":
                continue

            print(info['name'])
            video_path = os.path.join(self.video_dir, video, video + ".mp4")
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            intro_start = convert_time(info["start"])
            intro_end = convert_time(info["end"])
            intro_start_f = int(intro_start * fps)
            intro_end_f = int(intro_end * fps)

            # создаём клипы по всему видео
            for start in range(0, total_frames - self.clip_len, self.stride):
                end = start + self.clip_len
                label = 1 if intro_start_f <= start < intro_end_f else 0
                self.clips.append({
                    "video": video_path,
                    "start": start,
                    "label": label,
                    "fps": fps
                })

            cap.release()
            if video == "-220020068_456255339":
                break

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        item = self.clips[idx]
        cap = cv2.VideoCapture(item["video"])
        frames = []

        cap.set(cv2.CAP_PROP_POS_FRAMES, item["start"])

        for _ in range(self.clip_len):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, self.resize)
            frame = frame / 255.0  # нормализация [0, 1]
            frames.append(frame)

        cap.release()

        # [T, H, W, C] → [C, T, H, W]
        frames = np.stack(frames)  # T, H, W, C
        frames = frames.transpose(3, 0, 1, 2)
        return torch.tensor(frames, dtype=torch.float32), torch.tensor(item["label"], dtype=torch.long)



dataset = IntroClipDataset("data_train_short", "labels_json/train_labels.json", clip_len=16)
loader = DataLoader(dataset, batch_size=4, shuffle=False)

def show_clip(clip_tensor, label=None):
    """
    Показывает все кадры в одном клипе на одной строке.
    clip_tensor: [C, T, H, W] (обычно [3, 16, 112, 112])
    """
    clip = clip_tensor.permute(1, 2, 3, 0).numpy()  # [T, H, W, C]
    num_frames = clip.shape[0]

    plt.figure(figsize=(num_frames*5, 5))
    for i in range(num_frames):
        plt.subplot(1, num_frames, i + 1)
        plt.imshow(clip[i])
        plt.axis('off')
    title = f"Label: {label}" if label is not None else "Clip"
    plt.suptitle(title)
    plt.show()

for clips, labels in loader:
    print(clips.shape[0])  # [B, C, T, H, W]
    print(labels)
    for i in range(clips.shape[0]):
        show_clip(clips[i], labels[i])
    
    

    

