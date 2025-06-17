import json
from datetime import datetime
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import json
import numpy as np
from torchvision import models, transforms
from tqdm import tqdm




def convert_time(t):
    dt = datetime.strptime(t, "%H:%M:%S")
    return dt.hour * 3600 + dt.minute * 60 + dt.second

# def calculate_diff_time(start, end):
#     start_seconds = convert_time(start)
#     end_seconds = convert_time(end)
#     diff_seconds = end_seconds - start_seconds
#     return diff_seconds


# with open("labels_json/train_labels.json", "r", encoding='utf-8') as f:
#     train_labels = json.load(f)

# # поиск ошибок в разметке когда время начала больше времени конца
# for video, info in train_labels.items():
#     diff_time = calculate_diff_time(info['start'], info['end'])
#     if diff_time < 0:
#         print(f"{video} - {info['name']}")


class IntroClipDataset(Dataset):
    def __init__(self, video_dir, annotation_path, transform=None, clip_len=16, stride=8):
        self.video_dir = video_dir
        self.clip_len = clip_len
        self.stride = stride  # кадры между клипами
        self.clips = []
        self.labels = []
        self.transform = transform if transform else transforms.Compose([
            transforms.ToTensor(),  # Преобразование в тензор
            transforms.Resize((240, 240)),  # Изменение размера
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])

        with open(annotation_path, "r", encoding='utf-8') as f:
            self.annotations = json.load(f)


        self._prepare_clips()

    def _prepare_clips(self):
    
        for video, info in self.annotations.items():
            video_path = os.path.join(self.video_dir, video, video + ".mp4")
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            intro_start = convert_time(info["start"])
            intro_end = convert_time(info["end"])
            intro_start_frame = int(intro_start * fps)
            intro_end_frame = int(intro_end * fps)

            frames = []
            frame_labels = []
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Пропускаем кадры для достижения нужного frame_rate
                if frame_count % (int(fps)) != 0:
                    frame_count += 1
                    continue

                # Разметка кадра
                label = 1 if intro_start_frame <= frame_count <= intro_end_frame else 0
                
                # Конвертируем BGR в RGB для torchvision
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                frame_labels.append(label)
                
                frame_count += 1


            # создаём клипы по всему видео
            for start in range(0, total_frames - self.clip_len):
                # end = start + self.clip_len
                label = 1 if intro_start_frame <= start < intro_end_frame else 0
                self.clips.append({
                    "video": video_path,
                    "start": start,
                    "label": label,
                    "fps": fps
                })

            cap.release()
            break

    def __len__(self):
        return len(self.clips)

    # def __getitem__(self, idx):
    #     item = self.clips[idx]
    #     cap = cv2.VideoCapture(item["video"])
    #     frames = []

    #     cap.set(cv2.CAP_PROP_POS_FRAMES, item["start"])

    #     # for _ in range(self.clip_len):
    #     #     ret, frame = cap.read()
    #     #     if not ret:
    #     #         break
    #     #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     #     frame = cv2.resize(frame, self.resize)
    #     #     frame = frame / 255.0  # нормализация [0, 1]
    #     #     frames.append(frame)

    #     for i in range(self.clip_len):
    #         # Берем кадр на секунде i: смещение по времени = i секунд
    #         frame_number = item["start"] + int(i * item["fps"])
    #         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    #         ret, frame = cap.read()
    #         if not ret:
    #             break
        
    #         # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         frame = cv2.resize(frame, self.resize)
    #         frame = frame / 255.0  # нормализация [0, 1]
    #         frames.append(frame)

    #     cap.release()

    #     # [T, H, W, C] → [C, T, H, W]
    #     frames = np.stack(frames)  # T, H, W, C
    #     frames = frames.transpose(3, 0, 1, 2)
    #     return torch.tensor(frames, dtype=torch.float32), torch.tensor(item["label"], dtype=torch.long)
    def __getitem__(self, idx):
        sequence_frames = self.clips[idx]
        sequence_labels = self.labels[idx]

        # Применяем трансформации к каждому кадру в последовательности
        processed_sequence = torch.stack([self.transform(frame) for frame in sequence_frames])
        
        return processed_sequence, torch.LongTensor(sequence_labels)


dataset = IntroClipDataset("data_train_short", "labels_json/train_labels.json", clip_len=16)
loader = DataLoader(dataset, batch_size=4, shuffle=False)

# def show_clip(clip_tensor, label=None):
#     """
#     Показывает все кадры в одном клипе на одной строке.
#     clip_tensor: [C, T, H, W] (обычно [3, 16, 112, 112])
#     """
#     clip = clip_tensor.permute(1, 2, 3, 0).numpy()  # [T, H, W, C]
#     num_frames = clip.shape[0]

#     plt.figure(figsize=(num_frames * 5, 5))
#     for i in range(num_frames):
#         plt.subplot(1, num_frames, i + 1)
#         plt.imshow(clip[i])
#         plt.axis('off')
#     title = f"Label: {label}" if label is not None else "Clip"
#     plt.suptitle(title)
#     plt.show()

# for clips, labels in loader:
#     print(clips.shape)  # [B, C, T, H, W]
#     print(labels)
#     for i in range(clips.shape[0]):
#         show_clip(clips[i], labels[i])
#     break

sequence_tensor, labels_tensor = dataset[0] # Берем 10-ю последовательность


def unnormalize_image(tensor):
    """Де-нормализует тензор изображения для визуализации."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    # Переводим тензор в numpy, меняем оси с (C, H, W) на (H, W, C)
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img

# Визуализируем несколько кадров из последовательности
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle("Пример последовательности из датасета", fontsize=16)

for i, ax in enumerate(axes.flatten()):
    frame_index_in_sequence = i * 7 # Берем каждый 7-й кадр, чтобы было разнообразие
    if frame_index_in_sequence >= len(sequence_tensor):
        ax.axis('off')
        continue
    
    frame = unnormalize_image(sequence_tensor[frame_index_in_sequence])
    label = "INTRO" if labels_tensor[frame_index_in_sequence] == 1 else "NOT INTRO"
    
    ax.imshow(frame)
    ax.set_title(f"Кадр {frame_index_in_sequence}\nМетка: {label}", color='green' if label == "INTRO" else 'red')
    ax.axis('off')

plt.show()


















class MyModel(torch.nn.Module):
    def __init__(self, num_classes=2):
        super(MyModel, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.cnn = torch.nn.Sequential(*list(resnet.children())[:-1]) 

        for param in self.cnn.parameters():
            param.requires_grad = False

        self.lstm = torch.nn.LSTM(input_size=2048, hidden_size=512, num_layers=2, batch_first=True, bidirectional=True)

        self.fc = torch.nn.Linear(512 * 2, num_classes)

    def forward(self, x):
        batch_size, channels, time, height, width = x.size()
        x = x.view(batch_size * time, channels, height, width)
        x = self.cnn(x)
        x = x.view(batch_size, time, -1)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x 


# model = MyModel()


def train(model, dataloader, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        model.train()
        for clips, labels in tqdm(dataloader):
            clips, labels = clips.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(clips)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")
    
    torch.save(model.state_dict(), "intro_model.pth")
    return model



