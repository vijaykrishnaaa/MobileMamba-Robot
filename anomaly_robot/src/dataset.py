"""
dataset.py — PyTorch Dataset for UCF-Crime frame/clip classification.

Supports two modes:
  • temporal_size=1 (default): loads individual frames
  • temporal_size>1: loads clips of T consecutive frames from the same video

Loads frames from data/frames/{split}/{class}/{video_name}/frame_XXXX.jpg
"""

import os
import random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

CLASS_NAMES = ['Normal', 'Assault', 'Robbery', 'Shooting', 'Fighting', 'Abuse']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

def get_transforms(split: str, img_size: int = 256):
    if split == 'train':
        return transforms.Compose([
            transforms.Resize(int(img_size / 0.875)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomRotation(degrees=10),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(img_size / 0.875)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ])

class CrimeFrameDataset(Dataset):
    def __init__(self, root_dir: str, split: str = 'train',
                 img_size: int = 256, temporal_size: int = 1):
        super().__init__()
        self.root_dir = Path(root_dir) / split
        self.split = split
        self.temporal_size = temporal_size
        self.transform = get_transforms(split, img_size)
        self.samples = []
        self.class_counts = {name: 0 for name in CLASS_NAMES}

        if temporal_size <= 1:
            for cls_name in CLASS_NAMES:
                cls_dir = self.root_dir / cls_name
                if not cls_dir.exists(): continue
                label = CLASS_TO_IDX[cls_name]
                for img_path in sorted(cls_dir.rglob('*.jpg')):
                    self.samples.append((str(img_path), label))
                    self.class_counts[cls_name] += 1
            print(f"[{split.upper()}] Loaded {len(self.samples)} frames")
        else:
            for cls_name in CLASS_NAMES:
                cls_dir = self.root_dir / cls_name
                if not cls_dir.exists(): continue
                label = CLASS_TO_IDX[cls_name]
                for vid_dir in sorted(cls_dir.iterdir()):
                    if vid_dir.is_dir():
                        frames = sorted(vid_dir.glob('*.jpg'))
                        if len(frames) >= temporal_size:
                            self.samples.append((frames, label))
                            self.class_counts[cls_name] += 1
            print(f"[{split.upper()}] Loaded {len(self.samples)} clips (T={temporal_size})")

    def __len__(self):
        return len(self.samples)

    @property
    def labels(self):
        return [s[1] for s in self.samples]

    def __getitem__(self, idx):
        # Safety for stale notebook objects (avoids AttributeError)
        t_size = getattr(self, 'temporal_size', 1)
        if t_size <= 1:
            img_path, label = self.samples[idx]
            image = Image.open(img_path).convert('RGB')
            if self.transform: image = self.transform(image)
            return image, label
        else:
            frames, label = self.samples[idx]
            n = len(frames)
            start = random.randint(0, max(0, n - t_size))
            clip = []
            for i in range(t_size):
                img = Image.open(str(frames[start + i])).convert('RGB')
                if self.transform: img = self.transform(img)
                clip.append(img)
            return torch.stack(clip, dim=0), label

    def get_class_weights(self):
        counts = [self.class_counts[name] for name in CLASS_NAMES]
        total = sum(counts)
        return torch.tensor([total / (6 * max(c, 1)) for c in counts], dtype=torch.float32)
