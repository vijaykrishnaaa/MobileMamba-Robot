"""
inference.py — VideoAnomalyDetector class for temporal clip inference.

Wraps the full inference pipeline: frame sampling, 8-frame clip aggregation, 
sliding-window anomaly scoring, GradCAM-based ROI extraction, and event logging.
"""

import os
import sys
import cv2
import yaml
import torch
import numpy as np
from pathlib import Path
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(_PROJECT_ROOT))

from src.model import AnomalyMobileMamba, CLASS_NAMES
from src.roi import extract_roi_bbox, draw_roi, crop_roi
from src.logger import EventLogger

class VideoAnomalyDetector:
    def __init__(self, model_path: str, config_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model = AnomalyMobileMamba(num_classes=6)
        ckpt = torch.load(model_path, map_location='cpu')
        state_dict = ckpt.get('model', ckpt)
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(self.device).eval()
        print(f"[Detector] Model loaded from {model_path} on {self.device}")

        img_size = 256
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(int(img_size / 0.875)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ])
        self.logger = EventLogger(os.path.join(_PROJECT_ROOT, 'outputs'))

    def preprocess(self, frame_bgr: np.ndarray) -> torch.Tensor:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self.transform(frame_rgb).unsqueeze(0).to(self.device)

    def extract_roi(self, frame_bgr: np.ndarray, crime_idx: int) -> dict | None:
        tensor = self.preprocess(frame_bgr)
        # GradCAM still uses single frame path in model.py
        cam, _, _ = self.model.get_gradcam(tensor, class_idx=crime_idx)
        return extract_roi_bbox(frame_bgr, cam, threshold=self.config.get('roi_threshold', 0.4))

    def analyze_video(self, video_path: str, save_events: bool = True) -> dict:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): raise FileNotFoundError(f"Cannot open: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        video_name = Path(video_path).stem
        frame_sample_rate = self.config.get('frame_sample_rate', 10)
        window_size = self.config.get('window_size', 16)
        anomaly_threshold = self.config.get('anomaly_threshold', 0.55)
        CLIP_SIZE = self.config.get('temporal_size', 8)

        print(f"[Detector] Analysing: {video_path} (Sample: 1/{frame_sample_rate})")

        frame_num = 0
        window_scores = []
        events = []
        in_event = False
        event_start = None
        event_frames = []
        frame_buffer = []

        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_num += 1

            if frame_num % frame_sample_rate != 0: continue

            tensor = self.preprocess(frame)
            frame_buffer.append(tensor.squeeze(0)) # [3, 256, 256]

            if len(frame_buffer) < CLIP_SIZE: continue

            # Temporal inference: send 8 frames to model
            clip = torch.stack(frame_buffer[-CLIP_SIZE:], dim=0).unsqueeze(0)
            with torch.no_grad():
                probs = torch.softmax(self.model(clip), dim=1)[0]

            anomaly_score = 1.0 - probs[0].item()
            crime_idx = probs.argmax().item()
            window_scores.append((anomaly_score, crime_idx, frame, frame_num, probs))

            if len(window_scores) >= window_size:
                avg_score = sum(s[0] for s in window_scores) / len(window_scores)
                
                # Dominant crime calc
                all_probs = torch.stack([s[4] for s in window_scores])
                mean_probs = all_probs.mean(dim=0)
                mean_probs[0] = 0.0 # Ignore Normal
                dominant_crime = int(mean_probs.argmax())

                if avg_score > anomaly_threshold:
                    if not in_event:
                        in_event = True
                        event_start = window_scores[0][3] / fps
                    event_frames.append(window_scores[len(window_scores) // 2])
                else:
                    if in_event:
                        event_end = window_scores[-1][3] / fps
                        mid_frame = event_frames[-1]
                        roi = self.extract_roi(mid_frame[2], dominant_crime)
                        event = {
                            'start_time': round(event_start, 2),
                            'end_time': round(event_end, 2),
                            'crime_type': CLASS_NAMES[dominant_crime],
                            'confidence': round(avg_score, 4),
                            'roi_bbox': roi,
                        }
                        events.append(event)
                        if save_events:
                            self.logger.log_event(event, video_name)
                        in_event = False
                        event_frames = []
                window_scores = window_scores[window_size // 2:]

        cap.release()
        return {'video': video_path, 'events': events, 'frame_count': frame_num}
