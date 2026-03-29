"""
logger.py — Event and timestamp logging for anomaly detection.

Writes per-event JSON files and a per-session CSV summary log.
"""

import json
import csv
import os
from datetime import datetime
from pathlib import Path


class EventLogger:
    """
    Logs detected anomaly events to disk.

    Output locations:
        outputs/events/  — one JSON file per event
        outputs/logs/    — per-session CSV summary
    """

    def __init__(self, output_dir: str = 'outputs'):
        self.events_dir = os.path.join(output_dir, 'events')
        self.logs_dir = os.path.join(output_dir, 'logs')
        self.rois_dir = os.path.join(output_dir, 'rois')
        os.makedirs(self.events_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.rois_dir, exist_ok=True)

        # Session CSV
        session_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = os.path.join(self.logs_dir, f'session_{session_ts}.csv')
        self._init_csv()
        self.event_count = 0

    def _init_csv(self):
        """Initialise the CSV file with headers."""
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'video_name', 'event_index', 'crime_type', 'confidence',
                'start_time', 'end_time', 'roi_x', 'roi_y', 'roi_w', 'roi_h',
            ])

    def log_event(self, event: dict, video_name: str) -> str:
        """
        Write an event to JSON and append to the session CSV.

        Args:
            event: dict with keys: start_time, end_time, crime_type,
                   confidence, roi_bbox
            video_name: source video name

        Returns:
            path to the saved JSON file
        """
        self.event_count += 1
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{video_name}_{event["crime_type"]}_{ts}.json'
        json_path = os.path.join(self.events_dir, filename)

        # Enrich event data
        event_record = {
            'video': video_name,
            'start_time': event.get('start_time'),
            'end_time': event.get('end_time'),
            'duration_seconds': round(
                event.get('end_time', 0) - event.get('start_time', 0), 2
            ),
            'crime_type': event.get('crime_type'),
            'confidence': round(event.get('confidence', 0), 4),
            'roi_bbox': event.get('roi_bbox'),
            'frame_sample_rate': event.get('frame_sample_rate', 10),
            'model': 'MobileMamba-B1-AnomalyV1',
        }

        # Save JSON
        with open(json_path, 'w') as f:
            json.dump(event_record, f, indent=2)

        # Append to CSV
        bbox = event.get('roi_bbox') or {}
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                video_name,
                self.event_count,
                event.get('crime_type', ''),
                round(event.get('confidence', 0), 4),
                event.get('start_time', ''),
                event.get('end_time', ''),
                bbox.get('x', ''),
                bbox.get('y', ''),
                bbox.get('w', ''),
                bbox.get('h', ''),
            ])

        return json_path

    def save_roi_crop(self, crop, video_name: str, crime_type: str) -> str | None:
        """Save a cropped ROI image to outputs/rois/."""
        if crop is None:
            return None
        import cv2
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{video_name}_{crime_type}_{ts}.jpg'
        path = os.path.join(self.rois_dir, filename)
        cv2.imwrite(path, crop)
        return path
