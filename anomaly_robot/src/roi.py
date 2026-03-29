"""
roi.py — Region of Interest extraction using GradCAM heatmaps.

Uses the GradCAM output from AnomalyMobileMamba to compute a bounding box
around the anomaly region in the frame.
"""

import cv2
import numpy as np
import torch


def extract_roi_bbox(frame_bgr: np.ndarray, cam_tensor: torch.Tensor,
                     threshold: float = 0.4) -> dict | None:
    """
    Compute a bounding box from a GradCAM heatmap.

    Args:
        frame_bgr: original BGR frame (H, W, 3)
        cam_tensor: GradCAM activation map (1, 1, h, w)
        threshold: activation cutoff for binarisation

    Returns:
        dict with keys {x, y, w, h} or None if no region found
    """
    h, w = frame_bgr.shape[:2]
    cam_np = cam_tensor.squeeze().cpu().numpy()
    cam_resized = cv2.resize(cam_np, (w, h))

    # Threshold the heatmap
    binary = (cam_resized > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(largest)
    return {'x': int(x), 'y': int(y), 'w': int(bw), 'h': int(bh)}


def draw_roi(frame_bgr: np.ndarray, bbox: dict | None,
             label: str, confidence: float) -> np.ndarray:
    """
    Draw a bounding box and label on the frame.

    Args:
        frame_bgr: BGR frame to annotate
        bbox: dict {x, y, w, h} or None
        label: crime type string
        confidence: detection confidence [0, 1]

    Returns:
        annotated frame (copy)
    """
    frame = frame_bgr.copy()
    if bbox is None:
        return frame

    x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    text = f'{label} {confidence * 100:.1f}%'
    cv2.putText(frame, text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return frame


def crop_roi(frame_bgr: np.ndarray, bbox: dict | None) -> np.ndarray | None:
    """Crop the frame to the ROI bounding box."""
    if bbox is None:
        return None
    x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
    return frame_bgr[y:y + h, x:x + w].copy()


def overlay_heatmap(frame_bgr: np.ndarray, cam_tensor: torch.Tensor,
                    alpha: float = 0.5) -> np.ndarray:
    """Overlay GradCAM heatmap on the frame for visualisation."""
    h, w = frame_bgr.shape[:2]
    cam_np = cam_tensor.squeeze().cpu().numpy()
    cam_resized = cv2.resize(cam_np, (w, h))
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8),
                                cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame_bgr, 1 - alpha, heatmap, alpha, 0)
    return overlay
