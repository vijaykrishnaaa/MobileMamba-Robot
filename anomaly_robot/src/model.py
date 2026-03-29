"""
model.py — AnomalyMobileMamba wrapper with temporal clip support.

Wraps the MobileMamba-B1 backbone with:
  • A 6-class classification head (Normal + 5 crime types)
  • Temporal aggregation — can process multi-frame clips [B, T, C, H, W]
  • GradCAM hooks on the last backbone stage for ROI extraction
  • A clean predict() / get_gradcam() interface
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Ensure the MobileMamba repo root is on sys.path ────────────────────────── #
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from model.mobilemamba.mobilemamba import MobileMamba, CFG_MobileMamba_B1


# ──────────────────────── Configuration ────────────────────────────────────── #
NUM_CLASSES = 6
CLASS_NAMES = ['Normal', 'Assault', 'Robbery', 'Shooting', 'Fighting', 'Abuse']


class AnomalyMobileMamba(nn.Module):
    """
    MobileMamba-B1 adapted for 6-class crime classification.

    Supports two input modes:
      • Single frame:  [B, 3, 256, 256]     → standard classification
      • Temporal clip:  [B, T, 3, 256, 256]  → per-frame features averaged,
        giving the model motion context across T frames.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained_path: str = None):
        super().__init__()

        # ── Instantiate backbone via the official config dict ──────────────── #
        self.backbone = MobileMamba(num_classes=num_classes, **CFG_MobileMamba_B1)

        # ── Replace the BN_Linear head with a stronger 2-layer head ──────── #
        in_features = CFG_MobileMamba_B1['embed_dim'][-1]  # 448
        self.backbone.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),          # stabilises activation magnitudes
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),            # regularisation against memorisation
            nn.Linear(256, num_classes),
        )
        # Initialise linear layers
        for m in self.backbone.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

        self.num_classes = num_classes

        # ── GradCAM storage ────────────────────────────────────────────────── #
        self.gradients = None
        self.activations = None
        self._register_hooks()

        # ── Optionally load pretrained weights ─────────────────────────────── #
        if pretrained_path and os.path.isfile(pretrained_path):
            self._load_pretrained(pretrained_path)

    # ───────────────────── Hook registration ───────────────────────────────── #
    def _register_hooks(self):
        """Register forward/backward hooks on the last backbone stage (blocks3)."""

        def fwd_hook(module, inp, out):
            self.activations = out

        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        target_layer = self.backbone.blocks3
        target_layer.register_forward_hook(fwd_hook)
        target_layer.register_full_backward_hook(bwd_hook)

    # ───────────────────── Weight loading ──────────────────────────────────── #
    def _load_pretrained(self, path: str):
        """Load pretrained weights (handles both original backbone & fine-tuned model)."""
        ckpt = torch.load(path, map_location='cpu')
        state_dict = ckpt.get('model', ckpt)
        
        # Check if keys have 'backbone.' prefix (indicates our AnomalyMobileMamba wrapper)
        is_full_model = any(k.startswith('backbone.') for k in state_dict.keys())
        
        if is_full_model:
            # Load with strict=False to allow for architecture upgrades (like our new head)
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            print(f"[Pretrained] Full Model loaded from {path} (Flexible Mode)")
            if missing:
                print(f"  Note: {len(missing)} new layers initialized fresh (e.g. BatchNorm/ReLU)")
        else:
            # Load only into the backbone part
            missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
            print(f"[Pretrained] Research Backbone loaded from {path}")
            print(f"  Missing keys (head expected): {len(missing)}")

    # ───────────────────── Feature extraction (backbone only) ──────────────── #
    def _extract_features(self, x):
        """
        Run input through backbone stages and global pool, but NOT through head.
        Returns the 448-dim feature vector per image.

        This mirrors MobileMamba.forward() but stops before self.head:
            patch_embed → blocks1 → blocks2 → blocks3 → adaptive_avg_pool → flatten
        """
        x = self.backbone.patch_embed(x)
        x = self.backbone.blocks1(x)
        x = self.backbone.blocks2(x)
        x = self.backbone.blocks3(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return x  # [B, 448]

    # ───────────────────── Forward pass ────────────────────────────────────── #
    def forward(self, x):
        if x.dim() == 5:
            # ── Temporal clip mode: x is [B, T, C, H, W] ──
            # Process each frame through backbone, then average features
            # across the T frames to capture motion context.
            B, T, C, H, W = x.shape
            x_flat = x.reshape(B * T, C, H, W)
            feats = self._extract_features(x_flat)       # [B*T, 448]
            feats = feats.reshape(B, T, -1).mean(dim=1)  # [B, 448]
            return self.backbone.head(feats)
        else:
            # ── Single frame mode: x is [B, C, H, W] ──
            return self.backbone(x)

    def predict(self, x):
        """Return (predicted_class, probabilities) for a batch."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
        return preds, probs

    # ───────────────────── GradCAM ─────────────────────────────────────────── #
    def get_gradcam(self, x, class_idx=None):
        """
        Compute GradCAM heatmap for input tensor x (single frame only).

        Args:
            x: input tensor (1, 3, H, W)
            class_idx: target class. If None, uses the predicted class.

        Returns:
            cam: normalised heatmap tensor (1, 1, h, w)
            class_idx: the class used for GradCAM
            probs: softmax probability vector
        """
        self.eval()
        self.zero_grad()
        x.requires_grad_(True)

        # Always use single-frame path for GradCAM
        logits = self.backbone(x)
        probs = torch.softmax(logits, dim=1)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        logits[0, class_idx].backward()

        grads = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (grads * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.detach(), class_idx, probs[0].detach()

    def get_class_name(self, idx: int) -> str:
        """Return the readable class name for a given index."""
        return CLASS_NAMES[idx] if 0 <= idx < len(CLASS_NAMES) else 'Unknown'


# ─────────────────────── Convenience loader ────────────────────────────────── #
def load_anomaly_model(checkpoint_path: str = None, device: str = 'cpu'):
    """
    Build and return an AnomalyMobileMamba on the given device.

    Args:
        checkpoint_path: path to .pth file (pretrained or fine-tuned)
        device: 'cpu' or 'cuda'
    """
    model = AnomalyMobileMamba(
        num_classes=NUM_CLASSES,
        pretrained_path=checkpoint_path,
    )
    model = model.to(device)
    return model
