"""
trainer.py — Training loop for MobileMamba anomaly detection.

Features:
  • Differential learning rates (backbone vs head)
  • Backbone freezing for the first N epochs
  • Class-weighted CrossEntropyLoss
  • Gradient clipping
  • Cosine annealing scheduler
  • TensorBoard logging
  • Best-model checkpointing

Can be run as a script:
    python anomaly_robot/src/trainer.py
"""

import os
import sys
import time
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ── Project imports ────────────────────────────────────────────────────────── #
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(_PROJECT_ROOT))  # repo root for MobileMamba

from src.dataset import CrimeFrameDataset
from src.model import AnomalyMobileMamba


# ══════════════════════════ Default Config ═══════════════════════════════════ #
DEFAULT_CFG = {
    'data_dir': os.path.join(_PROJECT_ROOT, 'data', 'frames'),
    'checkpoint_dir': os.path.join(_PROJECT_ROOT, 'checkpoints'),
    'log_dir': os.path.join(_PROJECT_ROOT, 'outputs', 'logs'),
    'pretrained_path': os.path.join(_PROJECT_ROOT, 'checkpoints', 'mobilemamba_b1_pretrained.pth'),
    'batch_size': 8,           # clips use more VRAM (8 clips × 8 frames = 64 images)
    'epochs': 60,
    'lr_backbone': 5e-6,       # very gentle — only blocks3 will unfreeze
    'lr_head': 3e-4,
    'weight_decay': 0.05,
    'freeze_epochs': 15,       # let head stabilise before partial unfreeze
    'grad_clip': 1.0,
    'img_size': 256,
    'num_workers': 4,
    'patience': 12,            # temporal training converges slower
    'warmup_epochs': 5,
    'mixup_alpha': 0.2,
    'temporal_size': 8,        # 8-frame clips for motion context
}


# ══════════════════════════ Trainer Class ════════════════════════════════════ #
class AnomalyTrainer:
    def __init__(self, cfg: dict = None):
        self.cfg = {**DEFAULT_CFG, **(cfg or {})}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Trainer] Using device: {self.device}")

        # ── Data ───────────────────────────────────────────────────────────── #
        temporal = self.cfg.get('temporal_size', 1)
        self.train_ds = CrimeFrameDataset(
            self.cfg['data_dir'], 'train', self.cfg['img_size'],
            temporal_size=temporal,
        )
        self.val_ds = CrimeFrameDataset(
            self.cfg['data_dir'], 'val', self.cfg['img_size'],
            temporal_size=temporal,
        )

        # ── Weighted Random Sampler (class-balanced) ───────────────────────── #
        train_labels = self.train_ds.labels
        class_counts = Counter(train_labels)
        num_samples = len(train_labels)
        sample_weights = []
        for label in train_labels:
            sample_weights.append(num_samples / (len(class_counts) * class_counts[label]))
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=num_samples,
            replacement=True,
        )
        print(f"[Trainer] WeightedRandomSampler: {dict(class_counts)}")

        self.train_loader = DataLoader(
            self.train_ds, batch_size=self.cfg['batch_size'],
            sampler=sampler,  # replaces shuffle=True — ensures balanced class exposure
            num_workers=self.cfg['num_workers'], pin_memory=True, drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_ds, batch_size=self.cfg['batch_size'], shuffle=False,
            num_workers=self.cfg['num_workers'], pin_memory=True,
        )

        # ── Model ──────────────────────────────────────────────────────────── #
        pretrained = self.cfg['pretrained_path']
        self.model = AnomalyMobileMamba(
            num_classes=6,
            pretrained_path=pretrained if os.path.isfile(pretrained) else None,
        ).to(self.device)

        # ── Loss with class weights + label smoothing ──────────────────────── #
        class_weights = self.train_ds.get_class_weights().to(self.device)
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=0.1,  # handles noisy crime labels
        )
        print(f"[Trainer] Class weights: {class_weights.tolist()}")

        # ── Optimizer with differential LRs ────────────────────────────────── #
        backbone_params = [p for n, p in self.model.named_parameters()
                           if 'backbone.head' not in n and p.requires_grad]
        head_params = list(self.model.backbone.head.parameters())

        self.optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': self.cfg['lr_backbone']},
            {'params': head_params,     'lr': self.cfg['lr_head']},
        ], weight_decay=self.cfg['weight_decay'])

        # ── Warmup + Cosine Annealing scheduler ────────────────────────────── #
        from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
        warmup_epochs = self.cfg.get('warmup_epochs', 5)
        warmup = LinearLR(
            self.optimizer, start_factor=0.1, end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=self.cfg['epochs'] - warmup_epochs,
            eta_min=1e-6,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )

        # ── Logging ────────────────────────────────────────────────────────── #
        os.makedirs(self.cfg['log_dir'], exist_ok=True)
        os.makedirs(self.cfg['checkpoint_dir'], exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.cfg['log_dir'])
        self.best_val_acc = 0.0

    # ───────────────────── Freeze / Unfreeze backbone ──────────────────────── #
    def _freeze_backbone(self):
        for n, p in self.model.named_parameters():
            if 'backbone.head' not in n:
                p.requires_grad = False
        print("[Trainer] Backbone FROZEN — training head only")

    def _unfreeze_backbone(self):
        """Partial unfreeze: only blocks3 + head.
        Keeps blocks1, blocks2, patch_embed FROZEN to prevent:
          - Destroying low-level pretrained features
          - The 17× slowdown from training all 16M parameters
        """
        for p in self.model.backbone.blocks3.parameters():
            p.requires_grad = True
        for p in self.model.backbone.head.parameters():
            p.requires_grad = True
        print("[Trainer] PARTIAL UNFREEZE — blocks3 + head only (blocks1/2 stay frozen)")

    # ───────────────────── Mixup helper ─────────────────────────────────────── #
    @staticmethod
    def _mixup_data(x, y, alpha=0.2):
        """Apply Mixup augmentation: blend pairs of samples."""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    @staticmethod
    def _mixup_criterion(criterion, pred, y_a, y_b, lam):
        """Compute loss for Mixup blended targets."""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    # ───────────────────── Single epoch ────────────────────────────────────── #
    def _train_one_epoch(self, epoch: int):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        use_mixup = self.cfg.get('mixup_alpha', 0) > 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.cfg['epochs']} [TRAIN]")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            if use_mixup:
                mixed_images, y_a, y_b, lam = self._mixup_data(
                    images, labels, self.cfg['mixup_alpha']
                )
                logits = self.model(mixed_images)
                loss = self._mixup_criterion(self.criterion, logits, y_a, y_b, lam)
                # For accuracy tracking, use the dominant label
                correct += (logits.argmax(1) == y_a).sum().item() if lam >= 0.5 \
                    else (logits.argmax(1) == y_b).sum().item()
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                correct += (logits.argmax(1) == labels).sum().item()

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           max_norm=self.cfg['grad_clip'])
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            total += images.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             acc=f"{correct/total*100:.1f}%")

        avg_loss = total_loss / max(total, 1)
        avg_acc = correct / max(total, 1)
        return avg_loss, avg_acc

    @torch.no_grad()
    def _validate(self, epoch: int):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(self.val_loader,
                                    desc=f"Epoch {epoch+1}/{self.cfg['epochs']} [VAL]"):
            images, labels = images.to(self.device), labels.to(self.device)
            logits = self.model(images)
            loss = self.criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += images.size(0)

        avg_loss = total_loss / max(total, 1)
        avg_acc = correct / max(total, 1)
        return avg_loss, avg_acc

    # ───────────────────── Full training run ───────────────────────────────── #
    def train(self):
        print(f"\n{'='*60}")
        print(f"  MobileMamba Anomaly Detection — Training")
        print(f"  Epochs: {self.cfg['epochs']}  |  Batch: {self.cfg['batch_size']}")
        print(f"  Backbone LR: {self.cfg['lr_backbone']}  |  Head LR: {self.cfg['lr_head']}")
        print(f"  Freeze epochs: {self.cfg['freeze_epochs']}")
        print(f"  Early stopping patience: {self.cfg.get('patience', 'disabled')}")
        print(f"{'='*60}\n")

        # Freeze backbone for initial epochs
        if self.cfg['freeze_epochs'] > 0:
            self._freeze_backbone()

        history = {'train_loss': [], 'train_acc': [],
                   'val_loss': [], 'val_acc': []}

        # Early stopping state
        patience = self.cfg.get('patience', 10)
        patience_counter = 0

        for epoch in range(self.cfg['epochs']):
            # Unfreeze backbone after freeze_epochs
            if epoch == self.cfg['freeze_epochs'] and self.cfg['freeze_epochs'] > 0:
                self._unfreeze_backbone()

            t0 = time.time()
            train_loss, train_acc = self._train_one_epoch(epoch)
            val_loss, val_acc = self._validate(epoch)
            self.scheduler.step()

            elapsed = time.time() - t0

            # Log
            print(f"\n  Epoch {epoch+1:02d}  |  "
                  f"Train Loss: {train_loss:.4f}  Acc: {train_acc*100:.1f}%  |  "
                  f"Val Loss: {val_loss:.4f}  Acc: {val_acc*100:.1f}%  |  "
                  f"Time: {elapsed:.0f}s\n")

            self.writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
            self.writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            # Save best model + early stopping
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                patience_counter = 0
                save_path = os.path.join(self.cfg['checkpoint_dir'], 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'num_classes': 6,
                }, save_path)
                print(f"  ★ Best model saved (val_acc={val_acc*100:.1f}%) → {save_path}")
            else:
                patience_counter += 1
                print(f"  ⏳ No improvement ({patience_counter}/{patience})")
                if patience_counter >= patience:
                    print(f"\n  🛑 Early stopping at epoch {epoch+1} "
                          f"(no improvement for {patience} epochs)")
                    break

        self.writer.close()
        print(f"\n[Trainer] Training complete. Best val accuracy: {self.best_val_acc*100:.1f}%")
        return history


# ══════════════════════════ CLI Entry Point ══════════════════════════════════ #
def main():
    parser = argparse.ArgumentParser(description='Train MobileMamba Anomaly Detector')
    parser.add_argument('--data-dir',    type=str, default=DEFAULT_CFG['data_dir'])
    parser.add_argument('--pretrained',  type=str, default=DEFAULT_CFG['pretrained_path'])
    parser.add_argument('--epochs',      type=int, default=DEFAULT_CFG['epochs'])
    parser.add_argument('--batch-size',  type=int, default=DEFAULT_CFG['batch_size'])
    parser.add_argument('--lr-backbone', type=float, default=DEFAULT_CFG['lr_backbone'])
    parser.add_argument('--lr-head',     type=float, default=DEFAULT_CFG['lr_head'])
    parser.add_argument('--freeze-epochs', type=int, default=DEFAULT_CFG['freeze_epochs'])
    args = parser.parse_args()

    cfg = {
        'data_dir': args.data_dir,
        'pretrained_path': args.pretrained,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr_backbone': args.lr_backbone,
        'lr_head': args.lr_head,
        'freeze_epochs': args.freeze_epochs,
    }

    trainer = AnomalyTrainer(cfg)
    trainer.train()


if __name__ == '__main__':
    main()
