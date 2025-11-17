#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPE Detector Training with Shared Backbone
==========================================
Train helmet/vest detector using pose model's backbone for better features.
"""

from pathlib import Path
from datetime import datetime
import yaml
import torch
from ultralytics import YOLO


# =============================================================================
# CONFIG
# =============================================================================

# Pretrained models
POSE_PT = r"C:\Users\ali\Projects\fall_research\unify_ppe\models\yolo11s-pose.pt"
DET_PT = r"C:\Users\ali\Projects\fall_research\unify_ppe\models\yolo11s.pt"

# Dataset
DET_DATA_YAML = r"C:\Users\ali\Projects\fall_research\unify_ppe\data\new_unify_safety.yaml"

# Training settings
SAVE_DIR = Path("unified_yolo11s_ppe")
DEVICE = "0"
EPOCHS = 80
BATCH_SIZE = 8
IMG_SIZE = 640
LR0 = 0.001
WARMUP_EPOCHS = 3


# =============================================================================
# HELPERS
# =============================================================================

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def write_yaml(obj, out):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True)


def transfer_backbone(det_model, pose_weights):
    """Transfer pose backbone weights to detection model"""
    print("\n🔄 Transferring Pose → Detection backbone...")
    
    # PyTorch 2.6+ compatibility fix
    ckpt = torch.load(pose_weights, map_location="cpu", weights_only=False)
    pose_sd = ckpt["model"].state_dict() if "model" in ckpt else ckpt
    det_sd = det_model.model.state_dict()
    
    transferred = skipped = 0
    for k, v in pose_sd.items():
        # Skip pose/keypoint head layers
        if any(x in k for x in ["pose", "kpt", "cv3", "cv4"]):
            skipped += 1
            continue
        
        if k in det_sd and det_sd[k].shape == v.shape:
            det_sd[k] = v
            transferred += 1
        else:
            skipped += 1
    
    det_model.model.load_state_dict(det_sd, strict=False)
    print(f"   ✓ Transferred: {transferred} layers")
    print(f"   ⊘ Skipped: {skipped} layers")

def freeze_backbone(model, keep_last_n=3):
    """Freeze backbone, train only last N layers"""
    print(f"\n🔒 Freezing backbone (keeping last {keep_last_n} layers trainable)...")
    
    layers = list(model.model.model)
    frozen_count = 0
    
    for layer in layers[:-keep_last_n]:
        for param in layer.parameters():
            param.requires_grad = False
            frozen_count += 1
    
    trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.model.parameters())
    
    print(f"   ✓ Frozen: {frozen_count} parameters")
    print(f"   ✓ Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def train_ppe_detector():
    """Main training pipeline"""
    
    ensure_dir(SAVE_DIR)
    
    print("\n" + "="*70)
    print("🚀 PPE DETECTOR TRAINING - SHARED BACKBONE APPROACH")
    print("="*70)
    
    # Stage 0: Load pretrained pose model
    print("\n📦 Stage 0: Load Pretrained YOLO11-s-Pose")
    print(f"   Loading: {POSE_PT}")
    
    # Stage 1: Load detector and transfer backbone
    print("\n📦 Stage 1: Initialize Detector")
    det_model = YOLO(DET_PT)
    print(f"   ✓ Loaded: {DET_PT}")
    
    transfer_backbone(det_model, POSE_PT)
    freeze_backbone(det_model, keep_last_n=3)
    
    # Stage 2: Train detection head
    print("\n📦 Stage 2: Train Detection Head")
    print(f"   Dataset: {DET_DATA_YAML}")
    print(f"   Classes: helmet, vest")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Batch: {BATCH_SIZE}")
    print(f"   Device: {DEVICE}")
    print("\n" + "="*70)
    print("🏋️ Training started...")
    print("="*70 + "\n")
    
    results = det_model.train(
        data=DET_DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=str(SAVE_DIR),
        name="det_ppe",
        exist_ok=True,
        imgsz=IMG_SIZE,
        lr0=LR0,
        warmup_epochs=WARMUP_EPOCHS,
        save=True,
        plots=True,
        val=True
    )
    
    # Save training info
    det_best = SAVE_DIR / "det_ppe" / "weights" / "best.pt"
    
    info = {
        "approach": "Shared Backbone (Pose → Detection)",
        "pose_backbone_source": str(POSE_PT),
        "detector_base": str(DET_PT),
        "trained_model": str(det_best),
        "dataset": str(DET_DATA_YAML),
        "classes": ["helmet", "vest"],
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "training_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": DEVICE
    }
    
    write_yaml(info, SAVE_DIR / "training_info.yaml")
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📁 Results saved to: {SAVE_DIR}")
    print(f"🏆 Best model: {det_best}")
    print(f"📊 Training plots: {SAVE_DIR}/det_ppe/")
    print(f"📝 Training info: {SAVE_DIR}/training_info.yaml")
    
    return info


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    train_info = train_ppe_detector()