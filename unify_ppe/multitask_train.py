#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multitask YOLO11-S (Shared Backbone: Detection + Pose)
======================================================
Stage 0 → Load pretrained YOLO11-s-pose (NO TRAIN)
Stage 1 → Copy pose backbone → YOLO11-s detector
Stage 2 → Train ONLY detection head on PPE dataset
Stage 3 → Export unified model (det + pose heads) to ONNX
"""

from ultralytics import YOLO
from pathlib import Path
import torch
import torch.nn as nn

# Paths
POSE_PT = r"C:\Users\ali\Projects\fall_research\unify_ppe\models\yolo11s-pose.pt"
DET_PT  = r"C:\Users\ali\Projects\fall_research\unify_ppe\models\yolo11s.pt"
DET_DATA_YAML = r"C:\Users\ali\Projects\fall_research\unify_ppe\data\new_unify_safety.yaml"
SAVE_DIR = Path("unified_yolo11s")

# ------------------------------
# Train detection head
# ------------------------------
def train_unified():
    det_model = YOLO(DET_PT)
    pose_model = YOLO(POSE_PT)

    # Transfer backbone weights from pose → detector
    pose_sd = pose_model.model.state_dict()
    det_sd = det_model.model.state_dict()
    for k, v in pose_sd.items():
        if "pose" in k or "kpt" in k or "head" in k:
            continue
        if k in det_sd and det_sd[k].shape == v.shape:
            det_sd[k] = v
    det_model.model.load_state_dict(det_sd, strict=False)

    # Freeze backbone
    layers = list(det_model.model.model)
    for lyr in layers[:-3]:
        for p in lyr.parameters():
            p.requires_grad = False

    # Train detection head
    det_model.train(
        data=DET_DATA_YAML,
        epochs=80,
        batch=8,
        device="0",
        project=str(SAVE_DIR),
        name="det_unify",
        imgsz=640,
        lr0=0.001,
        warmup_epochs=3
    )
    return SAVE_DIR / "det_unify" / "weights" / "best.pt"

# ------------------------------
# Unified model (det + pose heads)
# ------------------------------
class DualHeadDetPose(nn.Module):
    """
    Shared backbone → Detection + Pose heads
    Backbone chiqishlari list of feature maps sifatida yig‘iladi.
    """
    def __init__(self, det_model: YOLO, pose_model: YOLO, backbone_end_idx=11):
        super().__init__()
        # Backbone layers (pose modeldan)
        self.backbone_layers = nn.ModuleList(list(pose_model.model.model)[:backbone_end_idx])
        # Heads
        self.det_head  = nn.Sequential(*list(det_model.model.model)[backbone_end_idx:])
        self.pose_head = nn.Sequential(*list(pose_model.model.model)[backbone_end_idx:])

    def forward(self, x):
        feats = []
        for layer in self.backbone_layers:
            x = layer(x)
            # Agar qatlam list qaytarsa, uni qo‘shamiz
            if isinstance(x, (list, tuple)):
                feats.extend(x)
            else:
                feats.append(x)
        # Endi feats = list of feature maps
        det_out = self.det_head(feats)
        pose_out = self.pose_head(feats)
        return det_out, pose_out


# ------------------------------
# Export to ONNX → TensorRT
# ------------------------------
def export_unified(det_best, pose_pt, out_path="yolo11s_multitask.onnx"):
    # Load models
    det_model = YOLO(str(det_best))
    pose_model = YOLO(pose_pt)

    # Unified model (det + pose heads)
    class DualHeadDetPose(nn.Module):
        def __init__(self, det_model: YOLO, pose_model: YOLO, backbone_end_idx=11):
            super().__init__()
            self.backbone_layers = nn.ModuleList(list(pose_model.model.model)[:backbone_end_idx])
            self.det_head  = nn.Sequential(*list(det_model.model.model)[backbone_end_idx:])
            self.pose_head = nn.Sequential(*list(pose_model.model.model)[backbone_end_idx:])

        def forward(self, x):
            feats = []
            for layer in self.backbone_layers:
                x = layer(x)
                # Agar qatlam list qaytarsa, uni qo‘shamiz
                if isinstance(x, (list, tuple)):
                    feats.extend(x)
                else:
                    feats.append(x)
            det_out = self.det_head(feats)
            pose_out = self.pose_head(feats)
            return det_out, pose_out

    unified = DualHeadDetPose(det_model, pose_model).eval()

    # Dummy input
    dummy = torch.randn(1, 3, 640, 640)

    # Export to ONNX
    torch.onnx.export(
        unified, dummy, out_path,
        input_names=["images"],
        output_names=["det_out", "pose_out"],
        opset_version=13,
        dynamic_axes={"images": {0: "batch"}}
    )
    print(f"✅ Exported multitask ONNX: {out_path}")
    print(f"👉 Convert with trtexec:\n trtexec --onnx={out_path} --saveEngine=yolo11s_multitask.engine --fp16")
