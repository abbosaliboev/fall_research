#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export unified multitask model (det + pose heads) to ONNX.
"""

from ultralytics import YOLO
import torch
import torch.nn as nn

# Paths
POSE_PT = r"C:\Users\ali\Projects\fall_research\unify_ppe\models\yolo11s-pose.pt"
DET_BEST = r"C:\Users\ali\Projects\fall_research\unify_ppe\unified_yolo11s\det_unify\weights\best.pt"

class DualHeadDetPose(nn.Module):
    def __init__(self, det_model: YOLO, pose_model: YOLO, backbone_end_idx=11):
        super().__init__()
        self.det_model = det_model.model
        self.pose_model = pose_model.model
        self.backbone_end_idx = backbone_end_idx

    def forward(self, x):
        # Backbone chiqishi (pose modeldan)
        feats = []
        for i, layer in enumerate(self.pose_model.model):
            x = layer(x)
            if i == self.backbone_end_idx - 1:
                # Shu nuqtadan keyin feature mapsni yig‘amiz
                if isinstance(x, (list, tuple)):
                    feats = list(x)
                else:
                    feats = [x]
                break

        # Head forward (to‘g‘ri formatda list beramiz)
        det_out = self.det_model.model[self.backbone_end_idx:](feats)
        pose_out = self.pose_model.model[self.backbone_end_idx:](feats)
        return det_out, pose_out



def export_unified(out_path="yolo11s_multitask.onnx"):
    det_model = YOLO(DET_BEST)
    pose_model = YOLO(POSE_PT)
    unified = DualHeadDetPose(det_model, pose_model).eval()

    dummy = torch.randn(1, 3, 640, 640)
    torch.onnx.export(
        unified, dummy, out_path,
        input_names=["images"],
        output_names=["det_out", "pose_out"],
        opset_version=13,
        dynamic_axes={"images": {0: "batch"}}
    )
    print(f"✅ Exported multitask ONNX: {out_path}")
    print(f"👉 Convert with trtexec:\n trtexec --onnx={out_path} --saveEngine=yolo11s_multitask.engine --fp16")

if __name__ == "__main__":
    export_unified()
