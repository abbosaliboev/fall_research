#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPE Detection Inference with Head Keypoints
===========================================
Uses head keypoints to improve helmet detection accuracy.
Similar to gloves detection via wrist keypoints.
"""

import cv2
import numpy as np
from pathlib import Path
import time
from ultralytics import YOLO


# =============================================================================
# CONFIG
# =============================================================================

# Trained models
TRAINED_PPE_MODEL = r"C:\Users\ali\Projects\fall_research\unify_ppe\unified_yolo11s_ppe\det_ppe\weights\best.pt"
POSE_MODEL = r"C:\Users\ali\Projects\fall_research\unify_ppe\models\yolo11s-pose.pt"

# Detection settings
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.3
HEAD_BBOX_EXPAND = 1.5

# Head keypoints indices (COCO-17)
HEAD_KEYPOINTS = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4
}


# =============================================================================
# PPE DETECTOR CLASS
# =============================================================================

class PPEDetectorWithPose:
    """PPE detection using head keypoints for improved accuracy"""
    
    def __init__(self, ppe_model_path, pose_model_path, device='0'):
        self.device = f"cuda:{device}" if device != 'cpu' else "cpu"
        
        print("📦 Loading models...")
        self.ppe_detector = YOLO(ppe_model_path)
        self.pose_model = YOLO(pose_model_path)
        print(f"   ✓ PPE Detector: {ppe_model_path}")
        print(f"   ✓ Pose Model: {pose_model_path}")
        print(f"   ✓ Device: {self.device}\n")
    
    
    def get_head_bbox(self, keypoints, expand_ratio=1.5):
        """Calculate head bounding box from keypoints"""
        head_kpts = keypoints[list(HEAD_KEYPOINTS.values())]
        visible = head_kpts[head_kpts[:, 2] > 0.5]
        
        if len(visible) < 2:
            return None
        
        x_coords = visible[:, 0]
        y_coords = visible[:, 1]
        
        x1, x2 = x_coords.min(), x_coords.max()
        y1, y2 = y_coords.min(), y_coords.max()
        
        w = x2 - x1
        h = y2 - y1
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        new_w = w * expand_ratio
        new_h = h * expand_ratio
        
        return (
            int(cx - new_w / 2),
            int(cy - new_h / 2),
            int(cx + new_w / 2),
            int(cy + new_h / 2)
        )
    
    
    def bbox_iou(self, box1, box2):
        """Calculate IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0
    
    
    def match_helmets_to_heads(self, helmet_boxes, head_boxes, iou_thresh=0.3):
        """Match helmet detections to head regions"""
        matches = []
        matched_heads = set()
        
        for h_idx, helmet in enumerate(helmet_boxes):
            best_iou = 0
            best_head = -1
            
            for head_idx, head in enumerate(head_boxes):
                iou = self.bbox_iou(helmet[:4], head)
                if iou > best_iou and iou > iou_thresh:
                    best_iou = iou
                    best_head = head_idx
            
            if best_head >= 0:
                matches.append((h_idx, best_head, best_iou))
                matched_heads.add(best_head)
        
        unmatched_heads = [i for i in range(len(head_boxes)) if i not in matched_heads]
        return matches, unmatched_heads
    
    
    def detect(self, image, conf_thresh=0.25, visualize=True):
        """Run PPE detection with head keypoints"""
        h, w = image.shape[:2]
        
        # 1. Detect PPE (helmet/vest)
        ppe_results = self.ppe_detector(image, conf=conf_thresh, verbose=False)[0]
        helmet_boxes = []
        vest_boxes = []
        
        if ppe_results.boxes is not None:
            for box in ppe_results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                
                if cls == 0:  # helmet
                    helmet_boxes.append((x1, y1, x2, y2, conf, cls))
                elif cls == 1:  # vest
                    vest_boxes.append((x1, y1, x2, y2, conf, cls))
        
        # 2. Detect pose keypoints
        pose_results = self.pose_model(image, conf=0.3, verbose=False)[0]
        head_boxes = []
        all_keypoints = []
        
        if pose_results.keypoints is not None:
            keypoints = pose_results.keypoints.xy.cpu().numpy()
            confidences = pose_results.keypoints.conf.cpu().numpy()
            
            for person_idx in range(len(keypoints)):
                kpts = np.concatenate([
                    keypoints[person_idx],
                    confidences[person_idx][:, None]
                ], axis=1)
                
                all_keypoints.append(kpts)
                
                head_bbox = self.get_head_bbox(kpts, expand_ratio=HEAD_BBOX_EXPAND)
                if head_bbox:
                    head_bbox = (
                        max(0, head_bbox[0]),
                        max(0, head_bbox[1]),
                        min(w, head_bbox[2]),
                        min(h, head_bbox[3])
                    )
                    head_boxes.append(head_bbox)
        
        # 3. Match helmets to heads
        matches, unmatched_heads = self.match_helmets_to_heads(
            helmet_boxes, head_boxes, iou_thresh=IOU_THRESHOLD
        )
        
        # 4. Build results
        results = {
            'num_people': len(head_boxes),
            'num_helmets': len(helmet_boxes),
            'num_vests': len(vest_boxes),
            'people_with_helmet': len(matches),
            'people_without_helmet': len(unmatched_heads),
            'helmet_compliance_rate': len(matches) / len(head_boxes) * 100 if head_boxes else 0,
            'violations': unmatched_heads,
            'helmet_boxes': helmet_boxes,
            'vest_boxes': vest_boxes,
            'head_boxes': head_boxes,
            'matches': matches,
            'keypoints': all_keypoints
        }
        
        # 5. Visualize
        vis_image = None
        if visualize:
            vis_image = self._draw_results(image.copy(), results)
        
        return results, vis_image
    
    
    def _draw_results(self, image, results):
        """Draw detection results on image"""
        
        # Draw head regions
        for head_idx, head in enumerate(results['head_boxes']):
            if head_idx in results['violations']:
                color = (0, 0, 255)  # Red - NO HELMET
                label = "⚠️ NO HELMET"
            else:
                color = (0, 255, 0)  # Green - SAFE
                label = "✓ HELMET OK"
            
            cv2.rectangle(image, (head[0], head[1]), (head[2], head[3]), color, 2)
            cv2.putText(image, label, (head[0], head[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw helmet detections
        for helmet in results['helmet_boxes']:
            cv2.rectangle(image, (int(helmet[0]), int(helmet[1])),
                        (int(helmet[2]), int(helmet[3])), (255, 0, 0), 2)
            cv2.putText(image, f"Helmet {helmet[4]:.2f}",
                       (int(helmet[0]), int(helmet[1])-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Draw vest detections
        for vest in results['vest_boxes']:
            cv2.rectangle(image, (int(vest[0]), int(vest[1])),
                        (int(vest[2]), int(vest[3])), (0, 255, 255), 2)
            cv2.putText(image, f"Vest {vest[4]:.2f}",
                       (int(vest[0]), int(vest[1])-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Draw head keypoints
        for kpts in results['keypoints']:
            for kp_idx in HEAD_KEYPOINTS.values():
                x, y, conf = kpts[kp_idx]
                if conf > 0.5:
                    cv2.circle(image, (int(x), int(y)), 4, (255, 255, 0), -1)
        
        # Draw summary
        summary = f"People: {results['num_people']} | " \
                 f"Helmets: {results['people_with_helmet']}/{results['num_people']} | " \
                 f"Compliance: {results['helmet_compliance_rate']:.1f}%"
        
        cv2.rectangle(image, (10, 10), (650, 50), (0, 0, 0), -1)
        cv2.putText(image, summary, (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return image
    
    
    def detect_video(self, video_path, output_path=None, show=True):
        """Process video file"""
        cap = cv2.VideoCapture(video_path)
        
        if output_path:
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, 
                                 cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps, (w, h))
        
        frame_count = 0
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            results, vis_frame = self.detect(frame, conf_thresh=CONF_THRESHOLD)
            
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Add FPS counter
            cv2.putText(vis_frame, f"FPS: {fps:.1f}", (w-150, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if output_path:
                out.write(vis_frame)
            
            if show:
                cv2.imshow('PPE Detection', vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"\n✅ Processed {frame_count} frames at {fps:.1f} FPS")


# =============================================================================
# MAIN - EXAMPLE USAGE
# =============================================================================

def main():
    # Initialize detector
    detector = PPEDetectorWithPose(
        ppe_model_path=TRAINED_PPE_MODEL,
        pose_model_path=POSE_MODEL,
        device='0'
    )
    
    # Test on image
    test_image = r"test_image.jpg"
    
    if Path(test_image).exists():
        print(f"🖼️ Processing image: {test_image}\n")
        
        image = cv2.imread(test_image)
        results, vis_image = detector.detect(image, conf_thresh=CONF_THRESHOLD)
        
        # Print results
        print("="*60)
        print("📊 DETECTION RESULTS")
        print("="*60)
        print(f"   People detected: {results['num_people']}")
        print(f"   Helmets detected: {results['num_helmets']}")
        print(f"   Vests detected: {results['num_vests']}")
        print(f"   People WITH helmet: {results['people_with_helmet']}")
        print(f"   People WITHOUT helmet: {results['people_without_helmet']}")
        print(f"   Helmet compliance: {results['helmet_compliance_rate']:.1f}%")
        
        if results['violations']:
            print(f"\n⚠️  SAFETY VIOLATIONS:")
            for idx in results['violations']:
                print(f"      Person #{idx+1}: NO HELMET DETECTED!")
        else:
            print(f"\n✅ All workers are wearing helmets!")
        
        print("="*60)
        
        # Save result
        output = "ppe_detection_result.jpg"
        cv2.imwrite(output, vis_image)
        print(f"\n💾 Result saved: {output}")
    
    # Test on video
    test_video = r"test_video.mp4"
    if Path(test_video).exists():
        print(f"\n🎥 Processing video: {test_video}\n")
        detector.detect_video(test_video, output_path="ppe_detection_output.mp4", show=True)


if __name__ == "__main__":
    main()