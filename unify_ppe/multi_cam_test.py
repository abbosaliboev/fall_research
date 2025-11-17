#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultra-Fast PPE Detection for Multi-Camera
=========================================
Optimizations:
- Reduced image size
- Batch processing ready
- Minimal overhead
- GPU optimization
- Frame buffering
Target: 60+ FPS single camera, 30+ FPS per camera (6-8 cameras)
"""

import cv2
import numpy as np
import time
import threading
from queue import Queue
from collections import deque, defaultdict
from ultralytics import YOLO
import torch


# =============================================================================
# OPTIMIZED CONFIG
# =============================================================================

TRAINED_PPE_MODEL = r"unified_yolo11s_ppe\det_ppe\weights\best.pt"

# Performance settings
INPUT_SIZE = 416  # 640 → 416 for 2.5× speed boost
CONF_THRESHOLD = 0.35
SKIP_FRAMES = 1  # Process every frame (but faster inference)
BATCH_SIZE = 1

# Smoothing (lighter)
SMOOTH_WINDOW = 3
VIOLATION_THRESHOLD = 2

# Threading
USE_THREADING = True  # Separate thread for inference


# =============================================================================
# ULTRA-FAST DETECTOR
# =============================================================================

class UltraFastPPEDetector:
    """Optimized detector for multi-camera deployment"""
    
    def __init__(self, model_path, device='0'):
        self.device = f"cuda:{device}" if torch.cuda.is_available() and device != 'cpu' else "cpu"
        
        print("="*70)
        print("⚡ ULTRA-FAST PPE DETECTOR")
        print("="*70)
        print(f"📦 Loading model: {model_path}")
        
        # Load with optimization
        self.model = YOLO(model_path)
        
        # Optimize model
        if self.device.startswith('cuda'):
            self.model.fuse()  # Fuse layers for speed
            print("✅ Model fused for speed")
        
        print(f"✅ Device: {self.device}")
        print(f"✅ Input size: {INPUT_SIZE}x{INPUT_SIZE}")
        print(f"✅ Expected FPS: 60+ (single cam)")
        print("="*70 + "\n")
        
        # Tracking
        self.person_states = defaultdict(lambda: {
            'history': deque(maxlen=SMOOTH_WINDOW),
            'last_seen': 0,
            'status': None
        })
        
        # FPS
        self.fps_history = deque(maxlen=30)
        self.frame_count = 0
        
        # Threading
        self.frame_queue = Queue(maxsize=2)
        self.result_queue = Queue(maxsize=2)
        self.running = False
        self.inference_thread = None
    
    
    def get_person_id(self, bbox):
        """Fast person ID from position"""
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        return f"{cx//150}_{cy//150}"  # 150px grid
    
    
    def detect_fast(self, frame):
        """Ultra-fast detection"""
        start = time.time()
        
        # Resize for speed
        h, w = frame.shape[:2]
        small_frame = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        
        # Run inference
        results = self.model(small_frame, conf=CONF_THRESHOLD, verbose=False)[0]
        
        # Scale factor
        scale_x = w / INPUT_SIZE
        scale_y = h / INPUT_SIZE
        
        helmet_boxes = []
        head_boxes = []
        
        if results.boxes:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                
                # Scale back to original size
                x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                
                if cls == 0:  # helmet
                    helmet_boxes.append((x1, y1, x2, y2, conf))
                elif cls == 2:  # head
                    head_boxes.append((x1, y1, x2, y2, conf))
        
        # Fast matching
        persons = []
        for head in head_boxes:
            hx1, hy1, hx2, hy2, h_conf = head
            head_cx = (hx1 + hx2) / 2
            head_cy = (hy1 + hy2) / 2
            head_h = hy2 - hy1
            
            has_helmet = False
            for helmet in helmet_boxes:
                mx1, my1, mx2, my2, m_conf = helmet
                helmet_cx = (mx1 + mx2) / 2
                helmet_cy = (my1 + my2) / 2
                
                dist = np.sqrt((head_cx - helmet_cx)**2 + (head_cy - helmet_cy)**2)
                
                if dist < head_h * 1.3:
                    has_helmet = True
                    break
            
            # Update tracking
            person_id = self.get_person_id(head[:4])
            self.person_states[person_id]['history'].append(has_helmet)
            self.person_states[person_id]['last_seen'] = self.frame_count
            
            # Stable status
            history = list(self.person_states[person_id]['history'])
            if len(history) >= SMOOTH_WINDOW:
                violation_count = sum([not x for x in history])
                if violation_count >= VIOLATION_THRESHOLD:
                    status = 'violation'
                else:
                    status = 'safe'
            else:
                status = self.person_states[person_id].get('status', 'unknown')
            
            self.person_states[person_id]['status'] = status
            
            persons.append({
                'bbox': (hx1, hy1, hx2, hy2),
                'status': status,
                'conf': h_conf
            })
        
        # FPS
        process_time = time.time() - start
        fps = 1.0 / process_time if process_time > 0 else 0
        self.fps_history.append(fps)
        
        return {
            'persons': persons,
            'fps': sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0,
            'process_time': process_time * 1000
        }
    
    
    def inference_worker(self):
        """Background inference thread"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                results = self.detect_fast(frame)
                
                # Clear old results
                while not self.result_queue.empty():
                    try:
                        self.result_queue.get_nowait()
                    except:
                        break
                
                self.result_queue.put(results)
            except:
                continue
    
    
    def draw_minimal(self, frame, results):
        """Minimal drawing for speed"""
        h, w = frame.shape[:2]
        
        violation_count = 0
        safe_count = 0
        
        for person in results['persons']:
            x1, y1, x2, y2 = person['bbox']
            status = person['status']
            
            if status == 'violation':
                color = (0, 0, 255)
                label = "NO HELMET"
                thickness = 3
                violation_count += 1
            elif status == 'safe':
                color = (0, 255, 0)
                label = "SAFE"
                thickness = 2
                safe_count += 1
            else:
                color = (0, 165, 255)
                label = "CHECK"
                thickness = 2
            
            # Draw
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Label
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Minimal panel
        panel = np.zeros((60, w, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)
        
        total = safe_count + violation_count
        compliance = (safe_count / total * 100) if total > 0 else 0
        
        # Info
        info = f"FPS: {results['fps']:.1f} | People: {total} | Safe: {safe_count} | Violations: {violation_count} | {compliance:.0f}%"
        cv2.putText(panel, info, (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return np.vstack([panel, frame])
    
    
    def run_rtsp(self, rtsp_url, display=True, benchmark=False):
        """Run with optional threading"""
        print(f"🎥 Connecting: {rtsp_url}")
        cap = cv2.VideoCapture(rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print("❌ Connection failed")
            return
        
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"✅ Connected: {w}x{h}")
        
        if USE_THREADING:
            print("🔧 Starting inference thread...")
            self.running = True
            self.inference_thread = threading.Thread(target=self.inference_worker)
            self.inference_thread.daemon = True
            self.inference_thread.start()
        
        print("\n" + "="*70)
        print("⚡ DETECTION STARTED")
        print("="*70)
        print("Press 'q' to quit, 'b' to toggle benchmark mode")
        print("="*70 + "\n")
        
        # Benchmark
        benchmark_frames = 300
        benchmark_start = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                
                # Detect
                if USE_THREADING:
                    # Put frame
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame)
                    
                    # Get results
                    try:
                        results = self.result_queue.get_nowait()
                    except:
                        continue
                else:
                    results = self.detect_fast(frame)
                
                # Draw
                if display and not benchmark:
                    vis = self.draw_minimal(frame, results)
                    cv2.imshow('PPE Detection', vis)
                
                # Benchmark mode
                if benchmark and self.frame_count >= benchmark_frames:
                    elapsed = time.time() - benchmark_start
                    avg_fps = benchmark_frames / elapsed
                    print(f"\n📊 BENCHMARK RESULTS:")
                    print(f"   Frames: {benchmark_frames}")
                    print(f"   Time: {elapsed:.2f}s")
                    print(f"   Average FPS: {avg_fps:.1f}")
                    print(f"   Frame time: {1000/avg_fps:.1f}ms")
                    break
                
                # Stats every 100 frames
                if self.frame_count % 100 == 0:
                    print(f"📊 Frame {self.frame_count} | FPS: {results['fps']:.1f} | "
                          f"Time: {results['process_time']:.1f}ms")
                
                # Keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('b'):
                    benchmark = not benchmark
                    if benchmark:
                        print("🔬 Benchmark mode ON")
                        benchmark_start = time.time()
                        self.frame_count = 0
                    else:
                        print("🔬 Benchmark mode OFF")
        
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted")
        
        finally:
            self.running = False
            cap.release()
            cv2.destroyAllWindows()
            
            print("\n" + "="*70)
            print("✅ Session ended")
            print("="*70)


# =============================================================================
# MULTI-CAMERA MANAGER
# =============================================================================

class MultiCameraManager:
    """Manage multiple cameras efficiently"""
    
    def __init__(self, model_path, device='0'):
        self.device = device
        self.model = UltraFastPPEDetector(model_path, device)
        self.cameras = {}
    
    
    def add_camera(self, cam_id, rtsp_url):
        """Add camera stream"""
        cap = cv2.VideoCapture(rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if cap.isOpened():
            self.cameras[cam_id] = {
                'cap': cap,
                'url': rtsp_url,
                'frame': None,
                'results': None,
                'fps': 0
            }
            print(f"✅ Camera {cam_id} added")
            return True
        else:
            print(f"❌ Camera {cam_id} failed")
            return False
    
    
    def run_multicam(self, grid_cols=3):
        """Run all cameras"""
        if not self.cameras:
            print("❌ No cameras added")
            return
        
        print(f"\n⚡ Starting {len(self.cameras)} cameras...")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                frames = []
                
                for cam_id, cam_data in self.cameras.items():
                    ret, frame = cam_data['cap'].read()
                    if not ret:
                        continue
                    
                    # Detect
                    results = self.model.detect_fast(frame)
                    
                    # Draw
                    vis = self.model.draw_minimal(frame, results)
                    
                    # Resize for grid
                    vis = cv2.resize(vis, (640, 420))
                    
                    # Add camera label
                    cv2.putText(vis, f"Camera {cam_id}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    
                    frames.append(vis)
                    cam_data['fps'] = results['fps']
                
                # Create grid
                if frames:
                    rows = []
                    for i in range(0, len(frames), grid_cols):
                        row = frames[i:i+grid_cols]
                        while len(row) < grid_cols:
                            row.append(np.zeros_like(frames[0]))
                        rows.append(np.hstack(row))
                    
                    grid = np.vstack(rows)
                    cv2.imshow('Multi-Camera PPE Detection', grid)
                
                frame_count += 1
                
                # Stats
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    overall_fps = frame_count / elapsed
                    print(f"\n📊 Frame {frame_count}:")
                    for cam_id, cam_data in self.cameras.items():
                        print(f"   Cam {cam_id}: {cam_data['fps']:.1f} FPS")
                    print(f"   Overall: {overall_fps:.1f} FPS")
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted")
        
        finally:
            for cam_data in self.cameras.values():
                cam_data['cap'].release()
            cv2.destroyAllWindows()


# =============================================================================
# BENCHMARK TEST
# =============================================================================

def run_benchmark(model_path, test_frames=300):
    """Benchmark test"""
    print("\n" + "="*70)
    print("🔬 BENCHMARK TEST")
    print("="*70)
    
    detector = UltraFastPPEDetector(model_path, device='0')
    
    # Create dummy frames
    dummy_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    
    print(f"Testing {test_frames} frames...")
    
    times = []
    start = time.time()
    
    for i in range(test_frames):
        frame_start = time.time()
        results = detector.detect_fast(dummy_frame)
        times.append(time.time() - frame_start)
        
        if (i+1) % 100 == 0:
            print(f"Progress: {i+1}/{test_frames}")
    
    total_time = time.time() - start
    avg_time = np.mean(times)
    avg_fps = 1.0 / avg_time
    
    print("\n" + "="*70)
    print("📊 BENCHMARK RESULTS")
    print("="*70)
    print(f"Total frames: {test_frames}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average FPS: {avg_fps:.1f}")
    print(f"Min FPS: {1.0/max(times):.1f}")
    print(f"Max FPS: {1.0/min(times):.1f}")
    print(f"Frame time: {avg_time*1000:.2f}ms")
    print("="*70)
    
    # Multi-camera prediction
    print(f"\n🎥 Multi-camera predictions:")
    for n_cams in [1, 2, 4, 6, 8]:
        predicted_fps = avg_fps / n_cams
        print(f"   {n_cams} cameras: ~{predicted_fps:.1f} FPS per camera")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Ultra-Fast PPE Detection')
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'multi', 'benchmark'],
                       help='Mode: single camera, multi camera, or benchmark')
    parser.add_argument('--rtsp', type=str, default=None,
                       help='RTSP URL for single camera')
    parser.add_argument('--rtsp-list', type=str, nargs='+', default=None,
                       help='List of RTSP URLs for multi camera')
    parser.add_argument('--device', type=str, default='0',
                       help='GPU device')
    parser.add_argument('--benchmark-frames', type=int, default=300,
                       help='Number of frames for benchmark')
    
    args = parser.parse_args()
    
    if args.mode == 'benchmark':
        run_benchmark(TRAINED_PPE_MODEL, args.benchmark_frames)
    
    elif args.mode == 'single':
        if not args.rtsp:
            print("❌ --rtsp required for single mode")
        else:
            detector = UltraFastPPEDetector(TRAINED_PPE_MODEL, args.device)
            detector.run_rtsp(args.rtsp, display=True, benchmark=False)
    
    elif args.mode == 'multi':
        if not args.rtsp_list:
            print("❌ --rtsp-list required for multi mode")
        else:
            manager = MultiCameraManager(TRAINED_PPE_MODEL, args.device)
            for i, url in enumerate(args.rtsp_list):
                manager.add_camera(f"cam_{i+1}", url)
            manager.run_multicam(grid_cols=3)