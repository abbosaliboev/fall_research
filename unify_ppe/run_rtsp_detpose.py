import cv2
import torch
import time
from ultralytics import YOLO
# NOTE: using the unified DualHeadDetPose requires careful handling of
# Ultralytics internal graph (Concat, route layers). Building an
# nn.Sequential slice of those layers can break the forward graph and
# cause errors like `cat() received an invalid combination of arguments`.
# For a quick working demo we run detector and pose models separately.


RTSP_URL = "rtsp://admin:Aminok434*@10.198.137.222:554/stream1"


def main():
    print("Loading YOLO models (detector and pose)...")

    # Load YOLO runner objects. These will run detector and pose separately.
    # This is simpler and avoids building a single nn.Sequential backbone which
    # can break Ultralytics' internal graph (Concat/Route layers).
    det = YOLO("#unified_yolo11s_detpose/det_unify/weights/best.pt")
    pose = YOLO("models/yolo11s-pose.pt")

    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        raise Exception("RTSP stream not opened!")

    print("[OK] RTSP streaming started.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream lost... reconnecting...")
            time.sleep(1)
            continue

        # Run detector and pose models separately. We pass the original frame
        # (Ultralytics accepts numpy arrays) and let YOLO handle preprocessing.
        with torch.no_grad():
            det_res = det(frame, imgsz=640, conf=0.25, verbose=False)
            pose_res = pose(frame, imgsz=640, conf=0.25, verbose=False)

        # ---- draw simple output info for debugging ----
        det_count = 0
        try:
            det_count = len(det_res[0].boxes) if det_res and det_res[0].boxes is not None else 0
        except Exception:
            det_count = 0

        pose_count = 0
        try:
            # pose_res[0].keypoints may exist when pose model finds people
            pose_count = len(pose_res[0].keypoints.xy[0]) if pose_res and pose_res[0].keypoints is not None else 0
        except Exception:
            pose_count = 0

        cv2.putText(frame, f"Det:{det_count} Pose:{pose_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Unified DetPose", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
