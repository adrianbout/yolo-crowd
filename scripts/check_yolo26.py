"""
Verify YOLO26 actually works here before trusting it.

Two separate questions, and the second is the one that matters. Loading the
checkpoint is easy - its modules already exist in the 8.3 line. But YOLO26
removes Distribution Focal Loss from the detection head, and an older head
that still expects DFL can decode those weights into boxes that look
plausible and are wrong. So this does not just load the model: it runs it on
real footage and draws what it found, next to the RGB detector's answer on
the same frame.

Usage
-----
  python scripts/check_yolo26.py
  python scripts/check_yolo26.py --video "hallways (2).mp4" --out check.jpg
  python scripts/check_yolo26.py --compare        # also run the yolo-crowd model
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import cv2
import numpy as np


def grab_frame(video: str, frac: float = 0.5) -> np.ndarray:
    cap = cv2.VideoCapture(video, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise SystemExit("Could not open " + video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * frac))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("Could not read a frame from " + video)
    return frame


def draw(frame, boxes, colour, label):
    out = frame.copy()
    for (x1, y1, x2, y2, conf) in boxes:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), colour, 2)
        cv2.putText(out, "%.2f" % conf, (int(x1), max(12, int(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, "%s - %d people" % (label, len(boxes)), (10, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def sanity(boxes, shape):
    """
    Whether these boxes could plausibly be people in this frame.

    A head decoding weights it does not understand tends to fail in one of a
    few visible ways: boxes outside the frame, boxes covering most of it, or
    hundreds of them stacked on one spot. None of this proves correctness -
    it catches the obvious wreck, not a subtle one.
    """
    h, w = shape[:2]
    notes = []
    if not boxes:
        notes.append("no detections at all - suspicious on busy footage")
        return notes
    areas = [((x2 - x1) * (y2 - y1)) / float(w * h) for x1, y1, x2, y2, _ in boxes]
    outside = sum(1 for x1, y1, x2, y2, _ in boxes
                  if x1 < -5 or y1 < -5 or x2 > w + 5 or y2 > h + 5)
    huge = sum(1 for a in areas if a > 0.5)
    if outside:
        notes.append("%d boxes fall outside the frame" % outside)
    if huge:
        notes.append("%d boxes cover over half the frame" % huge)
    if len(boxes) > 300:
        notes.append("%d boxes - at or past the one-to-one head's cap" % len(boxes))
    if max(areas) > 0.9:
        notes.append("largest box is nearly the whole frame")
    if not notes:
        notes.append("boxes look plausible (sizes %.3f-%.3f of frame)" % (min(areas), max(areas)))
    return notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="hallways (2).mp4")
    ap.add_argument("--weights", default="weights/yolo26m.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default="yolo26_check.jpg")
    ap.add_argument("--compare", action="store_true", help="also run the yolo-crowd RGB detector")
    args = ap.parse_args()

    import ultralytics
    print("ultralytics %s" % ultralytics.__version__)
    import torch
    print("torch %s, cuda %s" % (torch.__version__, torch.cuda.is_available()))
    print()

    frame = grab_frame(args.video)
    print("frame from %s: %dx%d" % (args.video, frame.shape[1], frame.shape[0]))

    # --- load -------------------------------------------------------------
    from ultralytics import YOLO
    try:
        model = YOLO(args.weights)
        print("loaded %s" % args.weights)
    except Exception as e:
        print("FAILED to load: %s" % e)
        print("-> upgrade with: pip install -U ultralytics")
        raise SystemExit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = model(frame, conf=args.conf, imgsz=args.imgsz, classes=[0],
                    device=device, verbose=False)

    boxes = []
    if results and results[0].boxes is not None and len(results[0].boxes):
        b = results[0].boxes
        xyxy = b.xyxy.cpu().numpy()
        confs = b.conf.cpu().numpy()
        boxes = [(x1, y1, x2, y2, c) for (x1, y1, x2, y2), c in zip(xyxy, confs)]

    print()
    print("YOLO26 found %d people" % len(boxes))
    for note in sanity(boxes, frame.shape):
        print("  - " + note)

    panels = [draw(frame, boxes, (0, 200, 255), "YOLO26")]

    # --- optional comparison ----------------------------------------------
    if args.compare:
        try:
            from detection.detector import YOLODetector
            rgb = YOLODetector(model_path="weights/yolo-crowd.pt", device=device,
                               class_filter=[0], img_size=args.imgsz,
                               confidence_threshold=args.conf, iou_threshold=0.45)
            dets = rgb.detect_batch(frames=[frame], camera_ids=["x"],
                                    inference_configs=[{"confidence_threshold": args.conf,
                                                        "iou_threshold": 0.45,
                                                        "img_size": args.imgsz}])["x"]
            crowd = [(d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3], d.confidence) for d in dets]
            print()
            print("yolo-crowd found %d people" % len(crowd))
            panels.append(draw(frame, crowd, (0, 255, 0), "yolo-crowd"))
        except Exception as e:
            print("comparison skipped: %s" % e)

    stacked = panels[0] if len(panels) == 1 else np.hstack(panels)
    cv2.imwrite(args.out, stacked)
    print()
    print("wrote %s - open it and count. If the boxes are not on people, the" % args.out)
    print("head is mis-decoding and you need ultralytics >= 8.4.")


if __name__ == "__main__":
    main()
