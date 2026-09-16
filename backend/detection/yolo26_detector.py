"""
YOLO26 Detector

Ultralytics' 2026 generation, wired in as a general-purpose person detector
alongside the custom yolo-crowd weights.

Two things distinguish it from the other Ultralytics wrappers here. It can run
end-to-end with no NMS pass at all - YOLO26 carries a second one-to-one head
for exactly that - which trades a little recall for a shorter, more
predictable pipeline on edge hardware. And this wrapper hands the whole batch
to the model in one call rather than looping frame by frame, so a batch of
twenty cameras is one forward pass instead of twenty.

Ultralytics >= 8.4 is the supported floor. The 8.3 line may still load the
yolo26 checkpoints - inspecting them shows only modules 8.3 already ships - but
YOLO26 removes Distribution Focal Loss from the detection head, and an older
head that still expects DFL can decode those weights into plausible-looking
nonsense rather than failing outright. Treat 8.3 output as unverified until
it has been eyeballed against a frame with known people in it.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from ultralytics import YOLO

from .detector import Detection

logger = logging.getLogger(__name__)

# Ultralytics fetches these on first use if they are not already on disk.
DEFAULT_MODEL = "weights/yolo26m.pt"

# COCO person class. YOLO26 detection weights ship with the COCO label set,
# same as YOLO11, so the filter is unchanged.
PERSON_CLASS = 0


class YOLO26Detector:
    """
    Person detection with Ultralytics YOLO26.

    Mirrors the interface every other detector here exposes, so the factory
    can route a camera to it without anything downstream knowing which model
    produced a box.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        device: str = "cuda",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        img_size: int = 640,
        half_precision: bool = False,
        end_to_end: bool = False
    ):
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.half_precision = half_precision
        # NMS-free mode. Off by default: the one-to-one head caps out at 300
        # detections per frame and trades a little recall, and a busy concourse
        # is exactly where that matters.
        self.end_to_end = end_to_end

        path = Path(model_path)
        if not path.exists() and path.parent != Path("."):
            # A bare name like "yolo26m.pt" is a download hint to Ultralytics;
            # a path into weights/ that does not exist is a missing file.
            raise FileNotFoundError(
                f"YOLO26 weights not found: {model_path}. Download them with "
                f"`yolo predict model=yolo26m.pt` or place the file there."
            )

        logger.info(f"Loading YOLO26 model from {model_path}")
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            raise RuntimeError(
                f"Could not load {model_path}. YOLO26 needs ultralytics >= 8.4 - "
                f"the installed version may be older. Original error: {e}"
            ) from e

        if device == "cuda":
            import torch
            if torch.cuda.is_available():
                self.model.to("cuda")
            else:
                logger.warning("CUDA requested but not available, using CPU")
                self.device = "cpu"
                self.half_precision = False

        logger.info(
            f"YOLO26 loaded on {self.device} "
            f"(img_size={img_size}, half={self.half_precision}, "
            f"end_to_end={self.end_to_end})"
        )

    def apply_preprocessing(self, frame: np.ndarray, config: Dict) -> np.ndarray:
        """Optional contrast work, matching what the other detectors offer."""
        if config.get("equalize_histogram", False) and len(frame.shape) == 3:
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

        if config.get("clahe", False) and len(frame.shape) == 3:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        if config.get("denoise", False):
            frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)

        return frame

    def detect_batch(
        self,
        frames: List[np.ndarray],
        camera_ids: List[str],
        inference_configs: List[Dict],
        preprocessing_configs: Optional[List[Dict]] = None
    ) -> Dict[str, List[Detection]]:
        """
        Detect people across a batch of frames.

        Cameras are grouped by the settings that must be identical within one
        forward pass - image size, thresholds - and each group goes through as
        a single call. Cameras sharing a profile, which is the normal case,
        therefore cost one pass between them rather than one each.
        """
        if not frames:
            return {}

        prepared = []
        for idx, frame in enumerate(frames):
            cfg = preprocessing_configs[idx] if preprocessing_configs else None
            if cfg and cfg.get("use_preprocessing", False):
                frame = self.apply_preprocessing(frame, cfg)
            prepared.append(frame)

        # Group by the settings Ultralytics applies per call, not per image.
        groups: Dict[tuple, List[int]] = {}
        for idx, cfg in enumerate(inference_configs):
            key = (
                cfg.get("confidence_threshold", self.confidence_threshold),
                cfg.get("iou_threshold", self.iou_threshold),
                cfg.get("img_size", self.img_size),
            )
            groups.setdefault(key, []).append(idx)

        detections_by_camera: Dict[str, List[Detection]] = {
            camera_ids[i]: [] for i in range(len(frames))
        }

        for (conf, iou, imgsz), indices in groups.items():
            batch = [prepared[i] for i in indices]
            try:
                results = self._infer(batch, conf, iou, imgsz)
            except Exception as e:
                logger.error(f"YOLO26 inference failed for {len(batch)} frames: {e}")
                continue

            for slot, result in zip(indices, results):
                detections_by_camera[camera_ids[slot]] = self._parse(result)

        return detections_by_camera

    def _infer(self, batch: List[np.ndarray], conf: float, iou: float, imgsz: int):
        """
        One forward pass over a list of frames.

        `nms` is only passed when end-to-end mode is on, so the wrapper stays
        loadable against builds that predate the argument rather than failing
        on an unexpected keyword.
        """
        kwargs = dict(
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=self.device,
            half=self.half_precision,
            classes=[PERSON_CLASS],
            verbose=False,
        )
        if self.end_to_end:
            kwargs["nms"] = False
        return self.model(batch, **kwargs)

    @staticmethod
    def _parse(result) -> List[Detection]:
        """Ultralytics result to the Detection objects the rest of the pipeline expects."""
        detections = []
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return detections

        # Pulled across as whole arrays rather than per-box, which avoids a
        # separate GPU sync for every person in the frame.
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy()

        for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, classes):
            detections.append(Detection(
                bbox=[float(x1), float(y1), float(x2), float(y2)],
                confidence=float(conf),
                class_id=int(cls),
            ))
        return detections

    def detect_single(
        self,
        frame: np.ndarray,
        inference_config: Dict,
        preprocessing_config: Optional[Dict] = None
    ) -> List[Detection]:
        """Detect on one frame."""
        result = self.detect_batch(
            frames=[frame],
            camera_ids=["single"],
            inference_configs=[inference_config],
            preprocessing_configs=[preprocessing_config] if preprocessing_config else None
        )
        return result.get("single", [])
