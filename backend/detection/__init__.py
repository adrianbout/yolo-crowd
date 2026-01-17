"""Detection package"""

from .detector import YOLODetector, Detection, DetectionAggregator
from .thermal_detector import ThermalYOLODetector
from .blob_hotspot_detector import BlobHotspotDetector
from .detector_factory import DetectorFactory
from .roi_filter import ROIFilter

__all__ = [
    "YOLODetector",
    "ThermalYOLODetector",
    "BlobHotspotDetector",
    "DetectorFactory",
    "Detection",
    "DetectionAggregator",
    "ROIFilter"
]
