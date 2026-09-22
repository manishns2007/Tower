"""Script to verify real YOLOv8-Nano inference with YOLODetector."""

import logging
import numpy as np

from inference.detector import YOLODetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("verify_detector")


def main() -> None:
    logger.info("Initializing YOLODetector with 'yolov8n.pt' and device='auto'...")
    try:
        detector = YOLODetector(model_path="yolov8n.pt", device="auto", confidence_threshold=0.25)
    except Exception as exc:
        logger.error("Failed to initialize YOLODetector: %s", exc)
        return

    logger.info("Detector initialized successfully. Active device: %s", detector.device)

    # Generate synthetic RGB test frame (640x480)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add a colored rectangle
    frame[100:300, 200:400] = [200, 100, 50]

    logger.info("Running detection on synthetic frame (shape: %s)...", frame.shape)
    detections = detector.detect(frame)
    logger.info("Inference completed. Number of detections on synthetic frame: %d", len(detections))
    for d in detections:
        logger.info(" - %s (conf: %.2f) at %s, center=%s", d.class_name, d.confidence, d.bbox, d.center)


if __name__ == "__main__":
    main()
