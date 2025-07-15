# Structured logging for detections
import logging
import json
from datetime import datetime

def setup_logging(cfg):
    logging.basicConfig(
        filename=cfg['log_file'],
        level=getattr(logging, cfg['level']),
        format="%(message)s"
    )

def log_detection(image_path, box, score, label, embedding):
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "image": image_path,
        "box": box,
        "score": score,
        "label": label,
        "embedding": embedding
    }
    logging.info(json.dumps(record))
