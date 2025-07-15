# Structured logging for detections
import logging
import json
from datetime import datetime


def setup_logging(cfg=None, log_file=None, level='INFO'):
    if cfg is not None:
        log_file = cfg.get('log_file', log_file)
        level = cfg.get('level', level)
    logging.basicConfig(
        filename=log_file,
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.info("Logging initialized.")

# MLflow logging utility
def log_mlflow_param(key, value):
    try:
        import mlflow
        mlflow.log_param(key, value)
    except ImportError:
        pass

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
