# Structured logging for detections
import logging
import json
from datetime import datetime

# Optional: MLflow for logging
try:
    import mlflow
except ImportError:
    mlflow = None
    logging.warning("MLflow not installed. MLflow logging will be skipped.")

def setup_logging(cfg=None, log_file: str = None, level: str = 'INFO'):
    """
    Sets up the logging configuration for the application.
    Args:
        cfg (dict, optional): Configuration dictionary. Can contain 'log_file' and 'level'.
        log_file (str, optional): Path to the log file. If None, logs to console.
        level (str, optional): Logging level (e.g., 'INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL').
    """
    try:
        # If config is provided, override defaults
        if cfg is not None:
            log_file = cfg.get('log_file', log_file)
            level = cfg.get('level', level)

        # Basic configuration
        logging.basicConfig(
            filename=log_file,
            level=getattr(logging, level.upper()), # Convert string level to logging constant
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        logging.info("Logging initialized.")
    except Exception as e:
        # Fallback to print if logging setup itself fails
        print(f"[ERROR] Logging setup failed: {e}. Logging to console.")
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s") # Default console logging
        logging.error(f"Original logging setup failed: {e}") # Log the error to new basic logger

def log_mlflow_param(key: str, value):
    """
    Logs a single parameter to MLflow.
    Args:
        key (str): The parameter name.
        value: The parameter value.
    """
    if mlflow:
        try:
            mlflow.log_param(key, value)
            logging.debug(f"MLflow param logged: {key}={value}")
        except Exception as e:
            logging.error(f"[ERROR] MLflow parameter logging failed for '{key}': {e}")
    else:
        logging.debug(f"MLflow not available. Skipping logging param: {key}={value}")

def log_detection(image_path: str, box: list, score: float, label: int, embedding: list = None):
    """
    Logs a single object detection record.
    Args:
        image_path (str): Path to the image where detection occurred.
        box (list): Bounding box coordinates (e.g., [x0, y0, x1, y1]).
        score (float): Confidence score of the detection.
        label (int): Class label ID of the detected object.
        embedding (list, optional): Feature embedding of the detected object.
    """
    try:
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "image": image_path,
            "box": box,
            "score": score,
            "label": label,
            "embedding": embedding if embedding is not None else [] # Ensure embedding is a list
        }
        # Log to standard logger (which might write to file/console)
        logging.info(json.dumps(record))

        # Optionally log to MLflow as a metric or artifact
        if mlflow:
            try:
                # Log individual metrics for easier querying in MLflow UI
                mlflow.log_metric(f"detection_score_image_{os.path.basename(image_path)}", score)
                mlflow.log_metric(f"detection_label_image_{os.path.basename(image_path)}", label)
                # For more complex data like boxes/embeddings, consider logging as an artifact
                # mlflow.log_dict(record, artifact_file=f"detection_record_{os.path.basename(image_path)}_{label}.json")
            except Exception as e:
                logging.error(f"[ERROR] MLflow detection logging failed for {image_path}: {e}")

    except Exception as e:
        logging.error(f"[ERROR] Detection logging failed for {image_path}: {e}")
