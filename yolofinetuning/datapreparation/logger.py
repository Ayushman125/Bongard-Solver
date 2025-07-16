import logging
import os
import json # For logging structured data

# Optional: MLflow for experiment tracking
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow not installed. MLflow logging will be skipped.")

def setup_logging(config: dict = None):
    """
    Sets up the logging configuration for the entire application.
    Args:
        config (dict, optional): A dictionary containing logging configuration.
                                 Expected keys: 'level' (e.g., 'INFO', 'DEBUG'),
                                 'log_file' (path to log file), 'console_output' (bool).
    """
    if config is None:
        config = {}

    log_level_str = config.get('level', 'INFO').upper()
    log_file = config.get('log_file')
    console_output = config.get('console_output', True)

    log_level = getattr(logging, log_level_str, logging.INFO)

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers to prevent duplicate logs if called multiple times
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        try:
            # Ensure the directory for the log file exists
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            logging.info(f"Logging output will also be saved to: {log_file}")
        except Exception as e:
            logging.error(f"Could not set up file logging to {log_file}: {e}")

    logging.info(f"Logging configured with level: {log_level_str}")

def log_detection(image_id: str, detections: list, log_level=logging.INFO):
    """
    Logs detection results for a single image in a structured format.
    Args:
        image_id (str): The ID of the image.
        detections (list): A list of dictionaries, where each dictionary represents a detection.
                           Example: [{'box': [x1, y1, x2, y2], 'score': 0.9, 'class': 'car'}, ...]
        log_level: The logging level (e.g., logging.INFO, logging.DEBUG).
    """
    log_entry = {
        "event": "detection_log",
        "image_id": image_id,
        "num_detections": len(detections),
        "detections": detections
    }
    # Log as a JSON string to keep it structured in logs
    logging.log(log_level, json.dumps(log_entry))

def log_mlflow_param(key: str, value):
    """
    Logs a parameter to MLflow if MLflow is available and an active run exists.
    Args:
        key (str): The name of the parameter.
        value: The value of the parameter.
    """
    if MLFLOW_AVAILABLE:
        try:
            mlflow.log_param(key, value)
            logging.debug(f"Logged MLflow parameter: {key}={value}")
        except Exception as e:
            logging.warning(f"Failed to log MLflow parameter '{key}': {e}. Is an MLflow run active?")
    else:
        logging.debug(f"MLflow not available. Skipping logging parameter: {key}")

def log_mlflow_metric(key: str, value: float, step: int = None):
    """
    Logs a metric to MLflow if MLflow is available and an active run exists.
    Args:
        key (str): The name of the metric.
        value (float): The value of the metric.
        step (int, optional): The step number for the metric (e.g., epoch, iteration).
    """
    if MLFLOW_AVAILABLE:
        try:
            mlflow.log_metric(key, value, step=step)
            logging.debug(f"Logged MLflow metric: {key}={value} (step={step})")
        except Exception as e:
            logging.warning(f"Failed to log MLflow metric '{key}': {e}. Is an MLflow run active?")
    else:
        logging.debug(f"MLflow not available. Skipping logging metric: {key}")

def log_mlflow_artifact(local_path: str, artifact_path: str = None):
    """
    Logs a file or directory as an artifact to MLflow.
    Args:
        local_path (str): The path to the file or directory to log.
        artifact_path (str, optional): The run-relative artifact path to which to log the artifact.
                                       If None, data is logged to the root of the run's artifact directory.
    """
    if MLFLOW_AVAILABLE:
        try:
            if os.path.isdir(local_path):
                mlflow.log_artifact(local_path, artifact_path)
                logging.info(f"Logged directory '{local_path}' to MLflow artifacts at '{artifact_path}'")
            elif os.path.isfile(local_path):
                mlflow.log_artifact(local_path, artifact_path)
                logging.info(f"Logged file '{local_path}' to MLflow artifacts at '{artifact_path}'")
            else:
                logging.warning(f"Cannot log '{local_path}' as MLflow artifact: Not a file or directory.")
        except Exception as e:
            logging.warning(f"Failed to log MLflow artifact '{local_path}': {e}. Is an MLflow run active?")
    else:
        logging.debug(f"MLflow not available. Skipping logging artifact: {local_path}")


# Example Usage (for testing this module directly)
if __name__ == "__main__":
    # Setup logging with a file and console output
    log_config = {
        'level': 'DEBUG',
        'log_file': 'app_debug.log',
        'console_output': True
    }
    setup_logging(log_config)

    logging.info("This is an INFO message.")
    logging.debug("This is a DEBUG message.")
    logging.warning("This is a WARNING message.")
    logging.error("This is an ERROR message.")
    logging.critical("This is a CRITICAL message.")

    # Example of structured detection logging
    sample_detections = [
        {'box': [10, 20, 50, 60], 'score': 0.95, 'class': 'car'},
        {'box': [100, 120, 150, 180], 'score': 0.88, 'class': 'person'}
    ]
    log_detection("image_001", sample_detections)

    # Example of MLflow logging (requires MLflow to be installed and potentially a tracking server)
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri("file:///tmp/mlruns") # Use a local file-based tracking URI for testing
            mlflow.set_experiment("logger_module_test")
            with mlflow.start_run(run_name="test_run_from_logger"):
                log_mlflow_param("learning_rate", 0.001)
                log_mlflow_metric("loss", 0.5, step=0)
                log_mlflow_metric("loss", 0.4, step=1)
                log_mlflow_metric("accuracy", 0.92, step=1)

                # Create a dummy file to log as artifact
                with open("dummy_artifact.txt", "w") as f:
                    f.write("This is a test artifact.")
                log_mlflow_artifact("dummy_artifact.txt", "my_artifacts/test.txt")
                os.remove("dummy_artifact.txt") # Clean up dummy file

                # Create a dummy directory to log as artifact
                os.makedirs("dummy_dir_artifact", exist_ok=True)
                with open("dummy_dir_artifact/file1.txt", "w") as f:
                    f.write("Content of file1.")
                log_mlflow_artifact("dummy_dir_artifact", "my_artifacts/test_dir")
                shutil.rmtree("dummy_dir_artifact") # Clean up dummy directory

            logging.info("MLflow logging test completed. Check your MLflow UI.")
        except Exception as e:
            logging.error(f"Error during MLflow test in logger.py: {e}")
    else:
        logging.info("MLflow is not installed, skipping MLflow logging examples.")

    # Clean up the created log file for direct module testing
    if os.path.exists('app_debug.log'):
        os.remove('app_debug.log')
