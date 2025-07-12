import torch
import logging
import cv2
import numpy as np
from pathlib import Path # Added for the __main__ block for testing

# Import CONFIG from main.py for global access
from main import CONFIG

# Import stubs for external detectors and NMS
# These should be replaced with actual implementations or library imports
# Assuming YOLO from ultralytics and FCOSDetector are available as classes
# and nms, postprocess are available.
# For ultralytics YOLO, it's typically `from ultralytics import YOLO`
# For FCOS, it's `from fcos_detector import FCOSDetector`
# For NMS, it's `from nms import nms` (assuming a custom nms module)
# For postprocess, it's `from yolo_fine_tuning import postprocess` (as per previous context)

# Placeholder imports for demonstration. In a real setup, these would be actual library imports.
try:
    from ultralytics import YOLO
except ImportError:
    logging.warning("ultralytics not found. Please install it: pip install ultralytics")
    class YOLO: # Dummy YOLO class for testing if not installed
        def __init__(self, weights, device='cpu'):
            logging.warning(f"Dummy YOLO initialized with weights: {weights} on device: {device}")
            self.weights = weights
            self.device = device
        def eval(self):
            pass
        def to(self, device):
            return self
        def predict(self, img, verbose=False, device='cpu'):
            logging.info(f"Dummy YOLO predict called with image shape: {img.shape}")
            # Return dummy results: [x1, y1, x2, y2, conf, class_id]
            dummy_results = torch.tensor([
                [10, 10, 50, 50, 0.9, 0],
                [15, 15, 55, 55, 0.85, 0],
                [100, 100, 150, 150, 0.7, 1]
            ])
            class DummyResult:
                def __init__(self, data):
                    self.boxes = self
                    self.data = data
            return [DummyResult(dummy_results)]

try:
    from fcos_detector import FCOSDetector
except ImportError:
    logging.warning("fcos_detector not found. Please ensure it's installed or available.")
    class FCOSDetector: # Dummy FCOSDetector class for testing if not installed
        def __init__(self, config):
            logging.warning(f"Dummy FCOSDetector initialized with config: {config}")
            self.config = config
        def load_weights(self, weights):
            logging.warning(f"Dummy FCOSDetector loading weights: {weights}")
            return self
        def eval(self):
            pass
        def to(self, device):
            return self
        def predict(self, img):
            logging.info(f"Dummy FCOSDetector predict called with image shape: {img.shape}")
            # Return dummy results: list of [x1, y1, x2, y2, score, class_id]
            return [
                [12, 12, 52, 52, 0.88, 0],
                [102, 102, 152, 152, 0.72, 1],
                [200, 200, 250, 250, 0.6, 2]
            ]

try:
    from nms import nms
except ImportError:
    logging.warning("nms module not found. Please ensure it's available.")
    # Fallback NMS implementation for testing purposes
    def nms(boxes, iou_threshold):
        """
        A simple, basic NMS implementation for testing purposes.
        Boxes are expected in [x1, y1, x2, y2, score] format.
        """
        if boxes.numel() == 0:
            return []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = boxes[:, 4]

        areas = (x2 - x1) * (y2 - y1)
        _, order = scores.sort(0, descending=True)

        keep = []
        while order.numel() > 0:
            i = order[0]
            keep.append(i.item())

            if order.numel() == 1:
                break

            xx1 = torch.max(x1[i], x1[order[1:]])
            yy1 = torch.max(y1[i], y1[order[1:]])
            xx2 = torch.min(x2[i], x2[order[1:]])
            yy2 = torch.min(y2[i], y2[order[1:]])

            w = torch.max(torch.tensor(0.0), xx2 - xx1)
            h = torch.max(torch.tensor(0.0), yy2 - yy1)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            
            # Find indices where IoU is less than the threshold
            inds = torch.where(ovr <= iou_threshold)[0]
            order = order[inds + 1] # +1 because order[0] was removed

        return keep

# Placeholder for postprocess from yolo_fine_tuning.py
# In a real scenario, this would be imported from your actual yolo_fine_tuning module.
def postprocess(results):
    """
    Dummy postprocess function. In a real scenario, this would convert raw model
    output (e.g., from YOLOv8's Results object) into a standardized format
    [x1, y1, x2, y2, score, class_id].
    """
    processed_preds = []
    if hasattr(results, 'boxes') and hasattr(results.boxes, 'data'):
        # Assuming ultralytics Results object
        for *xyxy, conf, cls in results.boxes.data.cpu().tolist():
            processed_preds.append([*xyxy, conf, cls])
    elif isinstance(results, list):
        # Assuming FCOSDetector or similar returns list of [x1,y1,x2,y2,score,class_id]
        processed_preds.extend(results)
    else:
        logging.warning(f"Unknown result format for postprocess: {type(results)}")
    return processed_preds


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class DetectorEnsemble:
    def __init__(self, device='cuda:0'):
        """
        Initializes the Detector Ensemble by loading multiple detector models.
        Args:
            device (str): The device to load models onto (e.g., 'cuda:0', 'cpu').
        """
        self.models = []
        self.device = device
        
        # Check if ensemble is enabled in CONFIG
        if not CONFIG.get('ensemble', {}).get('enabled', False):
            logger.warning("Detector Ensemble is disabled in CONFIG. Not loading models.")
            return
        
        logger.info(f"Initializing Detector Ensemble on device: {self.device}")
        for d_cfg in CONFIG['ensemble']['detectors']:
            model = None
            try:
                if d_cfg['type'] == 'yolov8':
                    logger.info(f"Loading YOLOv8 model from: {d_cfg['weights']}")
                    model = YOLO(d_cfg['weights'])
                elif d_cfg['type'] == 'fcos':
                    logger.info(f"Loading FCOS model from: {d_cfg['weights']} with config: {d_cfg['config']}")
                    # FCOSDetector stub needs to be a class that can be instantiated and load weights
                    model = FCOSDetector(d_cfg['config'])
                    model.load_weights(d_cfg['weights'])
                else:
                    logger.warning(f"Unsupported detector type in ensemble: {d_cfg['type']}. Skipping.")
                    continue
                
                if model:
                    model.to(self.device).eval()
                    self.models.append(model)
                    logger.info(f"Successfully loaded {d_cfg['type']} model.")
            except Exception as e:
                logger.error(f"Failed to load {d_cfg['type']} model from {d_cfg['weights']}: {e}. Skipping this model.", exc_info=True)
        
        if not self.models:
            logger.warning("No detector models were successfully loaded for the ensemble. Ensemble will not function.")
            CONFIG['ensemble']['enabled'] = False  # Disable ensemble if no models loaded

    @torch.no_grad()
    def detect(self, img_np):
        """
        Performs detection using all models in the ensemble and fuses their predictions.
        Args:
            img_np (np.ndarray): Input image in NumPy array format (HWC, BGR or RGB).
        Returns:
            list: Fused list of detected bounding boxes, each as [x1, y1, x2, y2, score, class_id].
        """
        if not self.models:
            logger.warning("No models in ensemble to perform detection. Returning empty list.")
            return []

        all_preds = []
        logger.info(f"Running inference with {len(self.models)} models in ensemble.")
        for model in self.models:
            try:
                # Ensure input image is RGB if models expect it, or BGR if OpenCV-based
                # Ultralytics YOLO expects RGB, OpenCV reads BGR.
                if isinstance(model, YOLO):  # Assuming Ultralytics YOLO
                    img_input = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                else:  # Assume other models can handle BGR or convert internally
                    img_input = img_np
                
                # Model prediction
                results = model.predict(img_input, verbose=False, device=self.device)
                
                # Process results from each model using the `postprocess` stub
                # The `postprocess` function should convert the model's native output
                # into the standardized [x1, y1, x2, y2, conf, class_id] format.
                if isinstance(results, list): # Ultralytics YOLO returns a list of Results objects
                    for res in results:
                        all_preds.extend(postprocess(res))
                else: # Assuming FCOSDetector or similar returns a single result object/list
                    all_preds.extend(postprocess(results))
                    
            except Exception as e:
                logger.error(f"Error during detection with one ensemble model ({type(model).__name__}): {e}. Skipping its predictions.", exc_info=True)
                continue
        
        if not all_preds:
            logger.info("No predictions from any ensemble model. Returning empty list.")
            return []
        
        logger.info(f"Collected {len(all_preds)} raw predictions from ensemble. Fusing...")
        
        # Filter out predictions that might be malformed or have very low scores
        valid_preds = [p for p in all_preds if len(p) >= 5 and p[4] > 0.01] # Score > 0.01
        if not valid_preds:
            return []
        
        # Convert to a tensor for NMS. Keep class_id for class-aware NMS if needed,
        # but the provided `nms` stub is class-agnostic, expecting [x1, y1, x2, y2, score].
        # So we'll pass only the first 5 elements for NMS.
        boxes_only = torch.tensor([p[:5] for p in valid_preds], device=self.device) # [N, 5] (x1,y1,x2,y2,conf)
        
        # Apply NMS using the stub. It returns indices to keep.
        # Use the iou_threshold from CONFIG['learned_nms'] as a general NMS threshold
        # for ensemble fusion.
        iou_threshold = CONFIG.get('learned_nms', {}).get('iou_threshold', 0.5)
        keep_indices = nms(boxes_only, iou_threshold=iou_threshold)
        
        # Filter the original predictions using the kept indices
        fused_predictions = [valid_preds[i] for i in keep_indices]
        
        logger.info(f"Ensemble fusion completed. Fused {len(fused_predictions)} predictions.")
        return fused_predictions

if __name__ == '__main__':
    # Example usage for testing the ensemble module
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    # Dummy CONFIG for testing
    dummy_config_for_test = {
        'ensemble': {
            'enabled': True,
            'detectors': [
                {'type': 'yolov8', 'weights': 'dummy_yolov8.pt'},  # This file won't exist, will cause error if not stubbed
                {'type': 'fcos', 'weights': 'dummy_fcos.pt', 'config': 'dummy_fcos_config.yaml'} # This file won't exist
            ]
        },
        'learned_nms': {
            'iou_threshold': 0.5  # For the NMS fusion step
        }
    }
    # Temporarily set global CONFIG for testing if it's not already set
    if 'CONFIG' not in globals():
        global CONFIG
        CONFIG = dummy_config_for_test
    else: # If CONFIG exists, update it for the test
        CONFIG.update(dummy_config_for_test)

    print("\n--- Testing DetectorEnsemble ---")
    
    # Create dummy weight files to avoid immediate file not found errors for stubs
    # if you are using the actual libraries, you'd need real weight files.
    Path('dummy_yolov8.pt').touch()
    Path('dummy_fcos.pt').touch()
    Path('dummy_fcos_config.yaml').touch()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    ensemble = DetectorEnsemble(device=device)  # Test on CPU or CUDA if available

    # Create a dummy image (e.g., a black image)
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)

    # Perform detection
    fused_boxes = ensemble.detect(dummy_img)
    print(f"\nFused boxes (from stub models): {fused_boxes}")
    print(f"Number of fused boxes: {len(fused_boxes)}")

    # Clean up dummy files
    Path('dummy_yolov8.pt').unlink()
    Path('dummy_fcos.pt').unlink()
    Path('dummy_fcos_config.yaml').unlink()

    # Test with ensemble disabled
    print("\n--- Testing DetectorEnsemble (Disabled in CONFIG) ---")
    CONFIG['ensemble']['enabled'] = False
    ensemble_disabled = DetectorEnsemble(device=device)
    fused_boxes_disabled = ensemble_disabled.detect(dummy_img)
    print(f"Fused boxes (ensemble disabled): {fused_boxes_disabled}")
    print(f"Number of fused boxes (ensemble disabled): {len(fused_boxes_disabled)}")
