import numpy as np
import logging
from collections import defaultdict # For better handling of precision/recall accumulation

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def iou_xyxy(bbox1: list, bbox2: list) -> float:
    """
    Compute Intersection-over-Union (IoU) for two bounding boxes.
    Boxes are expected in [x0, y0, x1, y1] format (top-left, bottom-right corners).
    Args:
        bbox1 (list): Bounding box 1 [x0, y0, x1, y1].
        bbox2 (list): Bounding box 2 [x0, y0, x1, y1].
    Returns:
        float: IoU value, or 0.0 if union area is zero.
    """
    # Determine the coordinates of the intersection rectangle
    xA = max(bbox1[0], bbox2[0])
    yA = max(bbox1[1], bbox2[1])
    xB = min(bbox1[2], bbox2[2])
    yB = min(bbox1[3], bbox2[3])

    # Compute the area of intersection rectangle
    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    # Compute the area of both bounding boxes
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # Compute the union area
    union_area = bbox1_area + bbox2_area - inter_area

    # Return the IoU
    return inter_area / union_area if union_area > 0 else 0.0

def _yolo_to_xyxy(bbox_yolo: list, img_w: int, img_h: int) -> list:
    """
    Converts a YOLO format bounding box [cx, cy, w, h] to absolute pixel [x0, y0, x1, y1].
    Args:
        bbox_yolo (list): Bounding box in YOLO format [cx, cy, w, h] (normalized).
        img_w (int): Image width.
        img_h (int): Image height.
    Returns:
        list: Bounding box in [x0, y0, x1, y1] format (absolute pixels).
    """
    cx, cy, w, h = bbox_yolo
    x0 = (cx - w/2) * img_w
    y0 = (cy - h/2) * img_h
    x1 = (cx + w/2) * img_w
    y1 = (cy + h/2) * img_h
    return [x0, y0, x1, y1]


class Evaluator:
    """
    A class to evaluate object detection metrics such as Mean Average Precision (mAP),
    Precision, and Recall. This implementation calculates mAP for multiple IoU thresholds
    and averages them, similar to COCO mAP.
    """
    def __init__(self, num_classes: int, iou_thresholds: list = None):
        """
        Initializes the Evaluator.
        Args:
            num_classes (int): The total number of classes in the dataset.
            iou_thresholds (list, optional): A list of IoU thresholds for mAP calculation.
                                             Defaults to [0.5, 0.75] if None.
        """
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds if iou_thresholds is not None else [0.5, 0.75]
        
        # Store all predictions and ground truths for batch processing
        # Format: {'image_id': {'preds': [{'bbox': [x0,y0,x1,y1], 'score': s, 'class_id': c}],
        #                       'gts': [{'bbox': [x0,y0,x1,y1], 'class_id': c, 'matched': False}]}}
        self.data = defaultdict(lambda: {'preds': [], 'gts': []})
        self.image_dims = {} # Store image dimensions: {'image_id': (width, height)}
        logging.info(f"Evaluator initialized for {num_classes} classes with IoU thresholds: {self.iou_thresholds}")

    def add_predictions(self, image_id: str, image_width: int, image_height: int,
                        predictions: list, ground_truths: list):
        """
        Adds predictions and ground truths for a single image.
        Args:
            image_id (str): Unique identifier for the image.
            image_width (int): Width of the image.
            image_height (int): Height of the image.
            predictions (list): List of dictionaries for predictions.
                                Each dict: {'bbox': [x0,y0,x1,y1], 'score': s, 'class_id': c}.
                                Bboxes are expected in absolute pixel [x0,y0,x1,y1] format.
            ground_truths (list): List of dictionaries for ground truths.
                                  Each dict: {'bbox': [cx,cy,w,h], 'class_id': c}.
                                  Bboxes are expected in normalized YOLO [cx,cy,w,h] format.
        """
        self.image_dims[image_id] = (image_width, image_height)
        self.data[image_id]['preds'].extend(predictions)
        
        # Convert ground truths from YOLO to XYXY for consistency
        converted_gts = []
        for gt in ground_truths:
            bbox_xyxy = _yolo_to_xyxy(gt['bbox'], image_width, image_height)
            converted_gts.append({'bbox': bbox_xyxy, 'class_id': gt['class_id'], 'matched': False})
        self.data[image_id]['gts'].extend(converted_gts)
        logging.debug(f"Added {len(predictions)} preds and {len(ground_truths)} GTs for {image_id}")

    def _compute_ap(self, recall: np.ndarray, precision: np.ndarray) -> float:
        """
        Computes Average Precision (AP) for a given Precision-Recall curve.
        Uses the 11-point interpolation method.
        Args:
            recall (np.ndarray): Array of recall values.
            precision (np.ndarray): Array of precision values.
        Returns:
            float: Average Precision.
        """
        # Sort by recall (ascending)
        order = np.argsort(recall)
        recall = recall[order]
        precision = precision[order]

        # Pad with (0,0) and (1,0) for interpolation
        recall = np.concatenate(([0.], recall, [1.]))
        precision = np.concatenate(([0.], precision, [0.]))

        # Interpolate precision values to ensure non-increasing property
        for i in range(precision.size - 1, 0, -1):
            precision[i - 1] = np.maximum(precision[i - 1], precision[i])

        # Compute AP by integrating the PR curve
        i = np.where(recall[1:] != recall[:-1])[0] # Find unique recall values
        ap = np.sum((recall[i + 1] - recall[i]) * precision[i + 1])
        return ap

    def evaluate(self) -> dict:
        """
        Computes mAP, precision, and recall across all added data.
        Returns:
            dict: A dictionary containing 'mAP', 'precision', 'recall', and 'ap_per_class_iou_thr'.
        """
        if not self.data:
            logging.warning("No data added to evaluator. Returning empty metrics.")
            return {'mAP': 0.0, 'precision': 0.0, 'recall': 0.0, 'ap_per_class_iou_thr': {}}

        all_aps = []
        ap_per_class_iou_thr = defaultdict(dict) # {class_id: {iou_thr: ap}}

        for iou_thr in self.iou_thresholds:
            logging.info(f"Evaluating for IoU threshold: {iou_thr}")
            
            # Initialize lists to accumulate true positives, false positives, and ground truths
            # for PR curve calculation across all classes and images for this IoU threshold
            all_preds_for_pr = [] # [{'score': s, 'is_tp': True/False, 'class_id': c}]
            all_gts_count = defaultdict(int) # {class_id: count}

            # Reset matched status for GTs for each IoU threshold evaluation
            for img_id in self.data:
                for gt in self.data[img_id]['gts']:
                    gt['matched'] = False
                    all_gts_count[gt['class_id']] += 1

            # Process predictions for each image
            for img_id, img_data in self.data.items():
                preds = img_data['preds']
                gts = img_data['gts'][:] # Create a copy to modify 'matched' status

                # Sort predictions by confidence score in descending order
                preds.sort(key=lambda x: x['score'], reverse=True)

                for pred in preds:
                    best_iou = 0.0
                    best_gt_idx = -1

                    # Find the best matching ground truth for the current prediction
                    for gt_idx, gt in enumerate(gts):
                        if gt['class_id'] == pred['class_id'] and not gt['matched']:
                            current_iou = iou_xyxy(pred['bbox'], gt['bbox'])
                            if current_iou > best_iou:
                                best_iou = current_iou
                                best_gt_idx = gt_idx
                    
                    # Determine if it's a True Positive (TP) or False Positive (FP)
                    if best_iou >= iou_thr:
                        all_preds_for_pr.append({'score': pred['score'], 'is_tp': True, 'class_id': pred['class_id']})
                        gts[best_gt_idx]['matched'] = True # Mark GT as matched
                    else:
                        all_preds_for_pr.append({'score': pred['score'], 'is_tp': False, 'class_id': pred['class_id']})
            
            # Calculate AP for each class at this IoU threshold
            for class_id in range(self.num_classes):
                class_preds = [p for p in all_preds_for_pr if p['class_id'] == class_id]
                class_preds.sort(key=lambda x: x['score'], reverse=True) # Ensure sorted by score

                tp_cumsum = np.cumsum([1 if p['is_tp'] else 0 for p in class_preds])
                fp_cumsum = np.cumsum([1 if not p['is_tp'] else 0 for p in class_preds])

                num_gts = all_gts_count[class_id]
                if num_gts == 0:
                    ap_per_class_iou_thr[class_id][iou_thr] = 0.0
                    continue

                precision = tp_cumsum / (tp_cumsum + fp_cumsum + np.finfo(float).eps) # Add epsilon to avoid div by zero
                recall = tp_cumsum / num_gts

                ap = self._compute_ap(recall, precision)
                ap_per_class_iou_thr[class_id][iou_thr] = ap
                logging.debug(f"  Class {class_id} @ IoU {iou_thr}: AP = {ap:.4f}")
                
                # If we need global precision/recall, we can accumulate here
                # For mAP, we average APs.

        # Calculate mAP (average over all classes and all IoU thresholds)
        total_aps_sum = 0.0
        total_ap_count = 0
        for class_id in ap_per_class_iou_thr:
            for iou_thr in ap_per_class_iou_thr[class_id]:
                total_aps_sum += ap_per_class_iou_thr[class_id][iou_thr]
                total_ap_count += 1
        
        mAP = total_aps_sum / total_ap_count if total_ap_count > 0 else 0.0
        logging.info(f"Calculated mAP: {mAP:.4f}")

        # For overall precision and recall, we can re-run with a single IoU threshold (e.g., 0.5)
        # or aggregate all true positives and false positives globally.
        # This is a simplified overall precision/recall, often calculated at a specific IoU threshold.
        # Let's calculate overall P/R at IoU=0.5 for simplicity.
        overall_tp = 0
        overall_fp = 0
        overall_fn = 0 # False Negatives = total GTs - total TPs

        # Re-evaluate for a single threshold (e.g., 0.5) to get overall P/R
        if 0.5 in self.iou_thresholds:
            iou_for_overall_pr = 0.5
        else:
            iou_for_overall_pr = self.iou_thresholds[0] if self.iou_thresholds else 0.5 # Fallback

        # Reset matched status for GTs
        for img_id in self.data:
            for gt in self.data[img_id]['gts']:
                gt['matched'] = False
        
        total_gts_overall = sum(len(d['gts']) for d in self.data.values())
        total_tps_overall = 0

        for img_id, img_data in self.data.items():
            preds = img_data['preds']
            gts = img_data['gts'][:] # Copy for this specific P/R calculation

            preds.sort(key=lambda x: x['score'], reverse=True)

            for pred in preds:
                best_iou = 0.0
                best_gt_idx = -1
                for gt_idx, gt in enumerate(gts):
                    if gt['class_id'] == pred['class_id'] and not gt['matched']:
                        current_iou = iou_xyxy(pred['bbox'], gt['bbox'])
                        if current_iou > best_iou:
                            best_iou = current_iou
                            best_gt_idx = gt_idx
                
                if best_iou >= iou_for_overall_pr:
                    overall_tp += 1
                    total_tps_overall += 1
                    gts[best_gt_idx]['matched'] = True
                else:
                    overall_fp += 1
        
        overall_fn = total_gts_overall - total_tps_overall

        overall_precision = overall_tp / (overall_tp + overall_fp + np.finfo(float).eps)
        overall_recall = overall_tp / (total_gts_overall + np.finfo(float).eps)

        logging.info(f"Overall Precision @ IoU={iou_for_overall_pr}: {overall_precision:.4f}")
        logging.info(f"Overall Recall @ IoU={iou_for_overall_pr}: {overall_recall:.4f}")

        return {
            'mAP': mAP,
            'precision': overall_precision,
            'recall': overall_recall,
            'ap_per_class_iou_thr': dict(ap_per_class_iou_thr) # Convert defaultdict to dict
        }

    def reset(self):
        """Resets the evaluator to clear all accumulated data."""
        self.data = defaultdict(lambda: {'preds': [], 'gts': []})
        self.image_dims = {}
        logging.info("Evaluator reset.")

# Example Usage (for testing this module directly)
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    evaluator = Evaluator(num_classes=2, iou_thresholds=[0.5, 0.75])

    # Image 1: Simple case
    img1_id = "img_001"
    img1_w, img1_h = 640, 480
    preds1 = [
        {'bbox': [10, 10, 100, 100], 'score': 0.9, 'class_id': 0}, # TP
        {'bbox': [150, 150, 250, 250], 'score': 0.8, 'class_id': 1}, # TP
        {'bbox': [300, 300, 400, 400], 'score': 0.4, 'class_id': 0}, # FP (low score)
        {'bbox': [500, 500, 600, 600], 'score': 0.7, 'class_id': 1}  # FP (no GT)
    ]
    gts1 = [
        {'bbox': [0.05, 0.05, 0.15, 0.15], 'class_id': 0}, # Corresponds to first pred
        {'bbox': [0.3, 0.3, 0.2, 0.2], 'class_id': 1}      # Corresponds to second pred
    ]
    evaluator.add_predictions(img1_id, img1_w, img1_h, preds1, gts1)

    # Image 2: More complex case with FNs and FPs
    img2_id = "img_002"
    img2_w, img2_h = 800, 600
    preds2 = [
        {'bbox': [50, 50, 150, 150], 'score': 0.95, 'class_id': 0}, # TP
        {'bbox': [200, 200, 300, 300], 'score': 0.85, 'class_id': 0}, # FP (no GT)
        {'bbox': [400, 400, 500, 500], 'score': 0.7, 'class_id': 1}  # TP
    ]
    gts2 = [
        {'bbox': [0.06, 0.06, 0.13, 0.13], 'class_id': 0}, # Corresponds to first pred
        {'bbox': [0.5, 0.5, 0.1, 0.1], 'class_id': 1},     # FN (no pred)
        {'bbox': [0.5, 0.5, 0.1, 0.1], 'class_id': 1}      # Corresponds to third pred
    ]
    evaluator.add_predictions(img2_id, img2_w, img2_h, preds2, gts2)


    results = evaluator.evaluate()
    print("\n--- Evaluation Results ---")
    print(f"mAP: {results['mAP']:.4f}")
    print(f"Overall Precision: {results['precision']:.4f}")
    print(f"Overall Recall: {results['recall']:.4f}")
    print("AP per Class and IoU Threshold:")
    for cls_id, iou_aps in results['ap_per_class_iou_thr'].items():
        for iou_thr, ap_val in iou_aps.items():
            print(f"  Class {cls_id} @ IoU {iou_thr}: {ap_val:.4f}")

    evaluator.reset()
    print("\nEvaluator reset. Data cleared.")
