# Detection metrics: IoU, mAP, precision, recall
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
import logging # Added for logging errors

def iou_xyxy(boxA, boxB):
    """
    Compute Intersection-over-Union (IoU) for two bounding boxes.
    Boxes are expected in [x0, y0, x1, y1] format (top-left, bottom-right corners).
    Args:
        boxA (list or np.array): Bounding box A.
        boxB (list or np.array): Bounding box B.
    Returns:
        float: IoU value.
    """
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    # Compute the area of both the prediction and ground-truth rectangles
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the union area
    union_area = boxA_area + boxB_area - inter_area

    # Handle case where union_area is zero to avoid division by zero
    return inter_area / union_area if union_area > 0 else 0.0

class Evaluator:
    """
    Evaluator for detection metrics: mAP, IoU, precision, recall.
    Collects ground-truth and prediction data, then computes metrics.
    Usage:
        ev = Evaluator(cfg)
        ev.update(gt_boxes_for_image, pred_boxes_for_image, scores_for_image)
        ev.finalize()
    """
    def __init__(self, cfg):
        """
        Initializes the Evaluator.
        Args:
            cfg (dict): Configuration dictionary, expected to contain 'iou_thresholds'.
        """
        self.iou_thrs = cfg.get('iou_thresholds', [0.5]) # Default to [0.5] if not provided
        self.gt_all = []    # List of ground-truth boxes for all images (each element is a list of boxes for an image)
        self.preds_all = [] # List of (boxes, scores) for all images (each element is a tuple for an image)

    def update(self, gt_boxes, pred_boxes, scores):
        """
        Updates the evaluator with predictions and ground truth for a single image.
        Args:
            gt_boxes (list): List of ground-truth boxes for the current image (e.g., [[x0,y0,x1,y1], ...]).
                             Can be empty if no GT boxes.
            pred_boxes (list): List of predicted boxes for the current image (e.g., [[x0,y0,x1,y1], ...]).
            scores (list): List of confidence scores for predicted boxes.
        """
        self.gt_all.append(gt_boxes if gt_boxes is not None else [])
        self.preds_all.append((pred_boxes if pred_boxes is not None else [], scores if scores is not None else []))

    def finalize(self):
        """
        Computes and prints Average Precision (AP) for each IoU threshold and Mean Average Precision (mAP).
        """
        logging.info("Finalizing evaluation and computing metrics...")
        try:
            aps = []
            for thr in self.iou_thrs:
                all_scores_flat = []
                all_labels_flat = [] # 1 for TP, 0 for FP

                # Process each image's predictions against its ground truths
                for (pred_boxes, pred_scores), gt_boxes in zip(self.preds_all, self.gt_all):
                    # Sort predictions by score in descending order
                    sorted_indices = np.argsort(pred_scores)[::-1]
                    pred_boxes_sorted = [pred_boxes[i] for i in sorted_indices]
                    pred_scores_sorted = [pred_scores[i] for i in sorted_indices]

                    # Keep track of which GT boxes have been matched to avoid double counting
                    matched_gt_flags = [False] * len(gt_boxes)

                    for i, pred_box in enumerate(pred_boxes_sorted):
                        all_scores_flat.append(pred_scores_sorted[i]) # Add the score
                        
                        is_matched_to_gt = False
                        best_iou = 0.0
                        best_gt_idx = -1

                        # Find the best matching GT box for the current prediction
                        for j, gt_box in enumerate(gt_boxes):
                            if not matched_gt_flags[j]: # Only consider unmatched GTs
                                current_iou = iou_xyxy(pred_box, gt_box)
                                if current_iou > best_iou:
                                    best_iou = current_iou
                                    best_gt_idx = j
                        
                        if best_iou >= thr:
                            is_matched_to_gt = True
                            matched_gt_flags[best_gt_idx] = True # Mark GT as matched
                        
                        all_labels_flat.append(int(is_matched_to_gt)) # 1 for TP, 0 for FP

                if not all_labels_flat: # Handle case with no predictions
                    logging.warning(f"No predictions found for IoU threshold {thr}. AP will be 0.")
                    ap = 0.0
                else:
                    # Compute Precision-Recall curve
                    # Note: precision_recall_curve expects actual labels (0 or 1) and prediction scores
                    p, r, _ = precision_recall_curve(all_labels_flat, all_scores_flat)
                    ap = auc(r, p)
                
                aps.append(ap)
                print(f"IoU={thr:.2f}: AP={ap:.4f}")
            
            if self.iou_thrs:
                mAP = np.mean(aps)
                print(f"mAP@{self.iou_thrs}: {mAP:.4f}")
            else:
                print("No IoU thresholds defined for mAP calculation.")

        except Exception as e:
            logging.error(f"[ERROR] Evaluator finalize failed: {e}")
            import traceback
            traceback.print_exc()

# --- CLI entry point ---
if __name__ == "__main__":
    import argparse
    # Configure basic logging for CLI usage
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="Detection metrics utility")
    parser.add_argument('--gt', required=True, help='Path to ground truth label file (YOLO format .txt)')
    parser.add_argument('--pred', required=True, help='Path to prediction label file (YOLO format .txt, with scores)')
    parser.add_argument('--iou_thresholds', type=float, nargs='+', default=[0.5, 0.75], help='IoU thresholds for AP calculation')
    args = parser.parse_args()

    # Helper to convert YOLO format (cx, cy, w, h) to xyxy (x0, y0, x1, y1)
    def yolo_to_xyxy(yolo_box, img_w=1.0, img_h=1.0): # Assuming normalized coordinates, so img_w/h=1.0
        cx, cy, w, h = yolo_box
        x0 = (cx - w/2) * img_w
        y0 = (cy - h/2) * img_h
        x1 = (cx + w/2) * img_w
        y1 = (cy + h/2) * img_h
        return [x0, y0, x1, y1]

    gt_boxes_yolo = []
    try:
        with open(args.gt, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5: # class_id cx cy w h
                    gt_boxes_yolo.append(list(map(float, parts[1:])))
                else:
                    logging.warning(f"Skipping malformed GT line: {line.strip()}")
    except FileNotFoundError:
        logging.error(f"Ground truth file not found: {args.gt}")
        exit(1)
    except Exception as e:
        logging.error(f"Error reading ground truth file: {e}")
        exit(1)

    pred_data = [] # List of (box_yolo, score)
    try:
        with open(args.pred, 'r') as f:
            for line in f:
                parts = line.strip().split()
                # Assuming prediction file format: class_id cx cy w h score
                if len(parts) == 6:
                    pred_data.append((list(map(float, parts[1:5])), float(parts[5])))
                elif len(parts) == 5: # If scores are not in file, assume 1.0
                    pred_data.append((list(map(float, parts[1:5])), 1.0))
                else:
                    logging.warning(f"Skipping malformed prediction line: {line.strip()}")
    except FileNotFoundError:
        logging.error(f"Prediction file not found: {args.pred}")
        exit(1)
    except Exception as e:
        logging.error(f"Error reading prediction file: {e}")
        exit(1)

    # Convert all boxes to xyxy format for the Evaluator
    gt_boxes_xyxy = [yolo_to_xyxy(box) for box in gt_boxes_yolo]
    pred_boxes_xyxy = [yolo_to_xyxy(box_data[0]) for box_data in pred_data]
    scores = [box_data[1] for box_data in pred_data]

    ev = Evaluator({'iou_thresholds': args.iou_thresholds})
    ev.update(gt_boxes_xyxy, pred_boxes_xyxy, scores)
    ev.finalize()
