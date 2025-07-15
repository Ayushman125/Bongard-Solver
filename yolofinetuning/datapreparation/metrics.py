# Detection metrics: IoU, mAP, precision, recall
import numpy as np
from sklearn.metrics import precision_recall_curve, auc

def iou(boxA, boxB):
    # box: [x0,y0,x1,y1]
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    boxA_area = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    boxB_area = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return inter / (boxA_area + boxB_area - inter + 1e-9)

class Evaluator:
    def __init__(self, cfg):
        self.iou_thrs = cfg['iou_thresholds']
        self.gt = []  # ground-truth boxes per image
        self.preds = []  # (boxes, scores) per image

    def update(self, gt_boxes, pred_boxes, scores):
        self.gt.append(gt_boxes or [])
        self.preds.append((pred_boxes, scores))

    def finalize(self):
        aps = []
        for thr in self.iou_thrs:
            precisions, recalls = [], []
            all_scores, all_labels = [], []
            # Flatten predictions
            for (boxes, scores), gt in zip(self.preds, self.gt):
                tp, fp = [], []
                for box, score in zip(boxes, scores):
                    matched = any(iou(box, g) >= thr for g in gt)
                    tp.append(int(matched)); fp.append(int(not matched))
                all_scores.extend(scores); 
                all_labels.extend(tp)
            # Compute Precision-Recall curve
            p, r, _ = precision_recall_curve(all_labels, all_scores)
            ap = auc(r, p)
            aps.append(ap)
            print(f"IoU={thr}: AP={ap:.4f}")
        print(f"mAP@{self.iou_thrs}: {np.mean(aps):.4f}")
