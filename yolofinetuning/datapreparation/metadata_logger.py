import json
import numpy as np

def compute_metadata(image_id, labels):
    areas = [l[3]*l[4] for l in labels]
    return {
        'id': image_id,
        'obj_count': len(labels),
        'avg_bbox_area': float(np.mean(areas) if areas else 0),
        'min_bbox_area': float(np.min(areas) if areas else 0),
        'max_bbox_area': float(np.max(areas) if areas else 0)
    }


def log_metadata(meta, path='metadata_log.jsonl'):
    with open(path, 'a') as f:
        f.write(json.dumps(meta) + "\n")
    # MLflow logging
    try:
        import mlflow
        mlflow.log_dict(meta, artifact_file="metadata.json")
    except ImportError:
        pass
