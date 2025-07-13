import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit

def split_dataset(label_dir: str,
                  metadata_fp: str,
                  out_splits_dir: str,
                  splits=(0.8, 0.1, 0.1)):
    label_dir = Path(label_dir)
    out_dir   = Path(out_splits_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata: each line is JSON with fields id, obj_count, rel_edge_count
    metas = [json.loads(l) for l in open(metadata_fp)]
    ids   = [m["id"] for m in metas]
    diff  = np.array([m["obj_count"] + m["rel_edge_count"] for m in metas])

    # Discretize difficulty into 3 bins for stratification
    bins = np.digitize(diff, np.percentile(diff, [33, 66]))

    # First split off test
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=splits[2], random_state=42)
    trainval_idx, test_idx = next(sss1.split(ids, bins))
    trainval_ids = [ids[i] for i in trainval_idx]
    trainval_bins= bins[trainval_idx]
    test_ids     = [ids[i] for i in test_idx]

    # Then split train/val
    val_frac = splits[1] / (splits[0] + splits[1])
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=42)
    tr_idx, val_idx = next(sss2.split(trainval_ids, trainval_bins))
    train_ids = [trainval_ids[i] for i in tr_idx]
    val_ids   = [trainval_ids[i] for i in val_idx]

    # Write out
    for phase, lst in zip(["train","val","test"], [train_ids, val_ids, test_ids]):
        (out_dir / f"{phase}.txt").write_text("\n".join(lst))

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--label_dir",      default="labels/yolo")
    p.add_argument("--metadata_fp",    default="output/metadata.jsonl")
    p.add_argument("--out_splits_dir", default="splits")
    args = p.parse_args()
    split_dataset(args.label_dir, args.metadata_fp, args.out_splits_dir)
