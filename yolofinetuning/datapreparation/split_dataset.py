def split_dataset(label_dir, splits=(0.8, 0.1, 0.1), out_dir='splits'):

import os, json, random
from metadata_logger import compute_metadata, log_metadata
import logging
try:
    import mlflow
except ImportError:
    mlflow = None

def split_dataset(label_dir, splits=(0.8, 0.1, 0.1), out_dir='splits', pseudo_label_func=None, active_learning_func=None, stratified=True, kfolds=None, seed=42):
    """
    Splits dataset into train/val/test or k-folds. Supports stratified splitting by class distribution.
    Args:
        label_dir: directory with label txt files
        splits: tuple, e.g. (0.8,0.1,0.1)
        out_dir: output directory
        pseudo_label_func: optional hook
        active_learning_func: optional hook
        stratified: if True, split by class distribution
        kfolds: if int, perform k-fold cross-validation
        seed: random seed
    """
    import numpy as np
    os.makedirs(out_dir, exist_ok=True)
    files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    random.seed(seed)
    random.shuffle(files)
    # Get main class for each file
    def get_main_class(lblfile):
        counts = {}
        for line in open(os.path.join(label_dir, lblfile)):
            c = int(line.split()[0])
            counts[c] = counts.get(c,0)+1
        return max(counts, key=counts.get) if counts else -1
    y = [get_main_class(f) for f in files]
    if kfolds:
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=kfolds, shuffle=True, random_state=seed)
        for fold, (train_idx, val_idx) in enumerate(skf.split(files, y)):
            for phase, idxs in zip(['train','val'], [train_idx, val_idx]):
                with open(f'{out_dir}/fold{fold}_{phase}.txt','w') as fout:
                    for i in idxs:
                        fname = files[i]
                        img_id = fname.split('.txt')[0]
                        labels = []
                        for line in open(os.path.join(label_dir, fname)):
                            parts = line.strip().split()
                            labels.append(list(map(float, parts)))
                        meta = compute_metadata(img_id, labels)
                        log_metadata(meta)
                        fout.write(img_id + '\n')
                        logging.info(f"Fold {fold} {phase}: {img_id}")
                        if mlflow:
                            mlflow.log_param(f"fold{fold}_{phase}_img", img_id)
        return
    if stratified:
        # Stratified split by class
        from sklearn.model_selection import train_test_split
        train_files, test_files, _, _ = train_test_split(files, y, test_size=1-sum(splits[:2]), stratify=y, random_state=seed)
        val_size = splits[1]/(splits[0]+splits[1])
        train_files, val_files, _, _ = train_test_split(train_files, [y[files.index(f)] for f in train_files], test_size=val_size, stratify=[y[files.index(f)] for f in train_files], random_state=seed)
        split_dict = {'train': train_files, 'val': val_files, 'test': test_files}
    else:
        n = len(files)
        split_idxs = [int(n * splits[0]), int(n * (splits[0]+splits[1]))]
        split_dict = {
            'train': files[:split_idxs[0]],
            'val': files[split_idxs[0]:split_idxs[1]],
            'test': files[split_idxs[1]:]
        }
    for phase, subset in split_dict.items():
        with open(f'{out_dir}/{phase}.txt','w') as fout:
            for fname in subset:
                img_id = fname.split('.txt')[0]
                labels = []
                for line in open(os.path.join(label_dir, fname)):
                    parts = line.strip().split()
                    labels.append(list(map(float, parts)))
                meta = compute_metadata(img_id, labels)
                log_metadata(meta)
                fout.write(img_id + '\n')
                logging.info(f"Split {phase}: {img_id}")
                if mlflow:
                    mlflow.log_param(f"split_{phase}_img", img_id)
    # Pseudo-labeling hook
    if pseudo_label_func is not None:
        pseudo_label_func(label_dir, model=None, out_label_dir=os.path.join(out_dir, "pseudo_labels"))
    # Active learning hook
    if active_learning_func is not None:
        selected = active_learning_func(label_dir, model=None, selection_count=100)
        logging.info(f"Active learning selected: {selected}")
        if mlflow:
            mlflow.log_param("active_learning_selected", selected)
