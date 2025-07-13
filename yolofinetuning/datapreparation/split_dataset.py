import os, json, random
from metadata_logger import compute_metadata, log_metadata

def split_dataset(label_dir, splits=(0.8, 0.1, 0.1), out_dir='splits'):
    os.makedirs(out_dir, exist_ok=True)
    files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    random.shuffle(files)
    n = len(files)
    split_idxs = [int(n * splits[0]), int(n * (splits[0]+splits[1]))]

    for phase, subset in zip(['train','val','test'], [
        files[:split_idxs[0]],
        files[split_idxs[0]:split_idxs[1]],
        files[split_idxs[1]:]
    ]):
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
