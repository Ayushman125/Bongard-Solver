import hashlib
from pathlib import Path

def normalize_sort_chk(label_dir):
    for fp in Path(label_dir).glob("*.txt"):
        lines = []
        for ln in fp.read_text().splitlines():
            cls, *coords = ln.split()
            coords = [float(c) for c in coords]
            # detect absolute coords >1, assume image 1000px
            if any(c>1 for c in coords):
                coords = [c/1000 for c in coords]
            lines.append((int(cls), coords))
        # sort
        lines.sort(key=lambda x: x[0])
        out = "\n".join(f"{c} " + " ".join(f"{v:.6f}" for v in coords) 
                        for c,coords in lines)
        fp.write_text(out)
        md5 = hashlib.md5(out.encode()).hexdigest()
        print(f"{fp.name} → checksum {md5}")

if __name__=="__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--label_dir", default="labels/yolo")
    args = p.parse_args()
    normalize_sort_chk(args.label_dir)