import hashlib
from pathlib import Path

def normalize_and_sort(label_dir: str, image_dir: str, pad_width=4):
    label_dir = Path(label_dir)
    image_dir = Path(image_dir)

    # 1) Zero‐pad filenames
    for img_fp in image_dir.glob("*.jpg"):
        idx = int(img_fp.stem)
        new_name = f"{idx:0{pad_width}d}.jpg"
        img_fp.rename(image_dir / new_name)

        lbl_old = label_dir / f"{img_fp.stem}.txt"
        if lbl_old.exists():
            lbl_old.rename(label_dir / f"{idx:0{pad_width}d}.txt")

    # 2) Sort and scale coords, compute checksum
    for lbl_fp in label_dir.glob("*.txt"):
        lines = [l.strip() for l in lbl_fp.read_text().splitlines() if l.strip()]
        entries = []
        for ln in lines:
            cls, *coords = ln.split()
            coords = list(map(float, coords))
            # if coords >1, assume absolute and normalize by 1000
            if max(coords) > 1:
                coords = [c / 1000 for c in coords]
            entries.append((int(cls), coords))

        # sort by class_id
        entries.sort(key=lambda x: x[0])
        out_lines = [f"{cls} {' '.join(f'{c:.6f}' for c in coords)}"
                     for cls, coords in entries]
        lbl_fp.write_text("\n".join(out_lines))

        # checksum
        md5 = hashlib.md5(lbl_fp.read_bytes()).hexdigest()
        print(f"{lbl_fp.name}: checksum {md5}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--label_dir", default="labels/yolo")
    p.add_argument("--image_dir", default="images/resized")
    args = p.parse_args()
    normalize_and_sort(args.label_dir, args.image_dir)
