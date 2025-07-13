
import hashlib
import cv2
from pathlib import Path
from argparse import ArgumentParser

def normalize_sort_checksum(label_dir, img_dir, pad_width=4):
    label_dir = Path(label_dir)
    img_dir   = Path(img_dir)

    # zero-pad filenames
    for img_fp in img_dir.glob("*.jpg"):
        idx = int(img_fp.stem)
        new_img = img_dir / f"{idx:0{pad_width}d}.jpg"
        img_fp.rename(new_img)
        lbl_fp = label_dir / f"{img_fp.stem}.txt"
        if lbl_fp.exists():
            lbl_fp.rename(label_dir / f"{idx:0{pad_width}d}.txt")

    # process labels
    for lbl_fp in label_dir.glob("*.txt"):
        img_fp = img_dir / f"{lbl_fp.stem}.jpg"
        if not img_fp.exists():
            continue
        img_h, img_w = cv2.imread(str(img_fp)).shape[:2]
        lines = []
        for ln in lbl_fp.read_text().splitlines():
            cls, *coords = ln.split()
            coords = [float(c) for c in coords]
            # normalize if absolute
            if any(c > 1.0 for c in coords):
                # x coords at even idx, y at odd
                for i, c in enumerate(coords):
                    coords[i] = c / (img_w if i % 2 == 0 else img_h)
            lines.append((int(cls), coords))

        # sort by class
        lines.sort(key=lambda x: x[0])
        out_txt = "\n".join(
            f"{cls} " + " ".join(f"{c:.6f}" for c in coords)
            for cls, coords in lines
        )
        lbl_fp.write_text(out_txt)

        # checksum
        md5 = hashlib.md5(out_txt.encode()).hexdigest()
        print(f"{lbl_fp.name} → md5 {md5}")

if __name__=="__main__":
    p = ArgumentParser()
    p.add_argument("--label_dir", default="labels/yolo")
    p.add_argument("--img_dir",   default="images/resized")
    p.add_argument("--pad_width", type=int, default=4)
    args = p.parse_args()
    normalize_sort_checksum(args.label_dir, args.img_dir, args.pad_width)