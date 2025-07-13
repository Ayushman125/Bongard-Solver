import os
import sys
import csv
import time
from datetime import datetime, timedelta
from pathlib import Path

# Optional safe‐delete to Recycle Bin
try:
    from send2trash import send2trash
    TRASH = True
except ImportError:
    TRASH = False

# ————————————— Configuration —————————————
DRIVE_TO_SCAN     = "C:\\"                  # Only C:\
NOT_USED_DAYS     = 180                     # Files not accessed in this many days
SKIP_DIR_NAMES    = {                        # Top‐level folders to never enter
    "windows", "program files", "program files (x86)"
}
REPORT_CSV        = "system_cleanup_audit.csv"
# ————————————————————————————————————————————

def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}")

def is_old(path_stat):
    """Return True if last access > NOT_USED_DAYS ago."""
    return datetime.now() - datetime.fromtimestamp(path_stat.st_atime) > timedelta(days=NOT_USED_DAYS)

def scan_c_drive():
    """Walk C:\\, skip system dirs, collect 'not in use' files."""
    log(f"Scanning {DRIVE_TO_SCAN} for files not accessed in >{NOT_USED_DAYS} days…")
    candidates = []
    for root, dirs, files in os.walk(DRIVE_TO_SCAN, topdown=True):
        # Skip system folders by name
        dirs[:] = [d for d in dirs if d.lower() not in SKIP_DIR_NAMES]
        for name in files:
            full = os.path.join(root, name)
            try:
                st = Path(full).stat()
            except Exception:
                continue
            if is_old(st):
                candidates.append({
                    "path": full,
                    "size_bytes": st.st_size,
                    "last_access": datetime.fromtimestamp(st.st_atime)
                })
    log(f"Found {len(candidates)} files not used in >{NOT_USED_DAYS} days.")
    return candidates

def export_report(cands):
    log(f"Writing audit report to {REPORT_CSV}…")
    with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Path","Size_MB","LastAccess"])
        for c in cands:
            w.writerow([
                c["path"],
                round(c["size_bytes"]/1_048_576, 2),
                c["last_access"].strftime("%Y-%m-%d %H:%M:%S")
            ])
    log("Report complete.")

def confirm_and_delete(cands):
    if not cands:
        log("No files to delete.")
        return
    ans = input("\n🚨 Delete ALL flagged files? (y/n): ").strip().lower()
    if ans != "y":
        log("Aborting deletion. Review the CSV and rerun if desired.")
        return

    log("Deleting files now…")
    for c in cands:
        try:
            if TRASH:
                send2trash(c["path"])
            else:
                os.remove(c["path"])
            print("Deleted:", c["path"])
        except Exception as e:
            print("Failed:", c["path"], "–", e)
    log("Deletion pass complete.")

def main():
    start = time.time()
    candidates = scan_c_drive()
    export_report(candidates)
    confirm_and_delete(candidates)
    log(f"Total runtime: {time.time() - start:.1f}s")

if __name__ == "__main__":
    main()
