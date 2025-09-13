import re
import shutil
from pathlib import Path

# === EDIT THIS: where raw folders live ===
SRC_IMAGES = Path(r"C:\datasets\CVC-ClinicDB\Original")
SRC_MASKS  = Path(r"C:\datasets\CVC-ClinicDB\Ground Truth")

# === Destination inside project ===
DEST_ROOT = Path(__file__).parent / "datasets" / "CVC-ClinicDB"

# Target split counts
N_TRAIN, N_VAL, N_TEST = 490, 61, 61  # total = 612

def numeric_key(p: Path) -> int:
    """
    Extract numeric part from filename stem, e.g. '12.png' -> 12
    Ensures correct ordering: 1,2,3,...,10,...,100...
    """
    m = re.search(r"\d+", p.stem)
    if not m:
        raise ValueError(f"No number found in filename: {p.name}")
    return int(m.group())

def main():
    # Collect PNGs and sort numerically
    imgs = sorted(SRC_IMAGES.glob("*.png"), key=numeric_key)
    msks = sorted(SRC_MASKS.glob("*.png"),  key=numeric_key)

    # Basic checks
    assert len(imgs) == len(msks), "Images and masks count mismatch."
    total = len(imgs)
    assert total == (N_TRAIN + N_VAL + N_TEST), f"Expected 612 files, found {total}."

    # Ensure 1–1 pairing by numeric id
    img_ids = [numeric_key(p) for p in imgs]
    msk_ids = [numeric_key(p) for p in msks]
    assert img_ids == msk_ids, "Image/mask numeric IDs do not match."


    expected = list(range(1, total + 1))
    if img_ids != expected:
        print("Warning: filenames are numeric but not exactly 1..612; proceeding by sorted order.")

    # Compute slices by order
    train_slice = slice(0, N_TRAIN)                      # 1..490
    val_slice   = slice(N_TRAIN, N_TRAIN + N_VAL)        # 491..551
    test_slice  = slice(N_TRAIN + N_VAL, total)          # 552..612

    splits = {
        "train": train_slice,
        "val":   val_slice,
        "test":  test_slice,
    }

    # Make destination dirs
    for split in splits:
        (DEST_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (DEST_ROOT / "masks"  / split).mkdir(parents=True, exist_ok=True)

    # Copy pairs
    def copy_range(rng: slice, split: str):
        for img, msk in zip(imgs[rng], msks[rng]):
            shutil.copy2(img, DEST_ROOT / "images" / split / img.name)
            shutil.copy2(msk, DEST_ROOT / "masks"  / split / msk.name)

    copy_range(train_slice, "train")
    copy_range(val_slice,   "val")
    copy_range(test_slice,  "test")

    # Report
    def count(p): return len(list(p.glob("*.png")))
    print("Done.")
    print("images/train:", count(DEST_ROOT / "images" / "train"),
          "val:", count(DEST_ROOT / "images" / "val"),
          "test:", count(DEST_ROOT / "images" / "test"))
    print("masks/train:",  count(DEST_ROOT / "masks" / "train"),
          "val:", count(DEST_ROOT / "masks" / "val"),
          "test:", count(DEST_ROOT / "masks" / "test"))

if __name__ == "__main__":
    main()