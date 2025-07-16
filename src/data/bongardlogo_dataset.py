import zipfile
import gdown

# Utility to download and extract Bongard-LOGO dataset if not present
def ensure_bongardlogo_dataset(root_dir: str = 'data/Bongard-LOGO/data', zip_url: str = 'https://drive.google.com/uc?id=1-1j7EBriRpxI-xIVqE6UEXt-SzoWvwLx'):
    """
    Download and extract the Bongard-LOGO dataset from Google Drive if not already present.
    Args:
        root_dir (str): Target directory for extracted dataset.
        zip_url (str): Direct download URL for the dataset zip file.
    """
    if os.path.exists(root_dir) and os.path.isdir(root_dir) and len(os.listdir(root_dir)) > 0:
        print(f"Bongard-LOGO dataset already present at {root_dir}.")
        return
    os.makedirs(os.path.dirname(root_dir), exist_ok=True)
    zip_path = os.path.join(os.path.dirname(root_dir), 'bongardlogo_dataset.zip')
    if not os.path.exists(zip_path):
        print(f"Downloading Bongard-LOGO dataset to {zip_path}...")
        gdown.download(zip_url, zip_path, quiet=False)
    else:
        print(f"Found existing zip at {zip_path}.")
    print(f"Extracting Bongard-LOGO dataset to {root_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(root_dir))
    print("Extraction complete.")
# Folder: bongard_solver/data/
# File: bongardlogo_dataset.py


import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

class BongardLogoDataset(Dataset):
    """
    Dataset for the B/W Bongard-LOGO problems.
    Each problem contains 6 positive and 6 negative images.
    """
    def __init__(self, root_dir: str, split: str = "train", img_size: int = 128):
        # Ensure dataset is present (download/unzip if needed)
        ensure_bongardlogo_dataset(root_dir)
        """
        Initializes the BongardLogoDataset.

        Args:
            root_dir (str): The root directory of the Bongard-LOGO dataset.
                            Expected structure: root_dir/{split}/problem_XXXX/images/{pos|neg}/
            split (str): The dataset split to use ('train', 'val', 'test').
            img_size (int): The target size for the images (img_size x img_size).
        """

        # For ShapeBongard_V2 structure: root_dir/ShapeBongard_V2/{hd,bd,ff}/.../images/*/{0,1}/*.png
        self.root = root_dir
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.samples = []
        # Find the ShapeBongard_V2 directory
        shape_bongard_root = os.path.join(self.root, "ShapeBongard_V2")
        if not os.path.isdir(shape_bongard_root):
            logger.error(f"ShapeBongard_V2 directory not found in {self.root}")
            raise FileNotFoundError(f"ShapeBongard_V2 directory not found in {self.root}")
        # Only use hd, bd, ff
        for split_folder in ["hd", "bd", "ff"]:
            split_path = os.path.join(shape_bongard_root, split_folder)
            if not os.path.isdir(split_path):
                continue
            # Recursively walk through all subfolders
            for dirpath, dirnames, filenames in os.walk(split_path):
                # Only look for folders named '0' or '1' directly under an 'images' folder
                if os.path.basename(os.path.dirname(dirpath)) == "images" and os.path.basename(dirpath) in ["0", "1"]:
                    label = 0 if os.path.basename(dirpath) == "0" else 1
                    for fn in filenames:
                        if fn.lower().endswith(".png"):
                            self.samples.append((os.path.join(dirpath, fn), label))
        logger.info(f"Initialized BongardLogoDataset (ShapeBongard_V2) with {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Retrieves an image and its corresponding label.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, int]: A tuple containing the transformed image tensor
                                      (shape: [1, H, W]) and its binary label (0 or 1).
        """
        path, label = self.samples[idx]
        try:
            # Open image and convert to grayscale ('L' mode)
            img = Image.open(path).convert("L")
            img = self.transform(img)               # torch.FloatTensor [1,H,W]
        except Exception as e:
            logger.error(f"Error loading image {path}: {e}")
            # Return a dummy tensor and label in case of error
            img = torch.zeros(1, self.img_size, self.img_size)
            label = -1 # Indicate an error or invalid label
        return img, label

if __name__ == '__main__':
    # Example Usage:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create a dummy dataset structure for testing
    dummy_root = "./dummy_bongardlogo_data"
    os.makedirs(os.path.join(dummy_root, "train", "problem_0001", "images", "pos"), exist_ok=True)
    os.makedirs(os.path.join(dummy_root, "train", "problem_0001", "images", "neg"), exist_ok=True)
    os.makedirs(os.path.join(dummy_root, "val", "problem_0002", "images", "pos"), exist_ok=True)
    os.makedirs(os.path.join(dummy_root, "val", "problem_0002", "images", "neg"), exist_ok=True)

    # Create dummy image files
    Image.new('L', (128, 128), color=0).save(os.path.join(dummy_root, "train", "problem_0001", "images", "pos", "pos_01.png"))
    Image.new('L', (128, 128), color=255).save(os.path.join(dummy_root, "train", "problem_0001", "images", "neg", "neg_01.png"))
    Image.new('L', (128, 128), color=100).save(os.path.join(dummy_root, "val", "problem_0002", "images", "pos", "pos_01.png"))

    print("--- Testing BongardLogoDataset (Train Split) ---")
    try:
        train_dataset = BongardLogoDataset(root_dir=dummy_root, split="train", img_size=64)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        print(f"Train dataset size: {len(train_dataset)}")
        for i, (img, label) in enumerate(train_loader):
            print(f"Batch {i}: Image shape {img.shape}, Label: {label}")
            if i == 0: # Check first batch
                assert img.shape == (img.shape[0], 1, 64, 64), "Image shape mismatch for grayscale"
                assert img.min() >= -1.0 and img.max() <= 1.0, "Image values not normalized to [-1,1]"
            if i > 2: break # Only print a few batches

        print("\n--- Testing BongardLogoDataset (Validation Split) ---")
        val_dataset = BongardLogoDataset(root_dir=dummy_root, split="val", img_size=64)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
        print(f"Validation dataset size: {len(val_dataset)}")
        for i, (img, label) in enumerate(val_loader):
            print(f"Batch {i}: Image shape {img.shape}, Label: {label}")
            if i > 0: break

    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the dummy data structure is correctly created.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    # Clean up dummy data
    import shutil
    if os.path.exists(dummy_root):
        shutil.rmtree(dummy_root)
        print(f"\nCleaned up dummy data at {dummy_root}")
