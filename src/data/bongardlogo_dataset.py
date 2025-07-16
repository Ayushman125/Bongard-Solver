# Folder: bongard_solver/data/
# File: bongardlogo_dataset.py

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import logging

logger = logging.getLogger(__name__)

class BongardLogoDataset(Dataset):
    """
    Dataset for the B/W Bongard-LOGO problems.
    Each problem contains 6 positive and 6 negative images.
    """
    def __init__(self, root_dir: str, split: str = "train", img_size: int = 128):
        """
        Initializes the BongardLogoDataset.

        Args:
            root_dir (str): The root directory of the Bongard-LOGO dataset.
                            Expected structure: root_dir/{split}/problem_XXXX/images/{pos|neg}/
            split (str): The dataset split to use ('train', 'val', 'test').
            img_size (int): The target size for the images (img_size x img_size).
        """
        self.root = os.path.join(root_dir, split)
        self.img_size = img_size

        # Transformation pipeline for grayscale images
        # ToTensor() converts PIL Image (H,W) to FloatTensor (1,H,W) in [0,1]
        # Normalize((0.5,), (0.5,)) maps values from [0,1] to [-1,1] for grayscale
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),                # yields [1,H,W], values 0..1
            transforms.Normalize((0.5,), (0.5,))  # for gray channel, maps to [-1,1]
        ])

        self.samples = []
        if not os.path.isdir(self.root):
            logger.error(f"Bongard-LOGO dataset root directory not found: {self.root}")
            raise FileNotFoundError(f"Bongard-LOGO dataset root directory not found: {self.root}")

        problem_dirs = sorted([d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))])
        if not problem_dirs:
            logger.warning(f"No problem directories found in {self.root}. Dataset will be empty.")

        for prob_dir_name in problem_dirs:
            problem_path = os.path.join(self.root, prob_dir_name)
            images_dir = os.path.join(problem_path, "images") # Assuming 'images' subdirectory
            
            if not os.path.isdir(images_dir):
                logger.warning(f"Skipping {problem_path}: 'images' subdirectory not found.")
                continue

            for side, label in [("pos", 1), ("neg", 0)]:
                side_path = os.path.join(images_dir, side)
                if not os.path.isdir(side_path):
                    logger.warning(f"Skipping {side_path}: Directory not found.")
                    continue

                for fn in os.listdir(side_path):
                    if fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                        self.samples.append((os.path.join(side_path, fn), label))
        
        logger.info(f"Initialized BongardLogoDataset for split '{split}' with {len(self.samples)} samples.")

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
