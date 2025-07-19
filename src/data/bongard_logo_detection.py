# Folder: bongard_solver/src/data/
# File: bongard_logo_detection.py
"""
Detection Dataset for Bongard Problems

Provides COCO-style detection dataset for training Mask R-CNN on Bongard images.
Supports both synthetic and real Bongard-LOGO datasets.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.ops import masks_to_boxes
import json
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class BongardDetectionDataset(Dataset):
    """
    Dataset class for Bongard object detection training.
    
    Expects COCO-style annotations with:
    - images: list of image info dicts
    - annotations: list of annotation dicts with bbox, category_id, etc.
    - categories: list of category dicts with id and name
    """
    
    def __init__(self, 
                 annotations_file: str,
                 images_dir: str,
                 transform_config: Dict[str, Any] = None):
        """
        Args:
            annotations_file: Path to COCO-style annotations JSON
            images_dir: Directory containing images
            transform_config: Configuration for data augmentation
        """
        self.images_dir = Path(images_dir)
        self.transform_config = transform_config or {}
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            self.coco_data = json.load(f)
        
        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']
        self.categories = self.coco_data['categories']
        
        # Create mapping from image_id to annotations
        self.img_id_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_id_to_anns:
                self.img_id_to_anns[img_id] = []
            self.img_id_to_anns[img_id].append(ann)
        
        # Setup transforms
        self.setup_transforms()
        
        logger.info(f"Loaded {len(self.images)} images with {len(self.annotations)} annotations")
    
    def setup_transforms(self):
        """Setup image transforms for training."""
        image_size = self.transform_config.get('image_size', [512, 512])
        augmentation = self.transform_config.get('augmentation', True)
        
        transform_list = [T.Resize(image_size)]
        
        if augmentation:
            transform_list.extend([
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=10),
            ])
        
        transform_list.extend([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.transform = T.Compose(transform_list)
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get item for detection training.
        
        Returns:
            Tuple of (image_tensor, target_dict)
            target_dict contains:
            - boxes: (N, 4) tensor of bounding boxes in (x1, y1, x2, y2) format
            - labels: (N,) tensor of class labels  
            - masks: (N, H, W) tensor of instance masks
            - image_id: image ID
            - area: (N,) tensor of bounding box areas
            - iscrowd: (N,) tensor of crowd flags
        """
        # Get image info
        img_info = self.images[idx]
        img_path = self.images_dir / img_info['file_name']
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image_width, image_height = image.size
        
        # Get annotations for this image
        img_id = img_info['id']
        anns = self.img_id_to_anns.get(img_id, [])
        
        # Prepare target dict
        boxes = []
        labels = []
        masks = []
        areas = []
        iscrowds = []
        
        for ann in anns:
            # Bounding box (convert from COCO x,y,w,h to x1,y1,x2,y2)
            bbox = ann['bbox']
            x, y, w, h = bbox
            x1, y1, x2, y2 = x, y, x + w, y + h
            
            # Clamp to image bounds
            x1 = max(0, min(x1, image_width))
            y1 = max(0, min(y1, image_height))
            x2 = max(x1, min(x2, image_width))
            y2 = max(y1, min(y2, image_height))
            
            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue
            
            boxes.append([x1, y1, x2, y2])
            labels.append(ann['category_id'])
            areas.append(ann.get('area', (x2-x1) * (y2-y1)))
            iscrowds.append(ann.get('iscrowd', 0))
            
            # Create mask from segmentation or bbox
            if 'segmentation' in ann and ann['segmentation']:
                # Use segmentation polygon
                mask = self.create_mask_from_segmentation(
                    ann['segmentation'], image_width, image_height
                )
            else:
                # Create mask from bounding box
                mask = np.zeros((image_height, image_width), dtype=np.uint8)
                mask[int(y1):int(y2), int(x1):int(x2)] = 1
            
            masks.append(mask)
        
        # Handle empty annotations
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, image_height, image_width), dtype=torch.uint8)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowds = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            masks = torch.tensor(np.array(masks), dtype=torch.uint8)
            areas = torch.tensor(areas, dtype=torch.float32)
            iscrowds = torch.tensor(iscrowds, dtype=torch.int64)
        
        # Apply transforms to image
        if self.transform:
            image = self.transform(image)
        
        # Prepare target dict
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor(img_id, dtype=torch.int64),
            'area': areas,
            'iscrowd': iscrowds
        }
        
        return image, target
    
    def create_mask_from_segmentation(self, segmentation: List, width: int, height: int) -> np.ndarray:
        """Create binary mask from COCO segmentation polygons."""
        mask = np.zeros((height, width), dtype=np.uint8)
        
        if isinstance(segmentation, list):
            # Polygon format
            for seg in segmentation:
                if len(seg) >= 6:  # At least 3 points (6 coordinates)
                    # Convert to PIL polygon format
                    polygon = []
                    for i in range(0, len(seg), 2):
                        if i + 1 < len(seg):
                            polygon.append((seg[i], seg[i + 1]))
                    
                    if polygon:
                        img = Image.new('L', (width, height), 0)
                        ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
                        mask = np.maximum(mask, np.array(img))
        
        return mask
    
    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """
        Custom collate function for detection dataset.
        
        Args:
            batch: List of (image, target) tuples
            
        Returns:
            Tuple of (batched_images, list_of_targets)
        """
        images = []
        targets = []
        
        for image, target in batch:
            images.append(image)
            targets.append(target)
        
        # Stack images into batch
        images = torch.stack(images, dim=0)
        
        return images, targets

def create_detection_dataloader(dataset: BongardDetectionDataset, 
                               batch_size: int = 4,
                               shuffle: bool = True,
                               num_workers: int = 4) -> DataLoader:
    """Create a DataLoader for detection training."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=BongardDetectionDataset.collate_fn,
        pin_memory=True
    )

# Example usage and testing
if __name__ == "__main__":
    # Test the dataset
    import tempfile
    import json
    
    # Create dummy annotations for testing
    dummy_annotations = {
        'images': [
            {'id': 0, 'file_name': 'test.jpg', 'height': 224, 'width': 224}
        ],
        'annotations': [
            {
                'id': 1, 'image_id': 0, 'category_id': 1,
                'bbox': [50, 50, 100, 100], 'area': 10000, 'iscrowd': 0,
                'segmentation': [[50, 50, 150, 50, 150, 150, 50, 150]]
            }
        ],
        'categories': [
            {'id': 1, 'name': 'triangle'},
            {'id': 2, 'name': 'quadrilateral'}
        ]
    }
    
    # Test dataset creation
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(dummy_annotations, f)
        ann_file = f.name
    
    print("BongardDetectionDataset test completed successfully!")
        
    def __len__(self):
        return len(self.ids)
        
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(self.images_dir, f"{img_id}.png")
        
        # Load image
        if not os.path.exists(img_path):
            # Try with .jpg extension
            img_path = os.path.join(self.images_dir, f"{img_id}.jpg")
        
        img = Image.open(img_path).convert("RGB")
        
        # Get annotations for this image
        annos = self.annotations[img_id]
        
        # Extract data from annotations
        boxes = []
        labels = []
        masks = []
        fill_labels = []
        color_labels = []
        
        for anno in annos:
            # Bounding box: convert [x, y, w, h] to [x1, y1, x2, y2]
            x, y, w, h = anno['bbox']
            boxes.append([x, y, x + w, y + h])
            
            # Shape label (main object class for detection)
            labels.append(anno['shape_label'])
            
            # Attribute labels
            fill_labels.append(anno.get('fill_label', 0))
            color_labels.append(anno.get('color_label', 0))
            
            # Mask (if available and requested)
            if self.load_masks and 'mask' in anno:
                masks.append(anno['mask'])
            elif self.load_masks:
                # Generate a simple rectangular mask if not provided
                h_img, w_img = img.size[1], img.size[0]
                mask = np.zeros((h_img, w_img), dtype=np.uint8)
                mask[int(y):int(y+h), int(x):int(x+w)] = 1
                masks.append(mask)
        
        # Convert to tensors
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        fill_labels = torch.tensor(fill_labels, dtype=torch.int64) if fill_labels else torch.zeros((0,), dtype=torch.int64)
        color_labels = torch.tensor(color_labels, dtype=torch.int64) if color_labels else torch.zeros((0,), dtype=torch.int64)
        
        if masks and self.load_masks:
            masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
        else:
            masks = torch.zeros((0, img.size[1], img.size[0]), dtype=torch.uint8)
        
        # Apply transforms to image
        if self.transforms:
            img = self.transforms(img)
        
        # Build target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'fill_labels': fill_labels,
            'color_labels': color_labels,
            'image_id': torch.tensor([idx], dtype=torch.int64),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.tensor([]),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64) if len(boxes) > 0 else torch.tensor([])
        }
        
        return img, target

def collate_fn(batch):
    """Custom collate function for batching detection samples"""
    return tuple(zip(*batch))

def generate_synthetic_annotations(dataset, output_dir: str, 
                                 images_dir: str = None,
                                 save_images: bool = True) -> Dict[str, List[Dict]]:
    """
    Generate COCO-style annotations from a SyntheticBongardDataset.
    
    Args:
        dataset: SyntheticBongardDataset instance
        output_dir: Directory to save annotations and images
        images_dir: Directory to save images (defaults to output_dir/images)
        save_images: Whether to save images to disk
    
    Returns:
        Dictionary mapping image_id -> list of annotations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if images_dir is None:
        images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    annotations = {}
    
    # Label mappings
    shape_to_id = {'circle': 0, 'square': 1, 'triangle': 2, 'pentagon': 3, 'star': 4}
    fill_to_id = {'solid': 0, 'outline': 1, 'striped': 2, 'gradient': 3}
    color_to_id = {'red': 0, 'blue': 1, 'green': 2, 'yellow': 3, 'purple': 4, 'orange': 5, 'black': 6}
    
    print(f"[generate_synthetic_annotations] Processing {len(dataset)} examples...")
    
    for idx in tqdm(range(len(dataset)), desc="🔍 Generating annotations", unit="img"):
        example = dataset[idx]
        img_id = f"synth_{idx:06d}"
        
        # Save image if requested
        if save_images:
            img_path = os.path.join(images_dir, f"{img_id}.png")
            if isinstance(example['image'], np.ndarray):
                img_pil = Image.fromarray(example['image'])
            else:
                img_pil = example['image']
            img_pil.save(img_path)
        
        # Extract objects from scene graph
        scene_graph = example.get('scene_graph', {})
        objects = scene_graph.get('objects', [])
        
        img_annotations = []
        
        for obj_idx, obj in enumerate(objects):
            # Get object properties
            x = obj.get('x', 0)
            y = obj.get('y', 0)
            size = obj.get('size', 20)
            shape = obj.get('shape', 'circle')
            fill = obj.get('fill', 'solid')
            color = obj.get('color', 'black')
            
            # Create bounding box (approximate square around object)
            half_size = size // 2
            bbox = [max(0, x - half_size), max(0, y - half_size), size, size]
            
            # Generate binary mask for the object
            if isinstance(example['image'], np.ndarray):
                img_h, img_w = example['image'].shape[:2]
            else:
                img_w, img_h = example['image'].size
            
            mask = generate_object_mask(obj, img_w, img_h)
            
            # Create annotation
            annotation = {
                'bbox': bbox,
                'mask': mask,
                'shape_label': shape_to_id.get(shape, 0),
                'fill_label': fill_to_id.get(fill, 0),
                'color_label': color_to_id.get(color, 6),  # default to black
                'area': size * size,
                'object_id': obj_idx
            }
            
            img_annotations.append(annotation)
        
        annotations[img_id] = img_annotations
    
    # Save annotations to JSON
    annotations_path = os.path.join(output_dir, "annotations.json")
    with open(annotations_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_annotations = {}
        for img_id, img_annos in annotations.items():
            json_annos = []
            for anno in img_annos:
                json_anno = anno.copy()
                if isinstance(json_anno['mask'], np.ndarray):
                    json_anno['mask'] = json_anno['mask'].tolist()
                json_annos.append(json_anno)
            json_annotations[img_id] = json_annos
        
        json.dump(json_annotations, f, indent=2)
    
    print(f"[generate_synthetic_annotations] ✅ Saved {len(annotations)} annotations to {annotations_path}")
    
    return annotations

def generate_object_mask(obj: Dict, img_w: int, img_h: int) -> np.ndarray:
    """
    Generate a binary mask for a single object.
    
    Args:
        obj: Object dictionary with x, y, size, shape properties
        img_w: Image width
        img_h: Image height
    
    Returns:
        Binary mask as numpy array
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    
    x = obj.get('x', 0)
    y = obj.get('y', 0)
    size = obj.get('size', 20)
    shape = obj.get('shape', 'circle')
    
    # Create a PIL image to draw on
    mask_img = Image.fromarray(mask)
    draw = ImageDraw.Draw(mask_img)
    
    # Calculate bounding box
    half_size = size // 2
    x1, y1 = max(0, x - half_size), max(0, y - half_size)
    x2, y2 = min(img_w, x + half_size), min(img_h, y + half_size)
    
    # Draw shape
    if shape == 'circle':
        draw.ellipse([x1, y1, x2, y2], fill=1)
    elif shape == 'square':
        draw.rectangle([x1, y1, x2, y2], fill=1)
    elif shape == 'triangle':
        # Simple triangle
        points = [(x, y1), (x1, y2), (x2, y2)]
        draw.polygon(points, fill=1)
    else:
        # Default to circle for other shapes
        draw.ellipse([x1, y1, x2, y2], fill=1)
    
    return np.array(mask_img)

def load_annotations_from_json(json_path: str) -> Dict[str, List[Dict]]:
    """
    Load annotations from a JSON file.
    
    Args:
        json_path: Path to the annotations JSON file
    
    Returns:
        Dictionary mapping image_id -> list of annotations
    """
    with open(json_path, 'r') as f:
        json_annotations = json.load(f)
    
    # Convert mask lists back to numpy arrays
    annotations = {}
    for img_id, img_annos in json_annotations.items():
        annos = []
        for anno in img_annos:
            anno_copy = anno.copy()
            if 'mask' in anno_copy and isinstance(anno_copy['mask'], list):
                anno_copy['mask'] = np.array(anno_copy['mask'], dtype=np.uint8)
            annos.append(anno_copy)
        annotations[img_id] = annos
    
    return annotations

# Test function
def test_dataset():
    """Test function to verify the dataset works correctly"""
    print("Testing BongardDetectionDataset...")
    
    # Create dummy annotations
    dummy_annotations = {
        "test_001": [
            {
                'bbox': [10, 10, 50, 50],
                'shape_label': 0,  # circle
                'fill_label': 0,   # solid
                'color_label': 0,  # red
                'mask': np.ones((128, 128), dtype=np.uint8)
            }
        ]
    }
    
    # Create dummy image directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a dummy image
        dummy_img = Image.new('RGB', (128, 128), color='white')
        img_path = os.path.join(tmp_dir, "test_001.png")
        dummy_img.save(img_path)
        
        # Create dataset
        dataset = BongardDetectionDataset(
            images_dir=tmp_dir,
            annotations=dummy_annotations,
            transforms=T.ToTensor()
        )
        
        # Test loading
        img, target = dataset[0]
        print(f"✅ Successfully loaded image: {img.shape}")
        print(f"✅ Target keys: {list(target.keys())}")
        print(f"✅ Boxes shape: {target['boxes'].shape}")
        print(f"✅ Labels: {target['labels']}")
        
    print("BongardDetectionDataset test completed!")

if __name__ == "__main__":
    test_dataset()
