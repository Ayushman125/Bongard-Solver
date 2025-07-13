import os
import numpy as np
from src.preprocess.crop_extractor import MaskRCNNCropper
from src.features.topo_features import TopologicalFeatureExtractor

class SceneGraphBuilder:
    def __init__(self, config):
        self.cropper = MaskRCNNCropper(conf_thresh=config.get('maskrcnn_conf_thresh', 0.7))
        self.topo = TopologicalFeatureExtractor(pixel_thresh=config.get('ph_pixel_thresh', 0.5))
        self.config = config

    def build_scene_graph(self, image_np, detected_bboxes=None, detected_masks=None, attribute_logits=None, relation_logits=None, graph_embed=None):
        # Use Mask R-CNN to get patches and masks
        patches = self.cropper.segment_shapes(image_np)
        scene = []
        for patch, mask in patches:
            # Extract appearance features (dummy placeholder)
            appearance = self.extract_appearance(patch)
            # Extract topological features
            ph_feat = self.topo.extract_features(mask)
            shape_descriptor = {
                'appearance': appearance,
                'topology': ph_feat.tolist(),
            }
            scene.append(shape_descriptor)
        return scene

    def extract_appearance(self, patch):
        # Placeholder: compute color, size, fill, etc. from patch
        # You should implement your own logic here
        return {
            'color': 'unknown',
            'size': 'unknown',
            'fill': 'unknown',
        }

def split_dataset(input_folder, output_folder, train_ratio=0.8):
    """
    Splits images from input_folder into train/val folders in output_folder.
    """
    images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    np.random.shuffle(images)
    n_train = int(len(images) * train_ratio)
    train_imgs = images[:n_train]
    val_imgs = images[n_train:]
    os.makedirs(os.path.join(output_folder, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'val'), exist_ok=True)
    for img in train_imgs:
        os.rename(os.path.join(input_folder, img), os.path.join(output_folder, 'train', img))
    for img in val_imgs:
        os.rename(os.path.join(input_folder, img), os.path.join(output_folder, 'val', img))
