import numpy as np
from PIL import Image
import cv2
from src.preprocess.crop_extractor import MaskRCNNCropper
from src.features.topo_features import TopologicalFeatureExtractor

img = cv2.imread("ShapeBongard_V2/sample_bongard.png")
cropper = MaskRCNNCropper()
topo = TopologicalFeatureExtractor()

patches = cropper.segment_shapes(img)
for i, (patch, mask) in enumerate(patches):
    ph_feat = topo.extract_features(mask)
    print(f"[{i}] PH vector length: {len(ph_feat)}, Non-zero: {np.count_nonzero(ph_feat)}")
