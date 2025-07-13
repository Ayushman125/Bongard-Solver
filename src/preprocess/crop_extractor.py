import torch
import numpy as np
from torchvision import transforms
from PIL import Image

class MaskRCNNCropper:
    """
    Uses Mask R-CNN to segment objects and extract crops from an image.
    """
    def __init__(self, model_path=None, conf_thresh=0.7):
        # Load your Mask R-CNN model here (torchvision or detectron2)
        # For demo, we use torchvision's pretrained model
        import torchvision
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        self.conf_thresh = conf_thresh

    def segment_shapes(self, image):
        """
        Args:
            image (np.ndarray or PIL.Image): Input image
        Returns:
            List of (patch, mask) tuples
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        img_tensor = transform(image)
        with torch.no_grad():
            outputs = self.model([img_tensor])
        patches = []
        for box, mask, score in zip(outputs[0]['boxes'], outputs[0]['masks'], outputs[0]['scores']):
            if score.item() < self.conf_thresh:
                continue
            bbox = box.int().tolist()
            mask_np = mask[0].cpu().numpy() > 0.5
            crop = img_tensor[:, bbox[1]:bbox[3], bbox[0]:bbox[2]].cpu().numpy()
            crop = np.transpose(crop, (1,2,0)) * 255
            crop = crop.astype(np.uint8)
            patches.append((crop, mask_np[bbox[1]:bbox[3], bbox[0]:bbox[2]]))
        return patches
