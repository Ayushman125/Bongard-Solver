# OWL-ViT detection and embedding extraction
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from torchvision.transforms import functional as TF

class EmbedDetector:
    """
    OWL-ViT detection and embedding extraction utility.
    Usage:
        det = EmbedDetector(model_cfg)
        boxes, scores, labels, embeddings = det.detect(image, prompts)
    """
    def __init__(self, model_cfg):
        self.device = torch.device(model_cfg['device'])
        self.processor = OwlViTProcessor.from_pretrained(model_cfg['name'])
        self.model = OwlViTForObjectDetection.from_pretrained(model_cfg['name'])
        self.model.to(self.device).eval()
        self.threshold = model_cfg['detection_threshold']
        self.max_queries = model_cfg['max_queries']

    def detect(self, image, prompts):
        """
        Detect objects and extract embeddings.
        Args:
            image: PIL.Image
            prompts: list of str
        Returns:
            boxes, scores, labels, embeddings
        """
        inputs = self.processor(text=[prompts], images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        # Post-process detections to get xyxy boxes and scores
        target_sizes = torch.tensor([[image.height, image.width]], device=self.device)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.threshold
        )[0]

        boxes = results['boxes'].cpu().numpy()        # [[x0,y0,x1,y1], ...]
        scores = results['scores'].cpu().numpy()      # [0.8, 0.5, ...]
        labels = results['labels'].cpu().numpy()      # indices into prompts[]

        # Extract CLIP embeddings for each crop
        embeddings = []
        for box in boxes:
            x0, y0, x1, y1 = [int(v) for v in box]
            crop = TF.crop(image, y0, x0, y1-y0, x1-x0)
            pix = self.processor.images_processor(crop, return_tensors="pt").pixel_values.to(self.device)
            clip_outputs = self.model.clip.vision_model(pix)
            embeddings.append(clip_outputs.pooler_output.cpu().detach().numpy()[0])

        return boxes, scores, labels, embeddings

# --- CLI entry point ---
if __name__ == "__main__":
    import argparse
    from PIL import Image
    parser = argparse.ArgumentParser(description="OWL-ViT detection utility")
    parser.add_argument('--img', required=True, help='Path to image')
    parser.add_argument('--prompt', required=True, nargs='+', help='Detection prompts')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--model', default='google/owlvit-base-patch32')
    args = parser.parse_args()
    cfg = {'device': args.device, 'name': args.model, 'detection_threshold': 0.3, 'max_queries': 10}
    det = EmbedDetector(cfg)
    image = Image.open(args.img).convert('RGB')
    boxes, scores, labels, embeddings = det.detect(image, args.prompt)
    print('Boxes:', boxes)
    print('Scores:', scores)
    print('Labels:', labels)
