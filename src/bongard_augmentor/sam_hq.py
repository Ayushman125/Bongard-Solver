import torch
import numpy as np
import cv2
import logging
from pathlib import Path
from segment_anything_hq import sam_hq_model_registry, SamHQAutomaticMaskGenerator, SamHQPredictor

class SAMHQEnsemble:
    def __init__(self, models=['vit_h', 'vit_b'], cache_dir="~/.cache/samhq", device="cuda", ensemble_weights=None):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ensemble_weights = ensemble_weights or [0.6, 0.4]
        self.models = {}
        self.predictors = {}
        self.is_initialized = False
        try:
            self._initialize_ensemble(models)
        except Exception as e:
            logging.error(f"SAM-HQ Ensemble initialization failed: {e}")
            self.is_initialized = False

    def _initialize_ensemble(self, model_types):
        checkpoint_urls = {
            "vit_b": "https://dl.fbaipublicfiles.com/segment_anything_hq/samhq_vit_b.pth",
            "vit_l": "https://dl.fbaipublicfiles.com/segment_anything_hq/samhq_vit_l.pth",
            "vit_h": "https://dl.fbaipublicfiles.com/segment_anything_hq/samhq_vit_h.pth"
        }
        for model_type in model_types:
            try:
                ckpt_path = self._get_checkpoint(model_type, checkpoint_urls[model_type])
                sam_model = sam_hq_model_registry[model_type](checkpoint=ckpt_path)
                sam_model.to(self.device)
                self.models[model_type] = sam_model
                self.predictors[model_type] = SamHQPredictor(sam_model)
                logging.info(f"Loaded SAM-HQ {model_type} successfully")
            except Exception as e:
                logging.error(f"Failed to load SAM-HQ {model_type}: {e}")
        if self.models:
            self.is_initialized = True
            logging.info(f"SAM-HQ Ensemble initialized with {len(self.models)} models")

    def _get_checkpoint(self, model_type, url):
        filename = f"samhq_{model_type}_checkpoint.pth"
        ckpt_path = self.cache_dir / filename
        if not ckpt_path.exists():
            logging.info(f"Downloading SAM-HQ {model_type} checkpoint...")
            try:
                import requests
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(ckpt_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logging.info(f"Downloaded SAM-HQ checkpoint to {ckpt_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to download SAM-HQ checkpoint: {e}")
        return str(ckpt_path)

    def generate_masks(self, image, prompts=None, use_ensemble=True):
        if not self.is_initialized:
            return None
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        if prompts is None:
            mask_gen = SamHQAutomaticMaskGenerator(self.models[list(self.models.keys())[0]])
            masks = mask_gen.generate(image)
            if masks:
                best_mask = max(masks, key=lambda m: m.get('area', 0))
                return best_mask['segmentation'].astype(np.uint8)*255
            return None
        all_masks = []
        all_scores = []
        for model_name, weight in zip(self.models.keys(), self.ensemble_weights):
            predictor = self.predictors[model_name]
            predictor.set_image(image)
            for prompt in prompts:
                try:
                    mask, score, _ = predictor.predict(
                        point_coords=prompt['point_coords'],
                        point_labels=prompt['point_labels'],
                        multimask_output=True
                    )
                    all_masks.extend([m.astype(np.uint8)*255 for m in mask])
                    all_scores.extend([s*weight for s in score.tolist()])
                except Exception as e:
                    logging.warning(f"SAM-HQ prediction failed for prompt: {e}")
        if all_masks and all_scores:
            best_idx = int(np.argmax(all_scores))
            return all_masks[best_idx]
        return None
