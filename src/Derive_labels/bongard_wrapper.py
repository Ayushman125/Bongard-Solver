import torch
from src.Derive_labels.models import FeatureExtractorDNN, AdaptiveClassifierDNN
from src.Derive_labels.meta_learning import MetaLearnerWrapper, MAML

class BongardLOGOModelWrapper:
    """Wrapper for integrating trained Bongard-LOGO model into current pipeline"""
    def __init__(self, model_checkpoint_path: str, device: str = 'cuda'):
        import sys, os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Bongard-LOGO/Bongard-LOGO_Baselines')))
        import models
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_checkpoint_path, map_location=self.device)
        # Reconstruct the full Bongard-LOGO model as in training
        self.model = models.make(checkpoint['model'], **checkpoint['model_args'])
        self.model.load_state_dict(checkpoint['model_sd'])
        self.model.eval()

    def infer(self, query_x, query_program):
        # Call the Bongard-LOGO ProgramDecoder with image and program tensor
        logits = self.model(query_x, query_program)
        prob = torch.sigmoid(logits).item()
        return prob
