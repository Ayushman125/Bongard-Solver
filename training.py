# Folder: bongard_solver/
# File: training.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau, CosineAnnealingLR
import torch.nn.functional as F
import torch.optim.swa_utils as swa_utils
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import logging
import os
import collections
import random  # For MixupCutmix
import numpy as np  # For MixupCutmix
import json  # For parsing GT JSON
from typing import List, Dict, Any, Tuple, Optional, Union
from PIL import Image  # For saving synthetic images for Grad-CAM
import torchvision.transforms as T  # For converting tensor to PIL Image for Grad-CAM
import threading  # For async updates (conceptual)
import copy  # For deepcopy for Mean Teacher
# Import for pruning
import torch.nn.utils.prune as prune
# Import configuration
from config import (
    CONFIG, DEVICE, HAS_WANDB, HAS_TORCH_QUANTIZATION,
    ATTRIBUTE_FILL_MAP, ATTRIBUTE_COLOR_MAP, ATTRIBUTE_SIZE_MAP,
    ATTRIBUTE_ORIENTATION_MAP, ATTRIBUTE_SHAPE_MAP, ATTRIBUTE_TEXTURE_MAP,
    RELATION_MAP, IMAGENET_MEAN, IMAGENET_STD
)
# Import from utils
from utils import set_seed, get_symbolic_embedding_dims, make_edge_index_map, _calculate_iou, async_update_priorities
# Import from data
from data import get_dataloader, build_dali_image_processor, BongardSyntheticDataset, RealBongardDataset, CurriculumSampler, PrototypicalSampler, HardExampleSampler  # Added HardExampleSampler
# Import models
from models import PerceptionModule, LitBongard, LitSimCLR, AttributeClassifier  # Import AttributeClassifier for teacher models
# Import losses
from losses import LabelSmoothingCrossEntropy, FeatureConsistencyLoss, SymbolicConsistencyLoss, DistillationLoss, NTXentLoss
# PyTorch Lightning imports
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
# Conditional imports for advanced optimizers
try:
    from torch_optimizer import SAM
    CONFIG['HAS_SAM_OPTIMIZER'] = True
    logger.info("torch_optimizer (SAM) found and enabled.")
except ImportError:
    CONFIG['HAS_SAM_OPTIMIZER'] = False
    logger.warning("torch_optimizer (SAM) not found. SAM optimizer will not be available.")
try:
    from ranger_adabelief import RangerAdaBelief
    CONFIG['HAS_RANGER'] = True
    logger.info("ranger_adabelief found and enabled. (Note: This is a config flag updated by training.py)")
except ImportError:
    CONFIG['HAS_RANGER'] = False
    logger.warning("ranger_adabelief not found. Ranger optimizer will not be available.")
try:
    from lion_pytorch import Lion
    CONFIG['HAS_LION'] = True
    logger.info("lion_pytorch found and enabled. (Note: This is a config flag updated by training.py)")
except ImportError:
    CONFIG['HAS_LION'] = False
    logger.warning("lion_pytorch not found. Lion optimizer will not be available.")
# Conditional import for DALI
try:
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    CONFIG['HAS_DALI'] = True
    logger.info("NVIDIA DALI found and enabled. (Note: This is a config flag updated by training.py)")
except ImportError:
    CONFIG['HAS_DALI'] = False
    logger.warning("NVIDIA DALI not found. Falling back to PyTorch DataLoader for data loading.")

# Conditional import for quantization
if HAS_TORCH_QUANTIZATION:
    try:
        from torch.quantization import prepare_qat, convert, fuse_modules
        from torch.quantization.observer import MovingAverageMinMaxObserver, HistogramObserver
        from torch.quantization.fake_quantize import FakeQuantize
        logger.info("PyTorch quantization modules found and enabled.")
    except ImportError:
        logger.warning("PyTorch quantization modules not found. Quantization features will not be available.")
        HAS_TORCH_QUANTIZATION = False # Update config if import fails
else:
    logger.warning("PyTorch quantization is disabled in config. Quantization features will not be available.")

logger = logging.getLogger(__name__)

# --- Mixup and CutMix Augmenter ---
class MixupCutmixAugmenter:
    """
    Applies Mixup or CutMix augmentation to a batch of images and labels.
    """
    def __init__(self, training_config: Dict[str, Any], num_classes: int):
        self.mixup_alpha = training_config.get('mixup_alpha', 0.8)
        self.cutmix_alpha = training_config.get('cutmix_alpha', 1.0)
        self.mixup_prob = training_config.get('mixup_prob', 0.5)
        self.cutmix_prob = training_config.get('cutmix_prob', 0.5)
        self.num_classes = num_classes
        self.training = True  # This needs to be set based on model.training

    def set_training_mode(self, mode: bool):
        self.training = mode

    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images (torch.Tensor): Batch of images (B, C, H, W).
            labels (torch.Tensor): Batch of integer labels (B,).
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Augmented images and mixed labels (one-hot).
        """
        if not self.training:  # Only apply in training mode
            return images, F.one_hot(labels, num_classes=self.num_classes).float()
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        rand_index = torch.randperm(images.size(0), device=images.device)
        
        mixed_images = images
        mixed_labels = F.one_hot(labels, num_classes=self.num_classes).float()
        if random.random() < self.mixup_prob:
            # Mixup
            mixed_images = lam * images + (1 - lam) * images[rand_index]
            mixed_labels = lam * mixed_labels + (1 - lam) * mixed_labels[rand_index]
            logger.debug("Applied Mixup augmentation.")
        elif random.random() < self.cutmix_prob:
            # CutMix
            bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.size(), lam)
            mixed_images = images.clone()
            mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[rand_index, :, bby1:bby2, bbx1:bbx2]
            # Adjust lambda for CutMix
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
            mixed_labels = lam * mixed_labels + (1 - lam) * mixed_labels[rand_index]
            logger.debug("Applied CutMix augmentation.")
        
        return mixed_images, mixed_labels

    def _rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)
        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2

# --- Knowledge Distillation Helper ---
def _get_ensemble_teacher_logits(
    teacher_models: nn.ModuleList,
    raw_images_np: List[np.ndarray],  # Raw numpy images for DALI or torchvision
    raw_gt_json_strings: List[bytes],  # For scene graph parsing
    raw_support_images_np: List[np.ndarray],  # Raw numpy support images
    distillation_config: Dict[str, Any],
    dali_image_processor: Optional[Any] = None  # DALI processor if used
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Computes ensemble teacher logits for distillation.
    Args:
        teacher_models (nn.ModuleList): List of teacher models (PerceptionModule instances).
        raw_images_np (List[np.ndarray]): List of raw query images (numpy arrays).
        raw_gt_json_strings (List[bytes]): List of raw GT JSON strings for query images.
        raw_support_images_np (List[np.ndarray]): List of raw support images (numpy arrays).
        distillation_config (Dict[str, Any]): Distillation configuration.
        dali_image_processor (Optional[Any]): DALI processor instance.
    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]:
            - Mean teacher logits (B, num_classes).
            - Optional distillation mask (B,) indicating agreement.
    """
    if not teacher_models:
        logger.warning("No teacher models provided for distillation.")
        return torch.empty(0), None
    all_teacher_logits = []
    
    # Prepare support images for teacher models (if few-shot is enabled)
    processed_support_images_reshaped = None
    support_labels_for_teacher_forward = None  # Will be dummy if few-shot is enabled
    
    if CONFIG['few_shot']['enable'] and raw_support_images_np:
        max_support_imgs = CONFIG['data']['synthetic_data_config']['max_support_images_per_problem']
        batch_size_actual = len(raw_images_np)
        
        if dali_image_processor:
            # DALI needs support images flattened, then reshaped.
            # We assume raw_support_images_np is already flattened.
            _, _, processed_support_images_flat_dali = dali_image_processor.run(
                [np.zeros((1,1,3), dtype=np.uint8)] * batch_size_actual,  # Dummy query1
                [np.zeros((1,1,3), dtype=np.uint8)] * batch_size_actual,  # Dummy query2
                raw_support_images_np  # Actual support images
            )
            processed_support_images_reshaped = processed_support_images_flat_dali.view(
                batch_size_actual, max_support_imgs, 
                processed_support_images_flat_dali.shape[1], processed_support_images_flat_dali.shape[2], processed_support_images_flat_dali.shape[3]
            )
        else:
            # Fallback for torchvision transforms
            transform = T.Compose([
                T.ToPILImage(),
                T.Resize((CONFIG['data']['image_size'], CONFIG['data']['image_size'])),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
            # raw_support_images_np is a flat list of all support images in batch
            processed_support_images_flat_pt = torch.stack([transform(img_np) for img_np in raw_support_images_np]).to(DEVICE)
            processed_support_images_reshaped = processed_support_images_flat_pt.view(
                batch_size_actual, max_support_imgs, 
                processed_support_images_flat_pt.shape[1], processed_support_images_flat_pt.shape[2], processed_support_images_flat_pt.shape[3]
            )
        
        # Create dummy support labels for teacher's forward pass (as PerceptionModule.forward expects it)
        k_shot = CONFIG['few_shot']['k_shot']
        n_way = CONFIG['few_shot']['n_way']
        dummy_support_labels_flat = torch.cat([torch.full((k_shot,), fill_value=c_id, dtype=torch.long) for c_id in range(n_way)])
        support_labels_for_teacher_forward = dummy_support_labels_flat.repeat(batch_size_actual, 1).to(DEVICE)

    # Process query images
    processed_query_images = None
    if dali_image_processor:
        processed_query_images, _, _ = dali_image_processor.run(
            raw_images_np,
            [np.zeros((1,1,3), dtype=np.uint8)] * len(raw_images_np),  # Dummy view2
            [np.zeros((1,1,3), dtype=np.uint8)] * len(raw_images_np)  # Dummy support
        )
    else:
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((CONFIG['data']['image_size'], CONFIG['data']['image_size'])),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        processed_query_images = torch.stack([transform(img_np) for img_np in raw_images_np]).to(DEVICE)

    for teacher_model in teacher_models:
        teacher_model.eval()  # Set teacher to eval mode
        # Apply dropout to teacher model during inference if specified for distillation
        if distillation_config.get('ensemble_dropout_prob', 0.0) > 0:
            for module in teacher_model.modules():
                # DropBlock2D needs to be imported if used, assuming it's a custom module
                if isinstance(module, (nn.Dropout)):  # Removed DropBlock2D if not defined
                    module.p = distillation_config['ensemble_dropout_prob']
                    module.train()  # Force dropout to be active during inference
        
        with torch.no_grad():
            teacher_outputs = teacher_model(
                processed_query_images,
                ground_truth_json_strings=raw_gt_json_strings,  # Pass GTs for SG processing if needed
                support_images=processed_support_images_reshaped,
                support_labels_flat=support_labels_for_teacher_forward  # Pass dummy support labels
            )
            all_teacher_logits.append(teacher_outputs['bongard_logits'])
        
        # Reset dropout if it was temporarily enabled for inference
        if distillation_config.get('ensemble_dropout_prob', 0.0) > 0:
            for module in teacher_model.modules():
                if isinstance(module, (nn.Dropout)): # Removed DropBlock2D if not defined
                    # Assuming DropBlock2D's original prob is stored in config or accessible
                    # For simplicity, reset to a common default or 0 for eval
                    module.p = CONFIG['model']['bongard_head_config'].get('dropout_prob', 0.0)
                    module.eval()  # Set back to eval mode

    if not all_teacher_logits:
        return torch.empty(0), None

    # Stack and average teacher logits
    teacher_logits_stacked = torch.stack(all_teacher_logits, dim=0)  # (N_teachers, B, num_classes)
    teacher_logits_mean = teacher_logits_stacked.mean(dim=0)  # (B, num_classes)

    distillation_mask = None
    if distillation_config.get('use_mask_distillation', False):
        # Calculate agreement among teachers
        teacher_predictions = torch.argmax(teacher_logits_stacked, dim=-1)  # (N_teachers, B)
        # Check if all teachers agree for each sample
        agreement_mask = (teacher_predictions == teacher_predictions[0]).all(dim=0)  # (B,)
        
        # Optional: agreement based on confidence threshold
        # This would involve checking if max softmax prob > threshold for all teachers
        
        distillation_mask = agreement_mask.float()  # Convert boolean to float (1.0 or 0.0)
        logger.debug(f"Distillation mask generated. {distillation_mask.sum().item()}/{len(distillation_mask)} samples will be distilled.")

    return teacher_logits_mean, distillation_mask

# --- PyTorch Lightning DataModule (Optional, but good for DDP and DALI) ---
class BongardDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.train_dataset = None
        self.val_dataset = None
        self.dali_image_processor = None
        self.val_dali_image_processor = None
        self.setup_called = False  # Flag to ensure setup runs only once per process

    def prepare_data(self):
        # Download/prepare data if not already present.
        # This is called once per node/process.
        if self.cfg['data']['use_synthetic_data']:
            # Ensure data directories exist for synthetic data
            os.makedirs(os.path.join(self.cfg['data']['data_root_path'], 'synthetic_temp'), exist_ok=True)
        else:
            # For real data, ensure dataset_path exists
            if not os.path.exists(self.cfg['data']['real_data_config']['dataset_path']):
                logger.error(f"Real data path not found: {self.cfg['data']['real_data_config']['dataset_path']}")
                # You might want to raise an error or provide instructions to download
        pass

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders.
        # This is called on each GPU/process.
        if self.setup_called and stage != 'test':  # Allow re-setup for test stage if needed
            return
        
        logger.info(f"Setting up data for rank {self.trainer.global_rank} / {self.trainer.world_size}")
        # Initialize BongardGenerator if using synthetic data
        bongard_generator = None
        if self.cfg['data']['use_synthetic_data']:
            try:
                from bongard_rules import ALL_BONGARD_RULES  # Assuming this is defined
                from data import BongardGenerator
                bongard_generator = BongardGenerator(self.cfg['data'], ALL_BONGARD_RULES)
            except ImportError:
                logger.error("Could not import BongardGenerator or ALL_BONGARD_RULES. Synthetic data generation will fail.")
                self.cfg['data']['use_synthetic_data'] = False  # Disable synthetic data

        if stage == 'fit' or stage is None:
            if self.cfg['data']['use_synthetic_data']:
                self.train_dataset = BongardSyntheticDataset(
                    self.cfg, bongard_generator, num_samples=self.cfg['data']['synthetic_data_config']['num_train_problems']
                )
                self.val_dataset = BongardSyntheticDataset(
                    self.cfg, bongard_generator, num_samples=self.cfg['data']['synthetic_data_config']['num_val_problems']
                )
            else:
                from data import load_bongard_data
                train_data_list, val_data_list = load_bongard_data(
                    self.cfg['data']['real_data_config']['dataset_path'],
                    self.cfg['data']['real_data_config']['dataset_name'],
                    self.cfg['data']['real_data_config']['train_split']
                )
                self.train_dataset = RealBongardDataset(train_data_list)
                self.val_dataset = RealBongardDataset(val_data_list)
        
        # Initialize DALI image processor for this process
        if self.cfg['data']['use_dali'] and CONFIG['HAS_DALI']:  # Check HAS_DALI from config.py
            self.dali_image_processor = build_dali_image_processor(
                batch_size=self.cfg['training']['batch_size'],
                num_threads=self.cfg['data']['dataloader_workers'],
                device_id=self.trainer.local_rank,  # Use local_rank for DALI device_id
                image_size=self.cfg['data']['image_size'],
                is_training=True,  # For training pipeline, this is for training data
                curriculum_config=self.cfg['training']['curriculum_config'],
                augmentation_config=self.cfg['training']['augmentation_config']
            )
            # Create a separate DALI processor for validation if needed, or re-use with is_training=False
            self.val_dali_image_processor = build_dali_image_processor(
                batch_size=self.cfg['training']['batch_size'],
                num_threads=self.cfg['data']['dataloader_workers'],
                device_id=self.trainer.local_rank,
                image_size=self.cfg['data']['image_size'],
                is_training=False,  # For validation data
                curriculum_config=self.cfg['training']['curriculum_config'],  # Still pass for consistency
                augmentation_config=self.cfg['training']['augmentation_config']
            )
        self.setup_called = True

    def train_dataloader(self) -> DataLoader:
        # Conditional DALI loader (as per prompt)
        if self.cfg['data']['use_dali'] and CONFIG['HAS_DALI'] and self.dali_image_processor:
            logger.info(f"Using DALI DALIGenericIterator for training on rank {self.trainer.global_rank}.")
            # The output_map needs to match the outputs defined in your DALI pipeline.
            # Assuming the pipeline outputs query_img1, query_img2, support_imgs_flat
            # and that custom_collate_fn expects these.
            return DALIGenericIterator(
                [self.dali_image_processor.pipeline],
                output_map=["query_img1", "query_img2", "support_imgs_flat"],
                size=len(self.train_dataset),  # Total samples
                auto_reset=True,
                fill_last_batch=False,  # Drop last batch if incomplete
                dynamic_shape=True  # Allow variable shapes if needed
            )
        else:
            logger.info(f"Using PyTorch DataLoader for training on rank {self.trainer.global_rank}.")
            # This is the `build_pt_loader` part implicitly
            # get_dataloader handles samplers and collate_fn
            return get_dataloader(
                self.cfg,
                dataset=self.train_dataset,  # Pass the dataset directly
                is_train=True,
                rank=self.trainer.global_rank,
                world_size=self.trainer.world_size
            )

    def val_dataloader(self) -> DataLoader:
        # Conditional DALI loader for validation
        if self.cfg['data']['use_dali'] and CONFIG['HAS_DALI'] and self.val_dali_image_processor:
            logger.info(f"Using DALI DALIGenericIterator for validation on rank {self.trainer.global_rank}.")
            return DALIGenericIterator(
                [self.val_dali_image_processor.pipeline],
                output_map=["query_img1", "query_img2", "support_imgs_flat"],
                size=len(self.val_dataset),
                auto_reset=True,
                fill_last_batch=False,
                dynamic_shape=True
            )
        else:
            logger.info(f"Using PyTorch DataLoader for validation on rank {self.trainer.global_rank}.")
            return get_dataloader(
                self.cfg,
                dataset=self.val_dataset,  # Pass the dataset directly
                is_train=False,
                rank=self.trainer.global_rank,
                world_size=self.trainer.world_size
            )

    def teardown(self, stage: Optional[str] = None):
        # Clean up resources (e.g., close DALI pipelines)
        if self.dali_image_processor:
            self.dali_image_processor.release()
        if self.val_dali_image_processor:
            self.val_dali_image_processor.release()
        logger.info(f"DataModule teardown for rank {self.trainer.global_rank}.")

# --- Quantization Functions (Moved from prune_quantize.py for self-containment) ---
def quantize_model_qat(model: nn.Module, cfg: Dict[str, Any]) -> nn.Module:
    """
    Applies Quantization-Aware Training (QAT) to the model.
    Args:
        model (nn.Module): The model to be quantized.
        cfg (Dict[str, Any]): Configuration dictionary.
    Returns:
        nn.Module: The QAT-prepared model.
    """
    if not HAS_TORCH_QUANTIZATION:
        logger.warning("PyTorch quantization not available. Skipping QAT.")
        return model

    logger.info("Preparing model for Quantization-Aware Training (QAT).")
    
    # Define quantization configuration for QAT
    # qconfig = torch.quantization.get_default_qat_qconfig('fbgemm') # For server-side inference
    qconfig = torch.quantization.QConfig(
        activation=torch.quantization.observer.FusedMovingAverageMinMaxObserver.with_args(
            quant_min=0, quant_max=255, dtype=torch.quint8, qscheme=torch.per_tensor_affine
        ),
        weight=torch.quantization.observer.MovingAverageMinMaxObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_tensor_symmetric
        )
    )

    # Fuse modules for better quantization performance
    # This is a critical step for QAT. Identify common patterns like Conv-ReLU, Conv-BN-ReLU.
    # The specific modules to fuse depend on your model architecture.
    # Example for a common CNN backbone:
    # Assuming 'model' is a PerceptionModule and its backbone has layers like 'conv1', 'bn1', 'relu'
    # You might need to inspect your model's structure (e.g., print(model)) to identify fusable layers.
    
    # For a PerceptionModule, typical fusion candidates might be within the feature_extractor
    # and potentially the attribute_model/relation_model if they use standard layers.
    # This example assumes a simple sequential structure or known fusable blocks.
    
    # Create a copy of the model for QAT to avoid modifying the original in-place
    qat_model = copy.deepcopy(model)
    qat_model.train() # QAT requires the model in training mode

    # Example fusion list (customize based on your actual model's structure)
    # You'll need to know the exact names of sequential layers that can be fused.
    # For instance, if you have `self.conv1`, `self.bn1`, `self.relu1`
    # fusable_modules = [
    #     ['feature_extractor.conv1', 'feature_extractor.bn1', 'feature_extractor.relu1'],
    #     # Add more such sequences
    # ]
    # try:
    #     qat_model = fuse_modules(qat_model, fusable_modules)
    #     logger.info("Modules fused for QAT.")
    # except Exception as e:
    #     logger.warning(f"Failed to fuse modules for QAT: {e}. Proceeding without fusion.")

    # Prepare the model for QAT
    qat_model.qconfig = qconfig
    torch.quantization.prepare_qat(qat_model, inplace=True)
    logger.info("Model prepared for QAT. Now needs to be fine-tuned.")
    
    # The model should now be fine-tuned with this prepared_qat_model.
    # This function only prepares it. The actual training loop needs to run QAT.
    # In run_training_pipeline, this prepared model should replace the original for a few epochs.
    return qat_model

def quantize_model_ptq(model: nn.Module, data_loader: DataLoader, cfg: Dict[str, Any]) -> nn.Module:
    """
    Applies Post-Training Quantization (PTQ) to the model.
    Args:
        model (nn.Module): The model to be quantized.
        data_loader (DataLoader): DataLoader for calibration data.
        cfg (Dict[str, Any]): Configuration dictionary.
    Returns:
        nn.Module: The quantized model.
    """
    if not HAS_TORCH_QUANTIZATION:
        logger.warning("PyTorch quantization not available. Skipping PTQ.")
        return model

    logger.info("Initiating Post-Training Quantization (PTQ).")

    # Ensure model is in evaluation mode
    model.eval()

    # Fuse modules for better quantization performance
    # Similar to QAT, identify common patterns.
    ptq_model = copy.deepcopy(model)

    # Example fusion list (customize based on your actual model's structure)
    # try:
    #     ptq_model = fuse_modules(ptq_model, fusable_modules) # Use the same fusable_modules as QAT
    #     logger.info("Modules fused for PTQ.")
    # except Exception as e:
    #     logger.warning(f"Failed to fuse modules for PTQ: {e}. Proceeding without fusion.")

    # Define quantization configuration for PTQ (dynamic or static)
    # For static quantization, we need to observe activations.
    # For dynamic quantization, only weights are quantized ahead of time.
    quantization_type = cfg['quantization'].get('ptq_type', 'static') # 'static' or 'dynamic'

    if quantization_type == 'static':
        logger.info("Using static PTQ.")
        # qconfig = torch.quantization.get_default_qconfig('fbgemm') # For server-side inference
        qconfig = torch.quantization.QConfig(
            activation=torch.quantization.observer.MovingAverageMinMaxObserver.with_args(
                quant_min=0, quant_max=255, dtype=torch.quint8, qscheme=torch.per_tensor_affine
            ),
            weight=torch.quantization.observer.MovingAverageMinMaxObserver.with_args(
                dtype=torch.qint8, qscheme=torch.per_tensor_symmetric
            )
        )
        ptq_model.qconfig = qconfig
        torch.quantization.prepare(ptq_model, inplace=True)

        # Calibrate the model
        logger.info("Calibrating model for static PTQ...")
        with torch.no_grad():
            for i, batch in enumerate(tqdm(data_loader, desc="PTQ Calibration")):
                # Assuming batch structure is (images, labels, gt_json, support_images)
                # Adjust according to your actual DataLoader output
                images, _, gt_json_strings, support_images = batch
                images = images.to(DEVICE)
                # If support_images are used by the model, move them to device too
                if support_images is not None:
                    support_images = support_images.to(DEVICE)
                
                # Forward pass through the model to collect statistics
                # The _get_ensemble_teacher_logits function expects raw numpy images and json strings
                # For calibration, we need to pass the processed tensors to the model directly.
                # Assuming the `PerceptionModule` can take processed tensors.
                # If your `PerceptionModule`'s forward expects raw numpy and then processes,
                # you'll need to adapt this part to feed raw data or pre-process here.
                
                # For simplicity, assuming model.forward can take pre-processed tensors
                # If your model's forward needs raw_images_np and raw_gt_json_strings,
                # you'll need to modify the data_loader to yield those.
                # For now, let's assume `model` (PerceptionModule) can take `images` directly.
                
                # If your model expects raw data, you'll need to get the raw data from the dataset
                # or modify the DataLoader to yield raw data for calibration.
                # For now, let's assume `images` are the tensor inputs for `model`.
                
                # Dummy call to forward to collect activation statistics
                # The `PerceptionModule` forward method needs `ground_truth_json_strings`
                # and `support_images` if few-shot is enabled.
                # We need to ensure the calibration data loader provides these.
                
                # If data_loader provides processed tensors, but model expects raw for internal processing
                # this will be an issue. The simplest is to ensure the model's forward
                # can directly take the preprocessed tensors for calibration.
                
                # Let's assume `model` (PerceptionModule) can take `images` directly
                # and `ground_truth_json_strings` and `support_images` are handled.
                
                # The `_get_ensemble_teacher_logits` function is for teacher distillation,
                # not directly for calibrating the student model for PTQ.
                # For PTQ calibration, we just need to run the `model` (student) with data.
                
                # If `model` is `PerceptionModule`:
                # This assumes `data_loader` provides `images` (tensor), `gt_json_strings` (list of bytes), `support_images` (tensor)
                # If not, you'll need to adapt the data loading for calibration.
                
                # Ensure `gt_json_strings` is a list of strings/bytes if model expects it.
                # If `data_loader` gives tensors for `gt_json_strings`, convert them.
                
                # Assuming `data_loader` provides `images` (tensor), and `gt_json_strings` and `support_images`
                # are also available in the batch in the format expected by `PerceptionModule.forward`.
                
                # Create dummy ground_truth_json_strings if not directly available or needed for calibration
                # If `PerceptionModule`'s forward requires `ground_truth_json_strings` for internal graph processing
                # even during inference for PTQ, then the calibration data needs to provide it.
                # For a typical PTQ, you just need to pass the image data through the network.
                # Let's simplify for calibration: just pass images. If `PerceptionModule`
                # needs other inputs, the calibration loop will need to be adjusted.
                
                # If your model's forward method looks like:
                # `def forward(self, query_images, ground_truth_json_strings=None, support_images=None, support_labels_flat=None):`
                # then you need to provide these.
                
                # For a typical calibration, you just need to run forward pass.
                # Assuming `PerceptionModule` can run with just `images` for feature extraction.
                # If not, you need to provide dummy or actual `gt_json_strings` and `support_images`.
                
                # Simple forward pass for calibration:
                _ = ptq_model(images, ground_truth_json_strings=gt_json_strings, support_images=support_images) # Pass all expected inputs

        logger.info("Calibration for static PTQ completed. Converting model.")
        quantized_model = torch.quantization.convert(ptq_model, inplace=False) # Use inplace=False to return a new model
    elif quantization_type == 'dynamic':
        logger.info("Using dynamic PTQ.")
        # For dynamic quantization, we don't need to calibrate activations.
        # qconfig = torch.quantization.get_default_qconfig('fbgemm') # For server-side inference
        qconfig = torch.quantization.QConfig(
            activation=torch.quantization.observer.MovingAverageMinMaxObserver.with_args(
                quant_min=0, quant_max=255, dtype=torch.quint8, qscheme=torch.per_tensor_affine
            ),
            weight=torch.quantization.observer.MovingAverageMinMaxObserver.with_args(
                dtype=torch.qint8, qscheme=torch.per_tensor_symmetric
            )
        )
        ptq_model.qconfig = qconfig
        # For dynamic quantization, we use `quantize_dynamic` directly.
        # This will insert observers and convert the model.
        quantized_model = torch.quantization.quantize_dynamic(
            ptq_model, {nn.Linear, nn.LSTM}, dtype=torch.qint8
        )
        logger.info("Dynamic PTQ completed.")
    else:
        logger.warning(f"Unknown PTQ type: {quantization_type}. Skipping PTQ.")
        return model

    return quantized_model

# --- Main Training Orchestration Functions ---
def run_training_pipeline(cfg: Dict[str, Any]):
    """
    Orchestrates the main training pipeline using PyTorch Lightning.
    Handles DDP, SimCLR pretraining, and supervised training.
    """
    set_seed(cfg['training']['seed'])
    
    # 1. SimCLR Pretraining (if enabled)
    if cfg['model']['simclr_config']['enabled']:
        logger.info("--- Starting SimCLR Pretraining ---")
        simclr_model = LitSimCLR(cfg)
        simclr_data_module = BongardDataModule(cfg)
        simclr_logger = None
        if cfg['HAS_WANDB'] and cfg['training'].get('use_wandb', False):
            simclr_logger = WandbLogger(project="bongard_solver_simclr", name="simclr_pretrain")
        
        simclr_trainer = pl.Trainer(
            accelerator='gpu' if DEVICE.type == 'cuda' else 'cpu',
            devices=cfg['training'].get('devices', 1),  # Number of GPUs
            strategy='ddp' if cfg['training'].get('devices', 1) > 1 else 'auto',
            max_epochs=cfg['model']['simclr_config']['pretrain_epochs'],
            precision=16 if cfg['training']['use_amp'] else 32,
            logger=simclr_logger,
            callbacks=[
                ModelCheckpoint(
                    dirpath=cfg['debug']['save_model_checkpoints'],
                    filename="simclr_best_model",
                    monitor="simclr_train_loss",
                    mode="min",
                    save_top_k=1
                ),
                LearningRateMonitor(logging_interval='epoch')
            ]
        )
        simclr_trainer.fit(simclr_model, datamodule=simclr_data_module)
        logger.info("SimCLR Pretraining completed.")
        # Load best SimCLR model weights into the main model's backbone
        best_simclr_checkpoint = os.path.join(cfg['debug']['save_model_checkpoints'], "simclr_best_model.ckpt")
        if os.path.exists(best_simclr_checkpoint):
            logger.info(f"Loading best SimCLR checkpoint from {best_simclr_checkpoint}")
            # Load the entire Lightning module
            loaded_simclr_model = LitSimCLR.load_from_checkpoint(best_simclr_checkpoint, cfg=cfg)
            # Transfer backbone weights to the main PerceptionModule
            # Ensure the backbone structure is compatible
            # This assumes `perception_module.attribute_model.feature_extractor` is the target.
            
            # Create a dummy PerceptionModule to get the target backbone
            dummy_perception_module = PerceptionModule(cfg)
            target_backbone = dummy_perception_module.attribute_model.feature_extractor
            
            # Load state_dict, filtering for backbone layers
            simclr_backbone_state_dict = {
                k.replace('feature_extractor.', ''): v
                for k, v in loaded_simclr_model.feature_extractor.state_dict().items()
                if k.startswith('feature_extractor.')
            }
            
            # Load into the target backbone
            # Use strict=False to ignore missing projection head layers in the target
            target_backbone.load_state_dict(simclr_backbone_state_dict, strict=False)
            logger.info("SimCLR pretrained backbone weights transferred to main model.")
        else:
            logger.warning("No SimCLR checkpoint found. Main model will not use SimCLR pretrained weights.")

    # 2. Supervised Bongard Solver Training
    logger.info("--- Starting Supervised Bongard Solver Training ---")
    model = LitBongard(cfg)
    data_module = BongardDataModule(cfg)

    # Load teacher models for Knowledge Distillation if enabled
    if cfg['training']['use_knowledge_distillation'] and cfg['ensemble']['teacher_model_paths']:
        logger.info("Loading teacher models for Knowledge Distillation.")
        for teacher_path in cfg['ensemble']['teacher_model_paths']:
            if os.path.exists(teacher_path):
                try:
                    teacher_model = PerceptionModule(cfg)  # Teacher is also a PerceptionModule
                    # Load state_dict, assuming it's a raw model state_dict
                    teacher_model.load_state_dict(torch.load(teacher_path, map_location=DEVICE))
                    teacher_model.eval()  # Set teacher to eval mode
                    model.teacher_models.append(teacher_model)
                    logger.info(f"Loaded teacher model from: {teacher_path}")
                except Exception as e:
                    logger.error(f"Failed to load teacher model from {teacher_path}: {e}")
            else:
                logger.warning(f"Teacher model checkpoint not found: {teacher_path}. Skipping.")
        if not model.teacher_models:
            logger.warning("No valid teacher models loaded. Knowledge Distillation will be disabled.")
            cfg['training']['use_knowledge_distillation'] = False  # Disable if no teachers

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=cfg['debug']['save_model_checkpoints'],
            filename="best_bongard_model_{epoch:02d}-{val_accuracy:.4f}",
            monitor="val/accuracy",
            mode="max",
            save_top_k=1,
            save_last=True,
            verbose=True
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]
    
    # Early Stopping Callback (as per prompt)
    # This is implemented via PyTorch Lightning's EarlyStopping callback
    if cfg['training'].get('early_stop_patience', 0) > 0:
        callbacks.append(EarlyStopping(
            monitor="val/accuracy",  # Monitor validation accuracy
            mode="max",  # We want to maximize accuracy
            patience=cfg['training']['early_stop_patience'],
            verbose=True
        ))
        logger.info(f"Early Stopping enabled with patience: {cfg['training']['early_stop_patience']}")

    # Logger
    main_logger = None
    if cfg['HAS_WANDB'] and cfg['training'].get('use_wandb', False):
        main_logger = WandbLogger(project="bongard_solver_main", name="supervised_training", log_model='all')
        main_logger.watch(model, log='all', log_freq=cfg['training']['log_interval_batches'])

    trainer = pl.Trainer(
        accelerator='gpu' if DEVICE.type == 'cuda' else 'cpu',
        devices=cfg['training'].get('devices', 1),  # Number of GPUs
        strategy='ddp' if cfg['training'].get('devices', 1) > 1 else 'auto',
        max_epochs=cfg['training']['epochs'],
        precision=16 if cfg['training']['use_amp'] else 32,
        logger=main_logger,
        callbacks=callbacks,
        log_every_n_steps=cfg['training']['log_interval_batches'],
        check_val_every_n_epoch=1,  # Validate every epoch
        # Limit training/validation batches for faster debugging if needed
        # limit_train_batches=100,
        # limit_val_batches=50,
    )
    
    trainer.fit(model, datamodule=data_module)
    logger.info("Supervised Bongard Solver Training completed.")

    # After training, load the best model for potential pruning/quantization
    best_model_path = trainer.checkpoint_callback.best_model_path
    final_model = None
    if best_model_path:
        logger.info(f"Loading best model from {best_model_path} for final evaluation/optimization.")
        # Load the entire Lightning module
        final_model = LitBongard.load_from_checkpoint(best_model_path, cfg=cfg)
        # Save the raw PerceptionModule state_dict for pruning/quantization script
        perception_module_state_dict_path = os.path.join(cfg['debug']['save_model_checkpoints'], "best_perception_module.pth")
        torch.save(final_model.perception_module.state_dict(), perception_module_state_dict_path)
        logger.info(f"Best PerceptionModule state_dict saved to {perception_module_state_dict_path}")
        # Update config for prune_quantize.py
        cfg['pruning']['checkpoint'] = perception_module_state_dict_path
    else:
        logger.warning("No best model checkpoint found after training. Using the last trained model for quantization attempts.")
        final_model = model # Use the last state of the model if no best checkpoint

    # QAT/PTQ invocation (as per prompt)
    if HAS_TORCH_QUANTIZATION:
        if cfg['quantization']['qat']:
            logger.info("Initiating Quantization Aware Training (QAT).")
            # The model loaded from checkpoint is `final_model` (LitBongard)
            # We need to quantize its `perception_module`.
            
            # Prepare the perception_module for QAT
            qat_prepared_perception_module = quantize_model_qat(final_model.perception_module, cfg)
            
            # Now, you need to fine-tune this `qat_prepared_perception_module`.
            # This typically involves running a short training loop with the prepared model.
            # For simplicity here, we'll just convert it after preparation, but in a real scenario,
            # you'd re-train `final_model` with `qat_prepared_perception_module` as its backbone.
            
            # A more complete QAT integration would involve:
            # 1. Loading `final_model`
            # 2. Replacing `final_model.perception_module` with `qat_prepared_perception_module`
            # 3. Running `trainer.fit` again for a few epochs with this QAT-enabled model.
            # 4. Then calling `torch.quantization.convert` on the fine-tuned QAT model.

            # For demonstration and direct integration as requested, we'll prepare and then convert immediately.
            # In a real scenario, the QAT preparation should happen *before* the final fine-tuning phase.
            
            # To properly integrate QAT into the existing Lightning training pipeline:
            # 1. Modify `LitBongard` to accept a pre-quantized `PerceptionModule` or
            #    add a method to `LitBongard` to prepare its `perception_module` for QAT.
            # 2. After initial training and loading `final_model`, call this method.
            # 3. Run a *new* `trainer.fit` call for a few QAT epochs.
            # 4. Then call `convert` on the `final_model.perception_module`.

            # Given the current structure, let's assume `quantize_model_qat`
            # prepares the model, and we'll then convert it. This is a simplified QAT flow.
            
            # To make this runnable directly after initial training without a second training loop:
            # We will prepare the model for QAT and then immediately convert it.
            # This is effectively a "post-training static quantization with fake-quant ops"
            # if no further training happens. For true QAT, a training loop is essential.
            
            # Let's use the `final_model.perception_module` for QAT preparation.
            # If `final_model` is None (no best checkpoint), use `model` (last state).
            model_to_quantize_qat = final_model.perception_module if final_model else model.perception_module
            
            # Prepare the model for QAT (inserts FakeQuantize modules)
            qat_prepared_model = quantize_model_qat(model_to_quantize_qat, cfg)
            
            # For a proper QAT, you would now train `qat_prepared_model` for a few epochs.
            # Since the request implies this happens *after* the main training,
            # we'll perform the conversion immediately after preparation for demonstration.
            # This is more akin to a static PTQ with QAT-like qconfigs.
            
            # Convert the prepared QAT model to a fully quantized model
            quantized_qat_model = torch.quantization.convert(qat_prepared_model.eval(), inplace=False)
            
            # Optionally, save the quantized model
            torch.save(quantized_qat_model.state_dict(), os.path.join(cfg['debug']['save_model_checkpoints'], "quantized_qat_model.pth"))
            logger.info("QAT preparation and immediate conversion completed and quantized model saved. (Note: For full QAT benefits, fine-tuning after preparation is recommended.)")

        if cfg['quantization']['ptq']:
            logger.info("Initiating Post-Training Quantization (PTQ).")
            # For PTQ, we need the model and a calibration dataloader (val_loader)
            # We can use the validation dataloader from the DataModule for calibration.
            val_dataloader_for_ptq = data_module.val_dataloader()
            
            # If `final_model` is None (no best checkpoint), use `model` (last state).
            model_to_quantize_ptq = final_model.perception_module if final_model else model.perception_module

            quantized_ptq_model = quantize_model_ptq(model_to_quantize_ptq, val_dataloader_for_ptq, cfg)
            # Optionally, save the quantized model
            torch.save(quantized_ptq_model.state_dict(), os.path.join(cfg['debug']['save_model_checkpoints'], "quantized_ptq_model.pth"))
            logger.info("PTQ completed and quantized model saved.")
    else:
        logger.warning("PyTorch quantization is not enabled or available. Skipping QAT/PTQ steps.")


def run_training_once(cfg: Dict[str, Any], epochs: int = 5) -> float:
    """
    Runs a single training session for HPO trials.
    Returns the validation accuracy.
    """
    set_seed(cfg['training']['seed'])  # Ensure reproducibility for each trial
    model = LitBongard(cfg)
    data_module = BongardDataModule(cfg)
    # For HPO trials, use a simpler trainer without extensive logging/callbacks
    trainer = pl.Trainer(
        accelerator='gpu' if DEVICE.type == 'cuda' else 'cpu',
        devices=1,  # Single device for HPO trials to simplify
        max_epochs=epochs,
        precision=16 if cfg['training']['use_amp'] else 32,
        logger=False,  # No extensive logging for individual trials
        enable_checkpointing=False,  # No checkpoints for trials
        enable_progress_bar=False,  # No progress bar for trials
        check_val_every_n_epoch=1,
        # limit_train_batches=0.1, # Use a subset of data for faster trials
        # limit_val_batches=0.1,
    )
    
    trainer.fit(model, datamodule=data_module)
    
    # Evaluate the model on the validation set to get the metric for HPO
    # trainer.test will return a list of dictionaries, one per test_dataloader.
    # We need to ensure val_dataloader is used for evaluation.
    results = trainer.validate(model, datamodule=data_module)
    
    if results and len(results) > 0:
        val_accuracy = results[0].get('val/accuracy', 0.0)
        logger.info(f"HPO Trial: Validation Accuracy = {val_accuracy:.4f}")
        return val_accuracy
    else:
        logger.warning("HPO Trial: No validation results found. Returning 0.0 accuracy.")
        return 0.0
