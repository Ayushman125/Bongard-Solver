# Folder: bongard_solver/core_models/
# File: optimizers.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler, OneCycleLR, ReduceLROnPlateau, CosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import logging
import math
from typing import Dict, Any, Optional, Union

# Import from config (assuming config.py is in the project root, so relative path from core_models)
try:
    from ..config import HAS_TIMM_OPTIM  # Check if timm.optim is preferred
except ImportError:
    HAS_TIMM_OPTIM = False
    logging.getLogger(__name__).warning("Could not import HAS_TIMM_OPTIM from config. Defaulting to False.")


# Added for Gradual Warmup
HAS_GRADUAL_WARMUP = False
try:
    from warmup_scheduler import GradualWarmupScheduler
    HAS_GRADUAL_WARMUP = True
    logging.getLogger(__name__).info("GradualWarmupScheduler found and enabled.")
except ImportError:
    logging.getLogger(__name__).warning("warmup_scheduler not found. Gradual Warmup functionality will be disabled.")

logger = logging.getLogger(__name__)

# Conditional imports for specific optimizers
# Prioritize timm.optim if HAS_TIMM_OPTIM is True
Lion = None
MADGRAD = None
if HAS_TIMM_OPTIM:
    try:
        from timm.optim import Lion as TimmLion, MADGRAD as TimmMADGRAD
        Lion = TimmLion
        MADGRAD = TimmMADGRAD
        logger.info("Using Lion/MADGRAD from timm.optim.")
    except ImportError:
        logger.warning("timm.optim not found. Lion/MADGRAD from timm will be disabled. Checking torch_optimizer as fallback.")
        HAS_TIMM_OPTIM = False  # Disable timm.optim if import fails
        try:
            from torch_optimizer import Lion as TorchOptimizerLion, MADGRAD as TorchOptimizerMADGRAD
            Lion = TorchOptimizerLion
            MADGRAD = TorchOptimizerMADGRAD
            logger.info("Using Lion/MADGRAD from torch_optimizer as timm.optim fallback.")
        except ImportError:
            logger.warning("torch_optimizer not found either. Lion/MADGRAD disabled.")
            Lion = None
            MADGRAD = None
else:
    logger.info("HAS_TIMM_OPTIM is False. Checking torch_optimizer for Lion/MADGRAD.")
    try:
        from torch_optimizer import Lion as TorchOptimizerLion, MADGRAD as TorchOptimizerMADGRAD
        Lion = TorchOptimizerLion
        MADGRAD = TorchOptimizerMADGRAD
        logger.info("Using Lion/MADGRAD from torch_optimizer.")
    except ImportError:
        logger.warning("torch_optimizer not found. Lion/MADGRAD disabled.")
        Lion = None
        MADGRAD = None

# --- Import Local SAM and SophiaG Implementations ---
# Assuming your actual sam.py and sophia.py files are in the same core_models directory.
SAM = None  # Initialize to None
try:
    from .sam import SAM
    logger.info("Successfully imported local SAM optimizer from .sam.py.")
except ImportError:
    logger.error("Could not import SAM from .sam.py. Please ensure sam.py is in the core_models directory.")
    # Define a dummy SAM to prevent NameError if import fails
    class SAM(optim.Optimizer):
        def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
            super().__init__(params, dict(rho=rho, adaptive=adaptive, **kwargs))
            self.base_optimizer = base_optimizer
            logger.warning("Using dummy SAM optimizer due to import failure.")
        def first_step(self, zero_grad=False): pass
        def second_step(self, zero_grad=False): pass
        def _grad_norm(self): return torch.tensor(1.0)  # Dummy norm
        def load_state_dict(self, state_dict): pass

SophiaG = None  # Initialize to None
try:
    from .sophia import SophiaG
    logger.info("Successfully imported local SophiaG optimizer from .sophia.py.")
except ImportError:
    logger.error("Could not import SophiaG from .sophia.py. Please ensure sophia.py is in the core_models directory.")
    # Define a dummy SophiaG to prevent NameError if import fails
    class SophiaG(optim.Optimizer):
        def __init__(self, params, lr=1e-3, betas=(0.965, 0.995), rho=0.04, weight_decay=1e-1,
                     gamma=0.99, k=10, eps=1e-12):
            super().__init__(params, dict(lr=lr, betas=betas, rho=rho, weight_decay=weight_decay,
                                          gamma=gamma, k=k, eps=eps))
            logger.warning("Using dummy SophiaG optimizer due to import failure.")
        def step(self, closure=None): return None

# --- Conditional imports for Performer and Nystromformer Attention ---
HAS_PERFORMER = False
try:
    from performer_pytorch import SelfAttention as PerformerSelfAttention
    HAS_PERFORMER = True
    logger.info("PerformerSelfAttention found and enabled.")
except ImportError:
    logger.warning("performer_pytorch not found. Performer attention will be disabled.")

HAS_NYSTROM = False
try:
    from nystrom_attention import NystromAttention
    HAS_NYSTROM = True
    logger.info("NystromAttention found and enabled.")
except ImportError:
    logger.warning("nystrom_attention not found. Nystrom attention will be disabled.")


# --- Optimizer Functions ---
def get_optimizer(model_or_params: Union[nn.Module, Any], config: Dict[str, Any]) -> optim.Optimizer:
    """
    Initializes and returns the specified optimizer.
    Args:
        model_or_params (Union[nn.Module, Any]): The model whose parameters are to be optimized,
                                                  or an iterable of parameters.
        config (Dict[str, Any]): The training configuration dictionary.
                                 Expected keys: 'optimizer', 'learning_rate', 'weight_decay', 'sam_rho' (if SAM).
    Returns:
        torch.optim.Optimizer: The initialized optimizer.
    """
    # Extract parameters from model if a model is passed
    params = model_or_params.parameters() if isinstance(model_or_params, nn.Module) else model_or_params

    optimizer_name = config.optimizer
    learning_rate = config.learning_rate
    weight_decay = getattr(config, 'weight_decay', 0.0)
    
    optimizer = None
    if optimizer_name == 'AdamW':
        optimizer = optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == 'SophiaG':
        if SophiaG is not None:
            optimizer = SophiaG(params, lr=learning_rate, weight_decay=weight_decay)
        else:
            logger.warning("SophiaG is not available. Falling back to AdamW.")
            optimizer = optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'Lion':
        if Lion is not None:
            optimizer = Lion(params, lr=learning_rate, weight_decay=weight_decay)
        else:
            logger.warning("Lion optimizer is not available. Falling back to AdamW.")
            optimizer = optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'MADGRAD':
        if MADGRAD is not None:
            optimizer = MADGRAD(params, lr=learning_rate, weight_decay=weight_decay)
        else:
            logger.warning("MADGRAD optimizer is not available. Falling back to AdamW.")
            optimizer = optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'SAM':
        if SAM is not None:
            # SAM wraps a base optimizer, pass weight_decay to base_optimizer
            base_optimizer = optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
            optimizer = SAM(params, base_optimizer, rho=getattr(config, 'sam_rho', 0.05))
        else:
            logger.warning("SAM optimizer is not available. Falling back to AdamW.")
            optimizer = optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    else:
        logger.warning(f"Optimizer '{optimizer_name}' not found or supported. Falling back to AdamW.")
        optimizer = optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    
    logger.info(f"Initialized optimizer: {optimizer_name}")
    return optimizer

# --- Scheduler Functions ---
def get_scheduler(optimizer: optim.Optimizer, config: Dict[str, Any], total_steps: int) -> Optional[_LRScheduler]:
    """
    Initializes and returns the specified learning rate scheduler.
    Args:
        optimizer (torch.optim.Optimizer): The optimizer to schedule.
        config (Dict[str, Any]): The training configuration dictionary.
                                 Expected keys: 'scheduler', 'scheduler_config', 'epochs'.
        total_steps (int): Total number of training steps (epochs * batches_per_epoch),
                           required for OneCycleLR.
    Returns:
        Optional[torch.optim.lr_scheduler._LRScheduler]: The initialized scheduler, or None if no scheduler.
    """
    scheduler_name = getattr(config, 'scheduler', 'None')
    scheduler_config = getattr(config, 'scheduler_config', {})
    
    scheduler = None
    if scheduler_name == 'OneCycleLR':
        scheduler = OneCycleLR(optimizer,
                               max_lr=scheduler_config['OneCycleLR']['max_lr'],
                               total_steps=total_steps,
                               pct_start=scheduler_config['OneCycleLR']['pct_start'],
                               anneal_strategy=scheduler_config['OneCycleLR'].get('anneal_strategy', 'cos'))
        logger.info("Using OneCycleLR scheduler.")
    elif scheduler_name == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, **scheduler_config['ReduceLROnPlateau'])
        logger.info("Using ReduceLROnPlateau scheduler.")
    elif scheduler_name == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=scheduler_config['CosineAnnealingLR'].get('eta_min', 1e-6))
        logger.info("Using CosineAnnealingLR scheduler.")
    elif scheduler_name == 'warmup_cosine':
        if HAS_GRADUAL_WARMUP:
            scheduler_cos = CosineAnnealingWarmRestarts(
                optimizer, 
                T_0=scheduler_config['warmup_cosine'].get('T_0', 10),
                T_mult=scheduler_config['warmup_cosine'].get('T_mult', 2),
                eta_min=scheduler_config['warmup_cosine'].get('eta_min', 1e-6)
            )
            scheduler = GradualWarmupScheduler(
                optimizer, 
                multiplier=scheduler_config['warmup_cosine'].get('multiplier', 1.0),
                total_epoch=scheduler_config['warmup_cosine'].get('warmup_epochs', 5),
                after_scheduler=scheduler_cos
            )
            logger.info(f"Initialized warmup_cosine scheduler with warmup_epochs={scheduler_config['warmup_cosine'].get('warmup_epochs', 5)}, T_0={scheduler_config['warmup_cosine'].get('T_0', 10)}.")
        else:
            logger.warning("warmup_cosine scheduler requested but GradualWarmupScheduler not found. Falling back to CosineAnnealingLR.")
            scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])
    elif scheduler_name != 'None':
        logger.warning(f"Scheduler '{scheduler_name}' not found or supported. No scheduler will be used.")
    
    if scheduler:
        logger.info(f"Initialized scheduler: {scheduler_name}")
    else:
        logger.info("No learning rate scheduler will be used.")
    return scheduler

# --- Attention Layer Functions ---
def get_attention_layer(cfg: Dict[str, Any]) -> nn.Module:
    """
    Returns an attention layer based on the configuration.
            T_max=config.epochs,
    Args:
        cfg (Dict[str, Any]): The configuration dictionary, specifically for 'attn' and 'model' settings.
    Returns:
        nn.Module: An initialized attention layer.
    Raises:
        ValueError: If an unsupported attention type is specified.
    """
    attn_type = cfg['attn'].get('type', 'multihead')
    
    # Ensure 'feature_dim' is available in the config, potentially from model inference
    # Or, if this function is called before model is fully built, it might need to be passed.
    # For now, assume it's set in config['model'] by PerceptionModule or similar.
    feat_dim = cfg['model'].get('feature_dim')
    if feat_dim is None:
        logger.error("Feature dimension (cfg['model']['feature_dim']) not found for attention layer. Using dummy 512.")
        feat_dim = 512  # Fallback
    
    heads = cfg['attn'].get('heads', 8)
    if attn_type == 'performer':
        if HAS_PERFORMER:
            logger.info(f"Using PerformerSelfAttention with dim={feat_dim}, heads={heads}.")
            return PerformerSelfAttention(
                dim=feat_dim,
                heads=heads,
                causal=cfg['attn'].get('causal', False)
            )
        else:
            logger.warning("Performer attention requested but library not found. Falling back to MultiheadAttention.")
            return nn.MultiheadAttention(feat_dim, heads, batch_first=True)
    elif attn_type == 'nystrom':
        if HAS_NYSTROM:
            num_landmarks = cfg['attn'].get('landmarks', 64)
            logger.info(f"Using NystromAttention with dim={feat_dim}, num_landmarks={num_landmarks}.")
            return NystromAttention(
                dim=feat_dim,
                num_landmarks=num_landmarks,
                heads=heads
            )
        else:
            logger.warning("Nystrom attention requested but library not found. Falling back to MultiheadAttention.")
            return nn.MultiheadAttention(feat_dim, heads, batch_first=True)
    elif attn_type == 'multihead':
        logger.info(f"Using standard MultiheadAttention with embed_dim={feat_dim}, num_heads={heads}.")
        return nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=heads,
            batch_first=True
        )
    else:
        raise ValueError(f"Unsupported attention type: {attn_type}")

