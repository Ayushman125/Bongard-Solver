# Folder: bongard_solver/

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler, OneCycleLR, ReduceLROnPlateau, CosineAnnealingLR
import logging
from typing import Dict, Any, Optional, Union

# Import from config (for conditional imports)
from config import HAS_SAM, HAS_SOPHIA, HAS_TIMM_OPTIM

logger = logging.getLogger(__name__)

# Conditional imports for specific optimizers
if HAS_SAM:
    try:
        from sam_pytorch import SAM # Or your specific SAM optimizer library
    except ImportError:
        logger.warning("SAM optimizer not found. SAM will be disabled.")
        HAS_SAM = False

if HAS_SOPHIA:
    try:
        from sophia import SophiaG # Assuming this is from sophia-optimizer library
    except ImportError:
        logger.warning("SophiaG optimizer not found. SophiaG will be disabled.")
        HAS_SOPHIA = False

if HAS_TIMM_OPTIM:
    try:
        from timm.optim import Lion, MADGRAD # Assuming these are from timm
    except ImportError:
        logger.warning("timm.optim not found. Lion/MADGRAD from timm will be disabled.")
        HAS_TIMM_OPTIM = False
        # Fallback to torch_optimizer if available and desired
        try:
            from torch_optimizer import Lion as TorchOptimizerLion, MADGRAD as TorchOptimizerMADGRAD
            Lion = TorchOptimizerLion
            MADGRAD = TorchOptimizerMADGRAD
            logger.info("Using Lion/MADGRAD from torch_optimizer as timm.optim fallback.")
        except ImportError:
            logger.warning("torch_optimizer not found either. Lion/MADGRAD disabled.")


def get_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """
    Initializes and returns the specified optimizer.

    Args:
        model (nn.Module): The model whose parameters are to be optimized.
        config (Dict[str, Any]): The training configuration dictionary.

    Returns:
        torch.optim.Optimizer: The initialized optimizer.
    """
    optimizer_name = config['optimizer']
    learning_rate = config['learning_rate']
    
    if optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_name == 'SophiaG' and HAS_SOPHIA:
        optimizer = SophiaG(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'Lion' and HAS_TIMM_OPTIM:
        optimizer = Lion(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'MADGRAD' and HAS_TIMM_OPTIM:
        optimizer = MADGRAD(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'SAM' and HAS_SAM:
        base_optimizer = optim.AdamW(model.parameters(), lr=learning_rate) # SAM wraps a base optimizer
        optimizer = SAM(model.parameters(), base_optimizer, rho=config['sam_rho'])
    else:
        logger.warning(f"Optimizer '{optimizer_name}' not found or supported. Falling back to AdamW.")
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    logger.info(f"Initialized optimizer: {optimizer_name}")
    return optimizer


def get_scheduler(optimizer: optim.Optimizer, config: Dict[str, Any], total_steps: int) -> Optional[_LRScheduler]:
    """
    Initializes and returns the specified learning rate scheduler.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to schedule.
        config (Dict[str, Any]): The training configuration dictionary.
        total_steps (int): Total number of training steps (epochs * batches_per_epoch),
                           required for OneCycleLR.

    Returns:
        Optional[torch.optim.lr_scheduler._LRScheduler]: The initialized scheduler, or None if no scheduler.
    """
    scheduler_name = config['scheduler']
    scheduler_config = config['scheduler_config']
    
    scheduler = None
    if scheduler_name == 'OneCycleLR':
        scheduler = OneCycleLR(optimizer,
                               max_lr=scheduler_config['OneCycleLR']['max_lr'],
                               total_steps=total_steps,
                               pct_start=scheduler_config['OneCycleLR']['pct_start'],
                               anneal_strategy=scheduler_config['OneCycleLR']['anneal_strategy'])
    elif scheduler_name == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, **scheduler_config['ReduceLROnPlateau'])
    elif scheduler_name == 'CosineAnnealingLR':
        # T_max can be total_epochs or total_steps depending on desired annealing
        # Here, assuming T_max refers to epochs for simplicity
        scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])
    elif scheduler_name != 'None':
        logger.warning(f"Scheduler '{scheduler_name}' not found or supported. No scheduler will be used.")
    
    if scheduler:
        logger.info(f"Initialized scheduler: {scheduler_name}")
    else:
        logger.info("No learning rate scheduler will be used.")
    return scheduler

# You could also define custom optimizer/scheduler classes here if needed.
# Example:
# class CustomOptimizer(optim.Optimizer):
#     def __init__(self, params, lr=1e-3):
#         super().__init__(params, {'lr': lr})
#         # ... custom init ...
#     def step(self, closure=None):
#         # ... custom step logic ...

