# Folder: bongard_solver/

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler, OneCycleLR, ReduceLROnPlateau, CosineAnnealingLR
import logging
import math # Import math for SophiaG placeholder
from typing import Dict, Any, Optional, Union

# Import from config (for conditional imports)
from config import HAS_TIMM_OPTIM

logger = logging.getLogger(__name__)

# Conditional imports for specific optimizers
# Prioritize timm.optim if HAS_TIMM_OPTIM is True
if HAS_TIMM_OPTIM:
    try:
        from timm.optim import Lion as TimmLion, MADGRAD as TimmMADGRAD
        Lion = TimmLion
        MADGRAD = TimmMADGRAD
        logger.info("Using Lion/MADGRAD from timm.optim.")
    except ImportError:
        logger.warning("timm.optim not found. Lion/MADGRAD from timm will be disabled. Checking torch_optimizer as fallback.")
        HAS_TIMM_OPTIM = False # Disable timm.optim if import fails
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
# Assuming your actual sam.py and sophia.py files are in the same project directory.
try:
    from sam import SAM
    logger.info("Successfully imported local SAM optimizer from sam.py.")
except ImportError:
    logger.error("Could not import SAM from sam.py. Please ensure sam.py is in the project directory.")
    # Define a dummy SAM to prevent NameError if import fails
    class SAM(optim.Optimizer):
        def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
            super().__init__(params, dict(rho=rho, adaptive=adaptive, **kwargs))
            self.base_optimizer = base_optimizer
            logger.warning("Using dummy SAM optimizer due to import failure.")
        def first_step(self, zero_grad=False): pass
        def second_step(self, zero_grad=False): pass
        def _grad_norm(self): return torch.tensor(1.0) # Dummy norm
        def load_state_dict(self, state_dict): pass


try:
    from sophia import SophiaG
    logger.info("Successfully imported local SophiaG optimizer from sophia.py.")
except ImportError:
    logger.error("Could not import SophiaG from sophia.py. Please ensure sophia.py is in the project directory.")
    # Define a dummy SophiaG to prevent NameError if import fails
    class SophiaG(optim.Optimizer):
        def __init__(self, params, lr=1e-3, betas=(0.965, 0.995), rho=0.04, weight_decay=1e-1,
                     gamma=0.99, k=10, eps=1e-12):
            super().__init__(params, dict(lr=lr, betas=betas, rho=rho, weight_decay=weight_decay,
                                          gamma=gamma, k=k, eps=eps))
            logger.warning("Using dummy SophiaG optimizer due to import failure.")
        def step(self, closure=None): return None


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
    elif optimizer_name == 'SophiaG': # Now directly refers to the local class
        if SophiaG is not None:
            optimizer = SophiaG(model.parameters(), lr=learning_rate)
        else:
            logger.warning("SophiaG is not available. Falling back to AdamW.")
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'Lion':
        if Lion is not None:
            optimizer = Lion(model.parameters(), lr=learning_rate)
        else:
            logger.warning("Lion optimizer is not available. Falling back to AdamW.")
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'MADGRAD':
        if MADGRAD is not None:
            optimizer = MADGRAD(model.parameters(), lr=learning_rate)
        else:
            logger.warning("MADGRAD optimizer is not available. Falling back to AdamW.")
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'SAM': # Now directly refers to the local class
        if SAM is not None:
            base_optimizer = optim.AdamW(model.parameters(), lr=learning_rate) # SAM wraps a base optimizer
            optimizer = SAM(model.parameters(), base_optimizer, rho=config['sam_rho'])
        else:
            logger.warning("SAM optimizer is not available. Falling back to AdamW.")
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
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

