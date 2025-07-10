# Folder: bongard_solver/
# File: main.py

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import os
import logging
import random
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from multiprocessing import Pool # For CPU-bound worker pool

# Import configuration
from config import CONFIG, DEVICE, HAS_WANDB

# Import data loading components
from data import (
    BongardSyntheticDataset, RealBongardDataset,
    CurriculumSampler, build_dali_image_processor,
    BongardGenerator # For synthetic data generation
)

# Import model and training components
from models import BongardSolverEnsemble
from optimizers import get_optimizer, get_scheduler
from replay_buffer import KnowledgeReplayBuffer
from training import _run_single_training_session_ensemble # The main training function

# Import pipeline_workers (new module)
try:
    from pipeline_workers import process_image_worker
    HAS_PIPELINE_WORKERS = True
    logger.info("pipeline_workers.py found.")
except ImportError:
    HAS_PIPELINE_WORKERS = False
    logger.warning("pipeline_workers.py not found. CPU-bound preprocessing will not be parallelized via multiprocessing.Pool.")


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- DDP Setup and Teardown ---
def setup(rank: int, world_size: int):
    """Initializes the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355' # Use a free port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    logger.info(f"Rank {rank} / {world_size} initialized.")

def cleanup():
    """Destroys the distributed environment."""
    dist.destroy_process_group()
    logger.info("Distributed process group destroyed.")

# --- Custom Collate Function for PyTorch DataLoader ---
def custom_collate_fn(batch: List[Tuple[np.ndarray, np.ndarray, int, bytes, bytes, float, np.ndarray, np.ndarray, int, List[np.ndarray], List[int], List[bytes], int]]) -> Tuple[List[np.ndarray], List[np.ndarray], torch.Tensor, List[bytes], List[bytes], torch.Tensor, List[np.ndarray], List[np.ndarray], torch.Tensor, List[np.ndarray], List[int], List[bytes], torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Custom collate function for PyTorch DataLoader to handle variable-length
    support image lists and prepare data for DALI and PER.
    """
    query_img1_batch_np = []
    query_img2_batch_np = []
    query_labels = []
    query_sg1_bytes = []
    query_sg2_bytes = []
    difficulties = []
    affine1 = []
    affine2 = []
    original_indices = []

    support_imgs_flat_batch_np = []
    support_labels_flat = []
    support_sgs_flat_bytes = []
    num_support_per_problem = []

    tree_indices_batch = []
    is_weights_batch = []

    max_support_images = CONFIG['data']['synthetic_data_config']['max_support_images_per_problem']
    image_size = CONFIG['data']['image_size'] # Use the final image size for padding

    # Assuming all samples in the batch have the same structure
    # The dataset's __getitem__ returns:
    # (query_img1_np, query_img2_np, query_label, query_sg1_bytes, query_sg2_bytes,
    #  difficulty, affine_matrix_view1, affine_matrix_view2, original_index,
    #  support_imgs_np_list, support_labels, support_sgs_bytes_list,
    #  num_actual_support_images)

    for item in batch:
        query_img1_batch_np.append(item[0])
        query_img2_batch_np.append(item[1])
        query_labels.append(item[2])
        query_sg1_bytes.append(item[3])
        query_sg2_bytes.append(item[4])
        difficulties.append(item[5])
        affine1.append(item[6])
        affine2.append(item[7])
        original_indices.append(item[8])

        # Handle support images: flatten and pad
        current_support_imgs = item[9] # List of np.ndarray
        current_support_labels = item[10] # List of int
        current_support_sgs = item[11] # List of bytes
        num_actual_support = item[12] # int

        num_support_per_problem.append(num_actual_support)

        for img_np, lbl, sg_bytes in zip(current_support_imgs, current_support_labels, current_support_sgs):
            support_imgs_flat_batch_np.append(img_np)
            support_labels_flat.append(lbl)
            support_sgs_flat_bytes.append(sg_bytes)
        
        # Pad with dummy images/labels/SGs if less than max_support_images
        for _ in range(num_actual_support, max_support_images):
            # Use a zero-filled image as padding for missing support images
            support_imgs_flat_batch_np.append(np.zeros((image_size, image_size, 3), dtype=np.uint8))
            support_labels_flat.append(-1) # Dummy label for padded images
            support_sgs_flat_bytes.append(json.dumps({}).encode('utf-8')) # Empty SG for padded

    # If PER is enabled, retrieve the tree_indices and is_weights for this batch
    # This assumes the CurriculumSampler has already set this information in the replay_buffer
    # for the current batch of `original_indices`.
    if CONFIG['training']['curriculum_config']['difficulty_sampling']:
        # This part needs to be carefully handled. The `collate_fn` is called by the DataLoader,
        # which might be in a separate process. The `replay_buffer` is typically shared.
        # The `CurriculumSampler` *yields* the original indices. The `replay_buffer`
        # should provide the PER info when those indices are sampled.
        # A simple way is to make `replay_buffer`'s `get_current_batch_info`
        # accessible or to pass `tree_indices` and `is_weights` directly from the sampler
        # into the dataset's `__getitem__` if that's feasible.
        # For now, let's assume `replay_buffer.get_current_batch_info()`
        # can retrieve the info for the `original_indices` in this batch.
        
        # This is a conceptual placeholder. In a real multi-process PER setup,
        # the sampler would directly yield (index, tree_index, is_weight) tuples,
        # and the collate_fn would unpack them.
        # For simplicity in this `collate_fn`, we will assume `tree_indices_batch`
        # and `is_weights_batch` are passed through the `batch` if PER is active.
        # This means `dataset.__getitem__` would need to return them, or the sampler
        # would modify the batch items before `collate_fn`.
        
        # Let's adjust the dataset's __getitem__ to return PER info.
        # (This was done in the previous `data.py` update, so this collate_fn is now correct).
        # The `item` tuple from `dataset.__getitem__` should now include `tree_index` and `is_weight`
        # as the last two elements if PER is enabled.
        
        # Re-evaluating the `dataset.__getitem__` signature:
        # (query_img1_np, query_img2_np, query_label, query_sg1_bytes, query_sg2_bytes,
        #  difficulty, affine_matrix_view1, affine_matrix_view2, original_index,
        #  support_imgs_np_list, support_labels, support_sgs_bytes_list,
        #  num_actual_support_images, tree_index, is_weight)

        # So, we need to extract `tree_index` and `is_weight` from each `item` in the batch
        # if PER is active.
        for item in batch:
            # Check if PER info is present in the item (last two elements)
            if len(item) > 13: # If tree_index and is_weight are present
                tree_indices_batch.append(item[13])
                is_weights_batch.append(item[14])
            else:
                tree_indices_batch.append(None)
                is_weights_batch.append(None)

        # Convert to tensors if not None
        tree_indices_tensor = torch.tensor(tree_indices_batch, dtype=torch.long) if tree_indices_batch[0] is not None else None
        is_weights_tensor = torch.tensor(is_weights_batch, dtype=torch.float) if is_weights_batch[0] is not None else None
    else:
        tree_indices_tensor = None
        is_weights_tensor = None


    return (
        query_img1_batch_np, # List of np.ndarray (for DALI)
        query_img2_batch_np, # List of np.ndarray (for DALI)
        torch.tensor(query_labels, dtype=torch.long),
        query_sg1_bytes, # List of bytes
        query_sg2_bytes, # List of bytes
        torch.tensor(difficulties, dtype=torch.float),
        np.stack(affine1), # np.ndarray
        np.stack(affine2), # np.ndarray
        torch.tensor(original_indices, dtype=torch.long),
        support_imgs_flat_batch_np, # List of np.ndarray (for DALI)
        torch.tensor(support_labels_flat, dtype=torch.long),
        support_sgs_flat_bytes, # List of bytes
        torch.tensor(num_support_per_problem, dtype=torch.long),
        tree_indices_tensor, # Optional Tensor
        is_weights_tensor # Optional Tensor
    )


# --- Main Training Process for a Single GPU ---
def run_training_process(rank: int, world_size: int):
    """
    Main function for each DDP training process.
    """
    if world_size > 1:
        setup(rank, world_size)
    
    set_seed(CONFIG['training']['seed'] + rank) # Set different seed for each rank
    logger.info(f"Process {rank}: Starting training session.")

    # --- Data Loading ---
    data_config = CONFIG['data']
    training_config = CONFIG['training']

    bongard_generator = BongardGenerator(data_config, ALL_BONGARD_RULES)

    if data_config['use_synthetic_data']:
        dataset = BongardSyntheticDataset(data_config, bongard_generator, num_samples=training_config['batch_size'] * 1000) # Example num_samples
    else:
        train_data_list, val_data_list = load_bongard_data(
            data_config['data_root_path'],
            data_config['real_data_config']['dataset_name'],
            data_config['real_data_config']['train_split_ratio']
        )
        dataset = RealBongardDataset(train_data_list) # Use train_data_list for training

    # Initialize CurriculumSampler
    curriculum_sampler = CurriculumSampler(
        dataset,
        training_config,
        num_replicas=world_size,
        rank=rank
    )
    # Assign sampler to dataset for image size updates
    dataset.curriculum_sampler = curriculum_sampler

    # PyTorch DataLoader
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=training_config['batch_size'],
        sampler=curriculum_sampler, # Use custom sampler
        num_workers=data_config['dataloader_workers'],
        pin_memory=True,
        collate_fn=custom_collate_fn,
        drop_last=True # Ensure full batches
    )
    logger.info(f"Process {rank}: DataLoader created with batch size {training_config['batch_size']} and {data_config['dataloader_workers']} workers.")

    # Validation DataLoader (without PER or curriculum effects)
    if data_config['use_synthetic_data']:
        val_dataset = BongardSyntheticDataset(data_config, bongard_generator, num_samples=training_config['batch_size'] * 100) # Smaller val set
    else:
        # For real data, use the val_data_list
        val_dataset = RealBongardDataset(val_data_list)
    
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        sampler=val_sampler,
        shuffle=(val_sampler is None), # Shuffle only if not using DistributedSampler
        num_workers=data_config['dataloader_workers'],
        pin_memory=True,
        collate_fn=custom_collate_fn,
        drop_last=True
    )
    logger.info(f"Process {rank}: Validation DataLoader created.")

    # --- DALI Image Processor (per GPU) ---
    dali_image_processor = build_dali_image_processor(
        batch_size=training_config['batch_size'],
        num_threads=data_config['dataloader_workers'],
        device_id=rank, # Each process gets its own GPU
        image_size=data_config['image_size'], # Fixed target size for DALI resize
        is_training=True,
        curriculum_config=training_config['curriculum_config']
    )
    logger.info(f"Process {rank}: DALIImageProcessor built.")

    # --- Model Initialization ---
    model = BongardSolverEnsemble(CONFIG['model']).to(rank)
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True) # find_unused_parameters=True if some parts of graph are not always used
    logger.info(f"Process {rank}: Model initialized and moved to device {rank}.")

    # --- Optimizer and Scheduler ---
    optimizer = get_optimizer(
        model=model,
        optimizer_name=training_config['optimizer'],
        learning_rate=training_config['learning_rate'],
        weight_decay=training_config['weight_decay'],
        sam_rho=training_config['sam_rho'] if training_config['use_sam'] else 0.0
    )
    scheduler = get_scheduler(
        optimizer=optimizer,
        scheduler_name=training_config['scheduler'],
        scheduler_config=training_config['scheduler_config'],
        total_epochs=training_config['epochs'],
        steps_per_epoch=len(train_loader) # For OneCycleLR
    )
    logger.info(f"Process {rank}: Optimizer ({training_config['optimizer']}) and Scheduler ({training_config['scheduler']}) initialized.")

    # --- Replay Buffer (for PER) ---
    replay_buffer = None
    if training_config['curriculum_learning'] and training_config['curriculum_config']['difficulty_sampling']:
        # The replay buffer is initialized by the CurriculumSampler, which is passed to the DataLoader.
        # We need a way to access it from the training loop.
        # The sampler itself holds the replay_buffer.
        replay_buffer = curriculum_sampler.replay_buffer
        logger.info(f"Process {rank}: KnowledgeReplayBuffer initialized.")

    # --- Teacher Models for Distillation ---
    teacher_models = []
    if training_config['use_knowledge_distillation'] and CONFIG['ensemble']['num_members'] > 0:
        logger.info(f"Process {rank}: Loading {CONFIG['ensemble']['num_members']} teacher models for distillation...")
        for i in range(CONFIG['ensemble']['num_members']):
            teacher_model_path = os.path.join(CONFIG['debug']['save_model_checkpoints'], f"best_model_member_{i}.pth")
            if os.path.exists(teacher_model_path):
                teacher_model = BongardSolverEnsemble(CONFIG['model']).to(rank)
                checkpoint = torch.load(teacher_model_path, map_location=f'cuda:{rank}')
                teacher_model.load_state_dict(checkpoint['model_state_dict'])
                teacher_model.eval() # Teachers should be in eval mode
                teacher_models.append(teacher_model)
                logger.info(f"Process {rank}: Loaded teacher model {i} from {teacher_model_path}")
            else:
                logger.warning(f"Process {rank}: Teacher model {i} not found at {teacher_model_path}. Skipping this teacher.")
        if not teacher_models:
            logger.warning(f"Process {rank}: Knowledge distillation enabled but no teacher models were loaded.")

    # --- Run Training Session ---
    logger.info(f"Process {rank}: Starting training loop for {training_config['epochs']} epochs.")
    best_accuracy, best_loss, best_metrics = _run_single_training_session_ensemble(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        current_rank=rank,
        is_ddp_initialized=(world_size > 1),
        model_idx=rank, # Each rank trains a "member" if thinking of ensemble training
        replay_buffer=replay_buffer,
        start_epoch=0,
        total_epochs=training_config['epochs'],
        teacher_models=teacher_models,
        dali_image_processor=dali_image_processor # Pass the DALI processor
    )
    logger.info(f"Process {rank}: Training finished. Best Val Accuracy: {best_accuracy:.4f}, Best Val Loss: {best_loss:.4f}")

    # Release DALI pipeline resources
    dali_image_processor.release()

    if world_size > 1:
        cleanup()

# --- Main Execution ---
def main():
    """Main entry point for the Bongard Solver training."""
    # Set the start method for multiprocessing
    # This must be called once in the main block before any processes are spawned.
    mp.set_start_method("spawn", force=True)
    logger.info("Multiprocessing start method set to 'spawn'.")

    # Determine number of GPUs for distributed training
    world_size = torch.cuda.device_count()
    if world_size > 1:
        logger.info(f"Found {world_size} GPUs. Starting distributed training.")
        mp.spawn(run_training_process, args=(world_size,), nprocs=world_size, join=True)
    else:
        logger.info("Running on a single GPU/CPU.")
        run_training_process(0, 1) # Rank 0, world_size 1

    logger.info("--- Bongard Solver Training Workflow finished. ---")

    # --- Example of using pipeline_workers.py for offline preprocessing ---
    # This part would typically run *before* the main training loop,
    # to preprocess a large dataset of raw images into structured scene graphs
    # or other features that are then loaded by the DataLoader.
    if HAS_PIPELINE_WORKERS:
        logger.info("\n--- Demonstrating pipeline_workers.py for offline preprocessing ---")
        dummy_image_paths = ["./data/dummy_image_1.png", "./data/dummy_image_2.png"] # Replace with actual paths
        # Create dummy image files for demonstration
        if not os.path.exists("./data"): os.makedirs("./data")
        Image.new('RGB', (224, 224), color = 'red').save(dummy_image_paths[0])
        Image.new('RGB', (224, 224), color = 'blue').save(dummy_image_paths[1])
        
        num_workers = min(os.cpu_count(), 4) # Limit workers for example
        logger.info(f"Launching a Pool with {num_workers} workers to process dummy images.")
        
        args_for_workers = [(path, CONFIG, i, num_workers) for i, path in enumerate(dummy_image_paths)]
        
        try:
            with Pool(processes=num_workers) as pool:
                # starmap applies the function with multiple arguments from each tuple in args_for_workers
                preprocessing_results = pool.starmap(process_image_worker, args_for_workers)
            logger.info("Preprocessing results from pipeline_workers:")
            for res in preprocessing_results:
                logger.info(f"  Processed {res['image_path']}: Objects={len(res['inferred_scene_graph']['objects'])}")
        except Exception as e:
            logger.error(f"Error during multiprocessing with pipeline_workers: {e}")
        finally:
            # Clean up dummy files
            for p in dummy_image_paths:
                if os.path.exists(p): os.remove(p)


if __name__ == "__main__":
    main()

