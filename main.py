# Folder: bongard_solver/
# File: main.py
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import os
import logging
import random
import numpy as np
import json  # Import json for handling scene graph bytes
from typing import List, Dict, Any, Tuple, Optional
from multiprocessing import Pool   # For CPU-bound worker pool
from PIL import Image  # For creating dummy images
import pytorch_lightning as pl  # Import PyTorch Lightning

# Import configuration
# Assuming config.py is in the same directory and contains CONFIG dictionary
try:
    from config import CONFIG, DEVICE, HAS_WANDB, load_config, IMAGENET_MEAN, IMAGENET_STD
except ImportError:
    # Fallback for testing if config.py is not directly available
    logging.warning("Could not import CONFIG from config.py. Using a minimal dummy config.")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    HAS_WANDB = False
    CONFIG = {
        'data': {'image_size': 224, 'dataloader_workers': 0, 'use_synthetic_data': True,
                 'synthetic_data_config': {'max_support_images_per_problem': 5}},
        'training': {'epochs': 1, 'batch_size': 2, 'seed': 42, 'use_amp': False,
                     'curriculum_learning': False, 'curriculum_config': {'difficulty_sampling': False},
                     'learning_rate': 1e-4, 'scheduler_config': {'CosineAnnealingLR': {'eta_min': 1e-6}},
                     'use_sam': False, 'sam_rho': 0.05, 'weight_decay': 0.0,
                     'use_mixup_cutmix': False, 'max_grad_norm': 0.0,
                     'validation_frequency_batches': 50, 'log_interval_batches': 10,
                     'consistency_loss_weight': 1.0, 'feature_consistency_weight': 0.5,
                     'symbolic_consistency_weight': 0.5, 'relation_loss_weight': 1.0,
                     'use_knowledge_distillation': False, 'distillation_config': {},
                     'use_rl_reformulation': False, 'rl_config': {},
                     'pruning': {'enabled': False}, 'quantization': {'qat': False, 'ptq': False, 'export_quantized_model': False}
                     },
        'model': {'use_mask_rcnn': False, 'use_persistent_homology': False, 'ph_feature_dim': 64,
                  'pretrained': True, 'n_attributes': 5, # Added for LitAttributeModel
                  'attribute_classifier_config': {'shape': 3, 'color': 3, 'fill': 2, 'size': 3, 'orientation': 2, 'texture': 2},
                  'bongard_head_config': {'num_classes': 2},
                  'simclr_config': {'enabled': False, 'pretrain_epochs': 0, 'projection_dim': 128, 'temperature': 0.5},
                  'support_set_encoder_config': {'output_dim': 128},
                  'relation_gnn_config': {'hidden_dim': 128}
                  },
        'debug': {'save_model_checkpoints': './checkpoints', 'save_grad_cam_outputs': './grad_cam_outputs',
                  'rule_eval_log_interval_batches': 100},
        'few_shot': {'enable': False, 'n_way': 2, 'k_shot': 1, 'q_query': 1, 'episodes': 10} # Added for few-shot
    }
    def load_config(path): return CONFIG  # Dummy load_config
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

# Import data loading components
try:
    from data import (
        BongardSyntheticDataset, RealBongardDataset,
        CurriculumSampler, build_dali_image_processor,
        BongardGenerator, get_dataloader, custom_collate_fn # Import get_dataloader and custom_collate_fn
    )
    from torch.utils.data.distributed import DistributedSampler
except ImportError:
    logging.warning("Could not import data loading components. Using dummy classes.")
    # Dummy classes to allow main.py to run for demonstration
    class BongardSyntheticDataset:
        def __init__(self, config, generator, num_samples):
            self.num_samples = num_samples
            self.image_size = config['data']['image_size']
            self.labels = [random.randint(0, 1) for _ in range(num_samples)] # Dummy labels for few-shot
        def __len__(self): return self.num_samples
        def __getitem__(self, idx):
            # Return dummy data matching custom_collate_fn's expectation
            dummy_img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            dummy_sg_bytes = json.dumps({}).encode('utf-8')
            return (dummy_img, dummy_img, 0, dummy_sg_bytes, dummy_sg_bytes,
                    0.5, np.eye(3), np.eye(3), idx,
                    [dummy_img] * CONFIG['data']['synthetic_data_config']['max_support_images_per_problem'],
                    [0] * CONFIG['data']['synthetic_data_config']['max_support_images_per_problem'],
                    [dummy_sg_bytes] * CONFIG['data']['synthetic_data_config']['max_support_images_per_problem'],
                    torch.tensor(CONFIG['data']['synthetic_data_config']['max_support_images_per_problem']),
                    None, None) # Added None for tree_indices, is_weights
        def set_image_size(self, size): pass
        def curriculum_sampler(self): return None
    class RealBongardDataset(BongardSyntheticDataset): pass  # Inherit for simplicity
    class CurriculumSampler:
        def __init__(self, dataset, training_config, num_replicas, rank): self.dataset = dataset; self.replay_buffer = None
        def __iter__(self): yield from range(len(self.dataset))
        def __len__(self): return len(self.dataset)
        def set_epoch(self, epoch): pass
        def update_priorities(self, indices, td_errors): pass
    def build_dali_image_processor(*args, **kwargs):
        class DummyDALIProcessor:
            def run(self, *inputs): # DALI run now expects multiple inputs
                # Return dummy processed tensors based on expected output types
                # For query images (view1, view2) and flattened support images
                img_size = CONFIG['data']['image_size']
                batch_size = len(inputs[0]) # Number of query images
                
                # Dummy processed query images (B, C, H, W)
                processed_query1 = torch.randn(batch_size, 3, img_size, img_size)
                processed_query2 = torch.randn(batch_size, 3, img_size, img_size)
                
                # Dummy processed support images (B*N_support, C, H, W)
                # Assuming inputs[2] is raw_support_images_flat_np
                num_support_flat = len(inputs[2])
                processed_support_flat = torch.randn(num_support_flat, 3, img_size, img_size)
                
                return processed_query1, processed_query2, processed_support_flat
            def release(self): pass
        return DummyDALIProcessor()
    class BongardGenerator:
        def __init__(self, data_config, rules): pass
        def generate_problem(self, difficulty=1.0): return None
    class DistributedSampler:
        def __init__(self, dataset, num_replicas, rank, shuffle): self.dataset = dataset
        def __iter__(self): yield from range(len(self.dataset))
        def __len__(self): return len(self.dataset)
        def set_epoch(self, epoch): pass
    def custom_collate_fn(batch): # Dummy collate_fn for fallback
        # This dummy collate_fn needs to match the expected output of the real one
        # (processed_query_images_view1, processed_query_images_view2, query_labels,
        #  query_gts_json_view1, query_gts_json_view2,
        #  processed_support_images_reshaped, support_labels_reshaped, support_sgs_reshaped,
        #  difficulties, original_indices, tree_indices, is_weights)
        
        # For LitBongard, we need a more complex batch structure.
        # Let's create dummy tensors that mimic the expected output from data.py
        batch_size = len(batch)
        img_size = CONFIG['data']['image_size']
        max_support_imgs = CONFIG['data']['synthetic_data_config']['max_support_images_per_problem']

        # Dummy processed images (B, C, H, W)
        processed_query_images_view1 = torch.randn(batch_size, 3, img_size, img_size)
        processed_query_images_view2 = torch.randn(batch_size, 3, img_size, img_size)
        
        # Dummy query labels (B,)
        query_labels = torch.randint(0, 2, (batch_size,))
        
        # Dummy GT JSONs (List[bytes])
        query_gts_json_view1 = [json.dumps({'objects': [], 'relations': []}).encode('utf-8')] * batch_size
        query_gts_json_view2 = [json.dumps({'objects': [], 'relations': []}).encode('utf-8')] * batch_size
        
        # Dummy support images (B, N_support, C, H, W)
        processed_support_images_reshaped = torch.randn(batch_size, max_support_imgs, 3, img_size, img_size)
        
        # Dummy support labels (B, N_support)
        support_labels_reshaped = torch.randint(0, 2, (batch_size, max_support_imgs))
        
        # Dummy support SGs (List[List[bytes]])
        support_sgs_reshaped = [[json.dumps({}).encode('utf-8')] * max_support_imgs] * batch_size
        
        # Dummy difficulties, original_indices, num_support_per_problem
        difficulties = torch.rand(batch_size)
        original_indices = torch.arange(batch_size)
        num_support_per_problem = torch.full((batch_size,), max_support_imgs, dtype=torch.long)
        
        # Dummy tree_indices and is_weights for PER (optional)
        tree_indices = torch.arange(batch_size)
        is_weights = torch.ones(batch_size)
        
        return (processed_query_images_view1, processed_query_images_view2, query_labels,
                query_gts_json_view1, query_gts_json_view2,
                processed_support_images_reshaped, support_labels_reshaped, support_sgs_reshaped,
                difficulties, original_indices, num_support_per_problem,
                tree_indices, is_weights)


# Import model and training components
try:
    from models import LitBongard # Import LitBongard for the main training loop
    # from optimizers import get_optimizer, get_scheduler # No longer needed directly by main.py
    # from replay_buffer import KnowledgeReplayBuffer # No longer needed directly by main.py
    # from training import _run_single_training_session_ensemble   # No longer needed directly by main.py
    # from torch.nn.parallel import DistributedDataParallel as DDP # PyTorch Lightning handles DDP
except ImportError:
    logging.warning("Could not import model/training components. Using dummy classes.")
    class LitBongard(pl.LightningModule): # Dummy LitBongard
        def __init__(self, cfg): super().__init__(); self.linear = torch.nn.Linear(10, 2)
        def forward(self, images, ground_truth_json_strings, support_images): return {'bongard_logits': torch.randn(images.shape[0], 2)}
        def training_step(self, batch, batch_idx): return torch.tensor(0.0)
        def validation_step(self, batch, batch_idx): return torch.tensor(0.0)
        def configure_optimizers(self): return [torch.optim.Adam(self.parameters(), lr=1e-3)], []
    # Dummy classes for other imports if needed for other parts of main.py
    class BongardSolverEnsemble(torch.nn.Module):
        def __init__(self, config): super().__init__(); self.linear = torch.nn.Linear(10, 2)
        def forward(self, x): return torch.randn(x[0].shape[0], 2)  # Dummy output

# Import pipeline_workers (new module)
try:
    from pipeline_workers import process_image_worker
    HAS_PIPELINE_WORKERS = True
    logger.info("pipeline_workers.py found.")
except ImportError:
    HAS_PIPELINE_WORKERS = False
    logger.warning("pipeline_workers.py not found. CPU-bound preprocessing will not be parallelized via multiprocessing.Pool.")
# Import scene_graph_builder (newly updated)
try:
    from scene_graph_builder import build_scene_graph
    HAS_SCENE_GRAPH_BUILDER = True
    logger.info("scene_graph_builder.py found.")
except ImportError:
    HAS_SCENE_GRAPH_BUILDER = False
    logger.warning("scene_graph_builder.py not found. Scene graph building will be skipped.")
# Import utils for set_seed
try:
    from utils import set_seed
except ImportError:
    logger.warning("Could not import set_seed from utils.py. Using local dummy function.")
    def set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.info(f"Random seed set to {seed}")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Main Training Process for a Single GPU (now handled by PyTorch Lightning) ---
def run_lightning_training(rank: int, world_size: int, cfg: Dict[str, Any]):
    """
    Main function for each DDP training process, using PyTorch Lightning.
    """
    if world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        logger.info(f"Rank {rank} / {world_size} initialized for Lightning Trainer.")
    
    set_seed(cfg['training']['seed'] + rank) # Set different seed for each rank
    logger.info(f"Process {rank}: Starting Lightning training session.")

    # --- Data Loading ---
    # get_dataloader now handles few-shot and DALI internally
    train_loader = get_dataloader(cfg, is_train=True, rank=rank, world_size=world_size)
    val_loader   = get_dataloader(cfg, is_train=False, rank=rank, world_size=world_size)
    
    logger.info(f"Process {rank}: Train DataLoader created with batch size {cfg['training']['batch_size']}.")
    logger.info(f"Process {rank}: Validation DataLoader created.")

    # --- Model Initialization ---
    # Instantiate LitBongard for the main training
    model = LitBongard(cfg)
    logger.info(f"Process {rank}: LitBongard model initialized.")

    # --- PyTorch Lightning Trainer ---
    trainer = pl.Trainer(
        max_epochs=cfg['training']['epochs'],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None, # Use 1 device per process
        precision=16 if cfg['training']['use_amp'] else 32,
        logger=pl.loggers.TensorBoardLogger("lightning_logs/", name="bongard_solver"),
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=cfg['debug']['save_model_checkpoints'],
                filename='best_bongard_model', # Changed filename for Bongard model
                monitor='val/loss', # Monitor validation loss
                mode='min',
                save_top_k=1
            ),
            pl.callbacks.LearningRateMonitor(logging_interval='epoch')
        ],
        strategy="ddp" if world_size > 1 else "auto", # Use DDP strategy for multi-GPU
        log_every_n_steps=cfg['training']['log_interval_batches'],
        val_check_interval=cfg['training']['validation_frequency_batches'] if val_loader else 1.0 # Validate every N batches or every epoch
    )
    logger.info(f"Process {rank}: PyTorch Lightning Trainer initialized.")

    # --- Fit the model ---
    trainer.fit(model, train_loader, val_loader)
    logger.info(f"Process {rank}: Training finished.")

    if world_size > 1:
        dist.destroy_process_group()
        logger.info(f"Process {rank}: Distributed process group destroyed.")

# --- Main Execution ---
def main():
    """Main entry point for the Bongard Solver training and inference demonstration."""
    # Set the start method for multiprocessing
    mp.set_start_method("spawn", force=True)
    logger.info("Multiprocessing start method set to 'spawn'.")
    
    # Load configuration
    global CONFIG # Ensure we modify the global CONFIG
    try:
        # Assuming config.py is imported directly, not loaded from a YAML file
        # If config is a .py file, it's already loaded when imported.
        # If you intend to load from config.yaml, uncomment the line below.
        # CONFIG = load_config("config.yaml") 
        logger.info("Configuration loaded from config.py (module import).")
    except Exception as e:
        logger.error(f"Error loading CONFIG: {e}. Using default internal configuration.")

    # Determine number of GPUs for distributed training
    world_size = torch.cuda.device_count()
    if world_size > 1:
        logger.info(f"Found {world_size} GPUs. Starting distributed training.")
        # Use mp.spawn for multi-GPU training with PyTorch Lightning
        mp.spawn(run_lightning_training, args=(world_size, CONFIG), nprocs=world_size, join=True)
    else:
        logger.info("Running on a single GPU/CPU.")
        run_lightning_training(0, 1, CONFIG)
    
    logger.info("--- Bongard Solver Training Workflow finished. ---")

    # --- Demonstrate Scene Graph Building with new components ---
    if HAS_SCENE_GRAPH_BUILDER:
        logger.info("\n--- Demonstrating Scene Graph Building with Mask R-CNN and Persistent Homology ---")
        # Create a dummy image for demonstration
        dummy_image_path = "./data/demo_image.png"
        if not os.path.exists("./data"): os.makedirs("./data")
        
        # Create a more complex dummy image for better demonstration
        demo_image = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.circle(demo_image, (70, 70), 30, (255, 255, 255), -1)  # White circle
        cv2.circle(demo_image, (70, 70), 15, (0, 0, 0), -1)  # Black circle inside (hole)
        cv2.rectangle(demo_image, (180, 50), (230, 100), (0, 255, 0), -1)  # Green square
        cv2.ellipse(demo_image, (128, 180), (60, 30), 0, 0, 360, (255, 0, 0), -1)  # Red ellipse
        
        Image.fromarray(demo_image).save(dummy_image_path)
        logger.info(f"Created dummy image at {dummy_image_path}")
        try:
            # Load the dummy image
            image_to_process = cv2.imread(dummy_image_path)
            if image_to_process is None:
                raise FileNotFoundError(f"Could not load image: {dummy_image_path}")
            image_to_process = cv2.cvtColor(image_to_process, cv2.COLOR_BGR2RGB)  # Convert to RGB
            # Build the scene graph using the updated function
            scene_graph = build_scene_graph(image_to_process)
            logger.info("\n--- Inferred Scene Graph for Demo Image ---")
            if scene_graph:
                for i, obj in enumerate(scene_graph):
                    logger.info(f"Object {i+1}:")
                    for attr, value in obj.items():
                        if attr == 'topo': # Changed from topology_features to topo as per scene_graph_builder
                            logger.info(f"  {attr}: [features of length {len(value)}]")
                        elif attr == 'relations':
                            logger.info(f"  {attr}: {value}")
                        else:
                            logger.info(f"  {attr}: {value}")
            else:
                logger.warning("No objects detected or scene graph building failed for the demo image.")
        except Exception as e:
            logger.error(f"Error during scene graph demonstration: {e}")
        finally:
            # Clean up dummy file
            if os.path.exists(dummy_image_path):
                os.remove(dummy_image_path)
                logger.info(f"Cleaned up dummy image: {dummy_image_path}")
    else:
        logger.warning("Skipping scene graph demonstration as scene_graph_builder.py is not available.")
    
    # --- Example of using pipeline_workers.py for offline preprocessing ---
    if HAS_PIPELINE_WORKERS:
        logger.info("\n--- Demonstrating pipeline_workers.py for offline preprocessing ---")
        dummy_image_paths = ["./data/worker_image_1.png", "./data/worker_image_2.png"]
        
        if not os.path.exists("./data"): os.makedirs("./data")
        Image.new('RGB', (224, 224), color = 'red').save(dummy_image_paths[0])
        Image.new('RGB', (224, 224), color = 'blue').save(dummy_image_paths[1])
        
        num_workers = min(os.cpu_count() or 1, 4)  # Ensure at least 1 worker if os.cpu_count() is None
        logger.info(f"Launching a Pool with {num_workers} workers to process dummy images.")
        
        args_for_workers = [(path, CONFIG, i, num_workers) for i, path in enumerate(dummy_image_paths)]
        
        try:
            with Pool(processes=num_workers) as pool:
                preprocessing_results = pool.starmap(process_image_worker, args_for_workers)
            logger.info("Preprocessing results from pipeline_workers:")
            for res in preprocessing_results:
                logger.info(f"  Processed {res['image_path']}: Objects={len(res['inferred_scene_graph']['objects'])}")
        except Exception as e:
            logger.error(f"Error during multiprocessing with pipeline_workers: {e}")
        finally:
            for p in dummy_image_paths:
                if os.path.exists(p): os.remove(p)
                logger.info(f"Cleaned up worker image: {p}")
    logger.info("--- Bongard Solver Main Execution finished. ---")

if __name__ == "__main__":
    main()
