# Folder: bongard_solver/
# File: hpo.py
import logging
import os
import argparse # Import argparse for command-line arguments
import json # For saving best config
from typing import Dict, Any, Tuple
# Conditional import for Ray Tune
try:
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.stopper import CombinedStopper, TrialPlateauStopper
    HAS_RAY_TUNE = True
    logger = logging.getLogger(__name__)
    logger.info("Ray Tune found and enabled for HPO.")
except ImportError:
    HAS_RAY_TUNE = False
    logger = logging.getLogger(__name__)
    logger.warning("Ray Tune not found. Hyperparameter optimization will be disabled.")
# Import necessary functions from training.py and config.py
try:
    from training import run_training_once # This function runs a single training trial
    from config import load_config, CONFIG # CONFIG for default values
except ImportError:
    logger.error("Could not import run_training_once or load_config. HPO will not function.")
    run_training_once = None
    load_config = None
logger = logging.getLogger(__name__)

def hpo_train_fn(config_update: Dict[str, Any], initial_cfg: Dict[str, Any]):
    """
    Wrapper function for Ray Tune.
    This function will be called by Ray Tune for each trial.
    It updates the initial configuration with trial-specific hyperparameters
    and then runs a single training session.
    """
    if run_training_once is None:
        raise RuntimeError("run_training_once is not available. HPO cannot proceed.")
    
    # Create a deep copy of the initial configuration to avoid modifying it across trials
    import copy
    trial_cfg = copy.deepcopy(initial_cfg)
    
    # Update the configuration with the current trial's hyperparameters
    # This part needs to be careful about nested dictionaries.
    # A simple update might not be enough for deeply nested configs.
    # For now, assuming top-level updates or simple nested updates.
    for key, value in config_update.items():
        if isinstance(value, dict) and key in trial_cfg and isinstance(trial_cfg[key], dict):
            trial_cfg[key].update(value)
        else:
            trial_cfg[key] = value
    
    # Set a unique seed for each trial if not already handled by run_training_once
    # Ray Tune often handles reproducibility, but explicitly setting seed can be good.
    # trial_cfg['training']['seed'] = tune.get_trial_id() # Example: use trial ID as seed
    
    # Run a single training session with the updated config
    # The `run_training_once` function should return the metric to optimize (e.g., validation accuracy)
    val_accuracy = run_training_once(trial_cfg, epochs=trial_cfg['training']['hpo_epochs'])
    
    # Report the metric to Ray Tune
    tune.report(val_accuracy=val_accuracy)

def run_hyperparameter_optimization(cfg: Dict[str, Any]):
    """
    Main function to orchestrate hyperparameter optimization using Ray Tune.
    """
    if not HAS_RAY_TUNE:
        logger.error("Ray Tune is not installed. Skipping hyperparameter optimization.")
        return
    if run_training_once is None or load_config is None:
        logger.error("Required functions for HPO are missing. Skipping hyperparameter optimization.")
        return
    logger.info("--- Starting Hyperparameter Optimization ---")
    
    # Define the search space for hyperparameters
    # This should be defined in your config.yaml or directly here.
    # Example search space (customize based on your needs)
    search_space = {
        "training": {
            "learning_rate": tune.loguniform(1e-5, 1e-3),
            "batch_size": tune.choice([16, 32, 64]),
            "weight_decay": tune.loguniform(1e-6, 1e-4),
            "optimizer": tune.choice(["AdamW", "ranger", "lion"]), # Assuming these are enabled
            "scheduler": tune.choice(["CosineAnnealingLR", "OneCycleLR"]),
            "hpo_epochs": cfg['training'].get('hpo_epochs', 5) # Number of epochs for each HPO trial
        },
        "model": {
            "backbone": tune.choice(["mobilenet_v3_small", "efficientnet_b0"]),
            "bongard_head_config": {
                "dropout_prob": tune.uniform(0.1, 0.5)
            }
        }
        # Add more hyperparameters to tune as needed
    }
    
    # Configure ASHA scheduler for early stopping of unpromising trials
    scheduler = ASHAScheduler(
        metric="val_accuracy",
        mode="max",
        max_t=cfg['training'].get('hpo_epochs', 5), # Max epochs for a trial
        grace_period=1, # Minimum number of epochs before stopping a trial
        reduction_factor=2 # Reduce the number of trials by this factor at each rung
    )
    
    # Define a stopper to stop trials if they plateau
    stopper = TrialPlateauStopper(
        metric="val_accuracy",
        mode="max",
        patience=cfg['training'].get('hpo_plateau_patience', 3), # Stop if no improvement for X epochs
        std=0.001 # Minimum change to be considered an improvement
    )
    
    # Combine stoppers if needed (e.g., ASHA + TrialPlateauStopper)
    combined_stopper = CombinedStopper(scheduler, stopper)
    
    # Run the Ray Tune experiment
    analysis = tune.run(
        tune.with_parameters(hpo_train_fn, initial_cfg=cfg), # Pass the initial full config
        config=search_space,
        num_samples=cfg['training'].get('hpo_num_samples', 10), # Number of different hyperparameter combinations to try
        scheduler=scheduler, # Use ASHA scheduler
        stop=stopper, # Use plateau stopper
        resources_per_trial={"cpu": cfg['training'].get('hpo_cpus_per_trial', 1), 
                             "gpu": cfg['training'].get('hpo_gpus_per_trial', 0)},
        local_dir=cfg['debug']['ray_tune_dir'], # Directory for Ray Tune results
        name="bongard_hpo_run",
        callbacks=[tune.logger.WandbLoggerCallback(project="bongard_solver_hpo", api_key_file="~/.wandb_api_key")] if cfg['HAS_WANDB'] and cfg['training'].get('use_wandb', False) else None
    )
    logger.info("Hyperparameter Optimization finished.")
    
    # Get the best trial and its configuration
    best_trial = analysis.best_trial
    logger.info(f"Best trial config: {best_trial.config}")
    logger.info(f"Best trial final validation accuracy: {best_trial.last_result['val_accuracy']:.4f}")
    
    # Optionally, save the best configuration
    best_config_path = os.path.join(cfg['debug']['save_model_checkpoints'], "best_hpo_config.json")
    with open(best_config_path, 'w') as f:
        json.dump(best_trial.config, f, indent=4)
    logger.info(f"Best HPO configuration saved to: {best_config_path}")
    return best_trial.config

def main():
    """
    Main entrypoint for running hyperparameter optimization from command line.
    """
    parser = argparse.ArgumentParser(description="Run Hyperparameter Optimization for Bongard Solver.")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to the main configuration YAML file.")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    run_hyperparameter_optimization(cfg)

if __name__ == "__main__":
    main()
