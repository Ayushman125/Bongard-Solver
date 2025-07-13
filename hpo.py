# Folder: bongard_solver/
# File: hpo.py
import logging
import os
import argparse  # Import argparse for command-line arguments
import json  # For saving best config
from typing import Dict, Any, Tuple
import copy  # For deepcopy
import joblib  # For saving Optuna study
# Conditional import for Ray Tune (previous HPO framework)
try:
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.stopper import CombinedStopper, TrialPlateauStopper
    HAS_RAY_TUNE = True
    logger = logging.getLogger(__name__)
    logger.info("Ray Tune found and enabled for HPO (as an alternative).")
except ImportError:
    HAS_RAY_TUNE = False
    logger = logging.getLogger(__name__)
    logger.warning("Ray Tune not found.")
# Conditional import for Optuna (new HPO framework)
try:
    import optuna
    from optuna.pruners import HyperbandPruner
    HAS_OPTUNA = True
    logger = logging.getLogger(__name__)
    logger.info("Optuna found and enabled for HPO.")
except ImportError:
    HAS_OPTUNA = False
    logger = logging.getLogger(__name__)
    logger.warning("Optuna not found. Hyperparameter optimization with Optuna will be disabled.")
# Import necessary functions from training.py and config.py
try:
    from training import run_training_once  # This function runs a single training trial
    from config import load_config, CONFIG  # CONFIG for default values
except ImportError:
    logger.error("Could not import run_training_once or load_config. HPO will not function.")
    run_training_once = None
    load_config = None
logger = logging.getLogger(__name__)
# Optuna logging callback
class OptunaLoggingCallback:
    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            logger.info(f"Trial {trial.number} finished with value: {trial.value:.4f} and parameters: {trial.params}")
        elif trial.state == optuna.trial.TrialState.PRUNED:
            logger.info(f"Trial {trial.number} pruned.")
# --- Optuna Objective Function ---
def objective(trial: optuna.trial.Trial, initial_cfg: Dict[str, Any]) -> float:
    """
    Optuna objective function for hyperparameter optimization.
    This function defines the search space for Optuna and calls the training function.
    Args:
        trial (optuna.trial.Trial): A trial object from Optuna, used to suggest hyperparameters.
        initial_cfg (Dict[str, Any]): The base configuration dictionary.
    Returns:
        float: The validation accuracy to be maximized.
    """
    if run_training_once is None:
        raise RuntimeError("run_training_once is not available. Optuna HPO cannot proceed.")
    
    # Create a deep copy of the initial configuration for this trial
    trial_cfg = copy.deepcopy(initial_cfg)
    
    # Suggest hyperparameters using Optuna's trial object
    # These should match the search space defined in your problem/config.
    trial_cfg['training']['learning_rate'] = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    trial_cfg['training']['batch_size'] = trial.suggest_categorical('batch_size', [16, 32, 64])
    trial_cfg['training']['weight_decay'] = trial.suggest_loguniform('weight_decay', 1e-6, 1e-4)
    trial_cfg['training']['optimizer'] = trial.suggest_categorical('optimizer', ["AdamW", "ranger", "lion"])
    trial_cfg['training']['scheduler'] = trial.suggest_categorical('scheduler', ["CosineAnnealingLR", "OneCycleLR"])
    
    # Nested parameters for model config
    trial_cfg['model']['backbone'] = trial.suggest_categorical('backbone', ["mobilenet_v3_small", "efficientnet_b0"])
    trial_cfg['model']['bongard_head_config']['dropout_prob'] = trial.suggest_uniform('bongard_head_dropout_prob', 0.1, 0.5)
    
    # Set the number of epochs for this trial (can be fixed or suggested)
    hpo_epochs = trial_cfg['training'].get('hpo_epochs', 5)
    
    # Run a single training session with the suggested hyperparameters
    val_accuracy = run_training_once(trial_cfg, epochs=hpo_epochs)
    
    # Report the metric to Optuna for pruning and optimization
    trial.report(val_accuracy, step=hpo_epochs)  # Report final accuracy after all epochs
    
    return val_accuracy
# --- Main HPO Functions (Ray Tune and Optuna) ---
# Ray Tune HPO (kept for reference from previous iterations)
def hpo_train_fn_ray_tune(config_update: Dict[str, Any], initial_cfg: Dict[str, Any]):
    """
    Wrapper function for Ray Tune.
    This function will be called by Ray Tune for each trial.
    It updates the initial configuration with trial-specific hyperparameters
    and then runs a single training session.
    """
    if run_training_once is None:
        raise RuntimeError("run_training_once is not available. HPO cannot proceed.")
    
    trial_cfg = copy.deepcopy(initial_cfg)
    
    for key, value in config_update.items():
        if isinstance(value, dict) and key in trial_cfg and isinstance(trial_cfg[key], dict):
            trial_cfg[key].update(value)
        else:
            trial_cfg[key] = value
    
    val_accuracy = run_training_once(trial_cfg, epochs=trial_cfg['training']['hpo_epochs'])
    tune.report(val_accuracy=val_accuracy)
def run_hyperparameter_optimization_ray_tune(cfg: Dict[str, Any]):
    """
    Main function to orchestrate hyperparameter optimization using Ray Tune.
    """
    if not HAS_RAY_TUNE:
        logger.error("Ray Tune is not installed. Skipping hyperparameter optimization with Ray Tune.")
        return
    if run_training_once is None or load_config is None:
        logger.error("Required functions for HPO are missing. Skipping hyperparameter optimization with Ray Tune.")
        return
    logger.info("--- Starting Hyperparameter Optimization with Ray Tune ---")
    
    search_space = {
        "training": {
            "learning_rate": tune.loguniform(1e-5, 1e-3),
            "batch_size": tune.choice([16, 32, 64]),
            "weight_decay": tune.loguniform(1e-6, 1e-4),
            "optimizer": tune.choice(["AdamW", "ranger", "lion"]),
            "scheduler": tune.choice(["CosineAnnealingLR", "OneCycleLR"]),
            "hpo_epochs": cfg['training'].get('hpo_epochs', 5)
        },
        "model": {
            "backbone": tune.choice(["mobilenet_v3_small", "efficientnet_b0"]),
            "bongard_head_config": {
                "dropout_prob": tune.uniform(0.1, 0.5)
            }
        }
    }
    
    scheduler = ASHAScheduler(
        metric="val_accuracy",
        mode="max",
        max_t=cfg['training'].get('hpo_epochs', 5),
        grace_period=1,
        reduction_factor=2
    )
    
    stopper = TrialPlateauStopper(
        metric="val_accuracy",
        mode="max",
        patience=cfg['training'].get('hpo_plateau_patience', 3),
        std=0.001
    )
    
    combined_stopper = CombinedStopper(scheduler, stopper)
    
    analysis = tune.run(
        tune.with_parameters(hpo_train_fn_ray_tune, initial_cfg=cfg),
        config=search_space,
        num_samples=cfg['training'].get('hpo_num_samples', 10),
        scheduler=scheduler,
        stop=stopper,
        resources_per_trial={"cpu": cfg['training'].get('hpo_cpus_per_trial', 1), 
                             "gpu": cfg['training'].get('hpo_gpus_per_trial', 0)},
        local_dir=cfg['debug']['ray_tune_dir'],
        name="bongard_hpo_run_ray_tune",
        callbacks=[tune.logger.WandbLoggerCallback(project="bongard_solver_hpo", api_key_file="~/.wandb_api_key")] if cfg['HAS_WANDB'] and cfg['training'].get('use_wandb', False) else None
    )
    logger.info("Ray Tune Hyperparameter Optimization finished.")
    
    best_trial = analysis.best_trial
    logger.info(f"Best Ray Tune trial config: {best_trial.config}")
    logger.info(f"Best Ray Tune trial final validation accuracy: {best_trial.last_result['val_accuracy']:.4f}")
    
    best_config_path = os.path.join(cfg['debug']['save_model_checkpoints'], "best_hpo_config_ray_tune.json")
    with open(best_config_path, 'w') as f:
        json.dump(best_trial.config, f, indent=4)
    logger.info(f"Best Ray Tune HPO configuration saved to: {best_config_path}")
    return best_trial.config
# Optuna HPO (new implementation)
def run_hyperparameter_optimization_optuna(cfg: Dict[str, Any], n_trials: int = 30, timeout: Optional[int] = None, n_jobs: int = 1):
    """
    Main function to orchestrate hyperparameter optimization using Optuna.
    """
    if not HAS_OPTUNA:
        logger.error("Optuna is not installed. Skipping hyperparameter optimization with Optuna.")
        return
    if run_training_once is None or load_config is None:
        logger.error("Required functions for HPO are missing. Skipping hyperparameter optimization with Optuna.")
        return
    logger.info("--- Starting Hyperparameter Optimization with Optuna ---")
    
    # Create an Optuna study
    study = optuna.create_study(
        direction="maximize",  # Maximize validation accuracy
        pruner=HyperbandPruner(
            min_resource=1,  # Minimum epoch for a trial
            max_resource=cfg['training'].get('hpo_epochs', 5),  # Max epochs for a trial
            reduction_factor=3  # Factor by which to reduce the number of trials
        ),
        sampler=optuna.samplers.TPESampler(seed=cfg['training']['seed'])  # For reproducibility
    )
    
    # Optimize the objective function
    # Pass timeout and n_jobs to study.optimize(...)
    study.optimize(lambda t: objective(t, cfg),
                   n_trials=n_trials,
                   timeout=timeout, # Exposed via --timeout
                   n_jobs=n_jobs,   # Exposed via --n-jobs
                   callbacks=[OptunaLoggingCallback()])
    
    logger.info("Optuna Hyperparameter Optimization finished.")
    
    # Print best trial results
    logger.info(f"Best Optuna trial: {study.best_trial.value:.4f} (accuracy) with params: {study.best_params}")
    
    # Optionally, save the best configuration
    best_config_path = os.path.join(cfg['debug']['save_model_checkpoints'], "best_hpo_config_optuna.json")
    with open(best_config_path, 'w') as f:
        json.dump(study.best_params, f, indent=4)
    logger.info(f"Best Optuna HPO configuration saved to: {best_config_path}")
    
    # Save study for CI artifacts
    # Use the output_dir from argparse if provided, otherwise default to cfg['hpo']['output_study_path']
    # The `output_dir` argument is now handled in `main` below.
    # The `cfg['hpo']['output_study_path']` is set in `main` before calling this function.
    study_path = cfg['hpo']['output_study_path']
    joblib.dump(study, study_path)
    logger.info(f"Optuna study saved to: {study_path}")
    
    return study.best_params
# --- Main Entry Point for CLI ---
def main():
    """
    Main entrypoint for running hyperparameter optimization from command line.
    Supports both Ray Tune and Optuna.
    """
    parser = argparse.ArgumentParser(description="Run Hyperparameter Optimization for Bongard Solver.")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to the main configuration YAML file.")
    parser.add_argument("--trials", type=int, default=30,
                        help="Number of HPO trials to run (for Optuna).")
    parser.add_argument("--hpo_framework", type=str, default="optuna",
                        choices=["optuna", "ray_tune"],
                        help="Choose HPO framework: 'optuna' or 'ray_tune'.")
    parser.add_argument("--output_dir", type=str, default="./hpo_results",
                        help="Directory to save HPO results (study, best config).")  # Added output_dir argument
    
    # 9.1 Expose --timeout and --n-jobs
    parser.add_argument('--timeout', type=int, default=None, # Default to None to use Optuna's default or no timeout
                        help="Stop optimization after the given number of seconds.")
    parser.add_argument('--n-jobs', type=int, default=1, # Default to 1 (sequential)
                        help="The number of parallel jobs. Set to -1 to use all available CPU cores.")

    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)  # Ensure output directory exists
    cfg = load_config(args.config)
    # Update config with output_dir for saving study and best config
    cfg['hpo']['output_study_path'] = os.path.join(args.output_dir, cfg['hpo'].get('output_study_filename', 'optuna_study.pkl'))
    cfg['debug']['save_model_checkpoints'] = args.output_dir  # Use output_dir for checkpoints too
    
    if args.hpo_framework == "optuna":
        # Pass timeout and n_jobs to the Optuna function
        run_hyperparameter_optimization_optuna(cfg, n_trials=args.trials, timeout=args.timeout, n_jobs=args.n_jobs)
    elif args.hpo_framework == "ray_tune":
        run_hyperparameter_optimization_ray_tune(cfg)
if __name__ == "__main__":
    main()
