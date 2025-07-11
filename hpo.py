# Folder: bongard_solver/
# File: hpo.py
import optuna
import logging
import os
import sys

# Add the parent directory to the Python path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import CONFIG, load_config
from training import run_training_once # This function is designed for HPO trials

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def objective(trial: optuna.Trial) -> float:
    """
    Objective function for Optuna, defining the search space and returning a metric.
    """
    # Load the base configuration
    cfg = load_config("config.yaml") # Assuming config.yaml exists for base settings

    # Define hyperparameter search space using Optuna trial suggestions
    # Training parameters
    cfg['training']['learning_rate'] = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    cfg['training']['weight_decay'] = trial.suggest_loguniform("wd", 1e-6, 1e-2)
    
    # Model parameters (example: dropout probability)
    # Ensure 'dropout_prob' exists in your CONFIG['model'] for this to work
    if 'dropout_prob' in cfg['model']['bongard_head_config']:
        cfg['model']['bongard_head_config']['dropout_prob'] = trial.suggest_uniform("drop", 0.0, 0.5)
    else:
        logger.warning("Dropout probability not found in model config. Skipping HPO for dropout.")

    # Add other hyperparameters to tune as needed
    # Example: batch size (if not fixed by DDP)
    # cfg['training']['batch_size'] = trial.suggest_categorical("batch_size", [16, 32, 64])

    # For HPO, run a shorter training duration
    hpo_epochs = cfg['training'].get('hpo_epochs', 5) # Define 'hpo_epochs' in your config for this

    logger.info(f"Trial {trial.number}: Starting with HPs: {trial.params}")
    
    # Run a single training session with the suggested hyperparameters
    # The `run_training_once` function should return a single metric (e.g., validation accuracy)
    try:
        val_acc = run_training_once(cfg, epochs=hpo_epochs)
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        # Propagate the exception or return a very low value to indicate failure
        raise optuna.exceptions.TrialPruned() # Prune trial if it fails
    
    logger.info(f"Trial {trial.number}: Validation Accuracy: {val_acc:.4f}")
    return val_acc

def run_hpo(n_trials: int = 30, study_name: str = "bongard_hpo", storage_url: Optional[str] = None):
    """
    Runs the Optuna hyperparameter optimization study.
    Args:
        n_trials (int): Number of optimization trials.
        study_name (str): Name of the Optuna study.
        storage_url (Optional[str]): URL for Optuna storage (e.g., "sqlite:///db.sqlite3").
                                     If None, an in-memory study is created.
    """
    logger.info(f"Starting Optuna HPO study: {study_name} for {n_trials} trials.")
    
    # Create an Optuna study
    # 'maximize' because we want to maximize validation accuracy
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True # Load existing study if it exists
    )
    
    # Optimize the objective function
    study.optimize(objective, n_trials=n_trials)

    logger.info("HPO study finished.")
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best value (validation accuracy): {study.best_value:.4f}")
    logger.info(f"Best hyperparameters: {study.best_params}")

    # You can also save the best hyperparameters to a file
    best_hps_path = "best_hps.json"
    with open(best_hps_path, 'w') as f:
        json.dump(study.best_params, f, indent=4)
    logger.info(f"Best hyperparameters saved to {best_hps_path}")

if __name__ == "__main__":
    # Example usage:
    # To run an in-memory study:
    run_hpo(n_trials=10)

    # To run a study with persistent storage (e.g., SQLite database):
    # Make sure to install `sqlalchemy` for database storage: pip install sqlalchemy
    # run_hpo(n_trials=30, storage_url="sqlite:///bongard_hpo.db")

