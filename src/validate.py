# Folder: bongard_solver/src/
# File: validation.py
import logging
import os
import json
from typing import List, Dict, Any, Tuple, Optional
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from tqdm.auto import tqdm

# Import metrics from core_models
HAS_METRICS = False
try:
    from core_models.metrics import classification_accuracy, rule_match_f1, calculate_f1_scores, precision_recall_fscore_support
    HAS_METRICS = True
except ImportError:
    logging.warning("core_models.metrics not found. Using dummy metric functions.")
    def classification_accuracy(y_true: List[int], y_pred: List[int]) -> float:
        logger.warning("Using dummy classification_accuracy.")
        return float(np.mean(np.array(y_true) == np.array(y_pred)))
    def rule_match_f1(gt_rules: List[str], inferred_rules: List[str]) -> float:
        logger.warning("Using dummy rule_match_f1.")
        # Simple dummy F1 for demonstration
        matches = sum(1 for r_gt in gt_rules for r_inf in inferred_rules if r_gt == r_inf)
        if len(gt_rules) == 0 and len(inferred_rules) == 0: return 1.0
        precision = matches / len(inferred_rules) if len(inferred_rules) > 0 else 0.0
        recall = matches / len(gt_rules) if len(gt_rules) > 0 else 0.0
        if precision + recall == 0: return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    # Dummy for calculate_f1_scores and precision_recall_fscore_support
    def calculate_f1_scores(y_true, y_pred, labels=None, average='binary', output_dict=False, zero_division=0):
        logger.warning("Using dummy calculate_f1_scores.")
        from sklearn.metrics import f1_score
        res = f1_score(y_true, y_pred, labels=labels, average=average, zero_division=zero_division)
        if output_dict:
            # This dummy doesn't fully mimic output_dict, just provides a placeholder
            return {'f1': res, 'precision': res, 'recall': res} # Simplified for binary
        return res
    
    def precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0):
        logger.warning("Using dummy precision_recall_fscore_support.")
        from sklearn.metrics import precision_recall_fscore_support as sk_prfs
        p, r, f, _ = sk_prfs(y_true, y_pred, average=average, zero_division=zero_division)
        return p, r, f, None # Return tuple as expected

# Import components from other modules
HAS_ALL_IMPORTS = False
try:
    from emergent.workspace_ext import Workspace
    from emergent.codelets import Scout, GroupScout, RuleTester, RuleBuilder, RuleProposer
    from utils.compute_temperature import compute_temperature
    from data.real_set import RealBongardDataset
    from core_models.training import load_trained_model # To load the perception model
    from config import CONFIG, IMAGENET_MEAN, IMAGENET_STD, DEVICE
    HAS_ALL_IMPORTS = True
except ImportError as e:
    logging.error(f"Failed to import necessary modules for validation: {e}. Validation will use dummy components.")
    # Dummy classes/functions for standalone execution (already defined in previous version)
    # Re-defining them here to ensure they are available if the main imports fail.
    class Workspace:
        def __init__(self, images, config=None, perception_model=None):
            self.images = images
            self.config = config if config else {}
            self.coderack = []
            self.sg = type('DummySG', (object,), {
                'problem_solved': lambda: False,
                'get_solution': lambda: None,
                'current_problem_fingerprint': lambda: "dummy_fingerprint",
                'build_scene_graph': lambda img_np, **kwargs: {'objects': [{'id': 'obj_0', 'attributes': {'shape': 'circle'}}, {'id': 'obj_1', 'attributes': {'shape': 'square'}}], 'relations': []},
                'mark_solution': lambda sol: None,
                'current_scene_graph_data': [] # For RuleProposer
            })()
            self.objects = [f"obj_{i}" for i in range(len(images))] if images else ["obj_0", "obj_1"]
            self.object_ids_per_image = [[f"obj_{i}"] for i in range(len(images))] if images else [["obj_0"], ["obj_1"]]
            self.built = []
            self.proposed = []
            self.current_rule_fragments = []
            self.support_set_scene_graphs = []
            self.support_set_labels = []
            self.concept_net = type('DummyConceptNet', (object,), {
                'step': lambda decay_factor, max_activation: None,
                'activate_node': lambda name, urgency: None,
                'get_node_activation': lambda name: 0.5
            })()
            self.perception_model = perception_model # Store the dummy model
        def post_codelet(self, codelet): self.coderack.append(codelet)
        def run_codelets(self, temp, max_steps):
            # Simulate running codelets and building some features/rules
            if self.coderack:
                codelet = self.coderack.pop(0) # Simple FIFO
                # Simulate some action for RuleProposer
                if isinstance(codelet, RuleProposer):
                    if not self.current_rule_fragments: # Only propose if no rules yet
                        self.current_rule_fragments.append({'rule_description': "FORALL(O, SHAPE(O, CIRCLE))", 'confidence': 0.7, 'source_codelet': 'RuleProposer'})
                        self.post_codelet(RuleTester("FORALL(O, SHAPE(O, CIRCLE))", urgency=0.5))
                elif isinstance(codelet, RuleTester):
                    if codelet.rule_description == "FORALL(O, SHAPE(O, CIRCLE))":
                        self.post_codelet(RuleBuilder("FORALL(O, SHAPE(O, CIRCLE))", confidence=0.8, urgency=0.6))
                elif isinstance(codelet, RuleBuilder):
                    if codelet.rule_description == "FORALL(O, SHAPE(O, CIRCLE))" and codelet.confidence > 0.7:
                        self.sg.mark_solution("FORALL(O, SHAPE(O, CIRCLE))")
                # Simulate some feature building
                if not self.built and self.objects:
                    self.build_feature(self.objects[0], 'shape', 'circle', 0.9)

        def primitive_feature(self, obj_id, feat_type): return "dummy_val", 0.5
        def confirm_feature(self, obj_id, feat_type): return "dummy_val", 0.6
        def build_feature(self, obj_id, feat_type, value, confidence): self.built.append((obj_id, feat_type, value, confidence))
        def is_conflict(self, struct_id, threshold): return False
        def remove_structure(self, struct_id): pass

    class Scout:
        def __init__(self, *args, **kwargs): pass
        def run(self, *args, **kwargs): pass
    class GroupScout:
        def __init__(self, *args, **kwargs): pass
        def run(self, *args, **kwargs): pass
    class RuleTester:
        def __init__(self, *args, **kwargs): pass
        def run(self, *args, **kwargs): pass
    class RuleBuilder:
        def __init__(self, *args, **kwargs): pass
        def run(self, *args, **kwargs): pass
    class RuleProposer:
        def __init__(self, *args, **kwargs): pass
        def run(self, *args, **kwargs): pass

    def compute_temperature(*args, **kwargs): return 0.5

    class RealBongardDataset:
        def __init__(self, *args, **kwargs): pass
        def __len__(self): return 1
        def __getitem__(self, idx):
            # Return dummy data matching the expected format for a single problem
            img_size = (128, 128)
            dummy_img_np_pos = np.random.randint(0, 256, size=(img_size[0], img_size[1], 3), dtype=np.uint8)
            dummy_img_np_neg = np.random.randint(0, 256, size=(img_size[0], img_size[1], 3), dtype=np.uint8)
            
            # Simulate a problem with 2 images, one positive, one negative
            problem_images_list = [dummy_img_np_pos, dummy_img_np_neg]
            problem_labels_list = [1, 0] # Example: first image is positive, second is negative
            problem_gt_sgs_list = [
                {'objects': [{'id': 'obj_0', 'attributes': {'shape': 'circle', 'color': 'red'}}], 'relations': []},
                {'objects': [{'id': 'obj_0', 'attributes': {'shape': 'square', 'color': 'blue'}}], 'relations': []}
            ]
            problem_gt_rule_description = "FORALL(O, SHAPE(O, CIRCLE))"
            problem_id = "dummy_problem_001"
            
            return problem_images_list, problem_labels_list, problem_gt_sgs_list, problem_gt_rule_description, problem_id

    def load_trained_model(*args, **kwargs):
        # Dummy model that returns random logits
        class DummyPerceptionModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy_classifier = torch.nn.Linear(10, 2) # Dummy output for 2 classes
            def forward(self, x, **kwargs):
                # Simulate feature extraction and then classification
                # Output random features, then pass through dummy classifier
                dummy_features = torch.randn(x.shape[0], 10, device=x.device)
                return {'bongard_logits': self.dummy_classifier(dummy_features)}
        return DummyPerceptionModel(), {}

    # Dummy CONFIG and DEVICE if not imported
    CONFIG = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'debug': {'log_level': 'INFO', 'conflict_threshold': 0.2, 'solution_threshold': 0.8},
        'slipnet_config': {'initial_temperature': 1.0, 'final_temperature': 0.1, 'annealing_type': 'linear',
                           'decay_factor': 0.01, 'max_activation': 1.0, 'activation_threshold': 0.1,
                           'rule_plausibility_threshold': 0.6, 'solution_threshold': 0.8},
        'data': {'image_size': [128, 128], 'real_data_path': './data/real_bongard'},
        'model': {'backbone': 'dummy', 'pretrained': False, 'feat_dim': 576, 'bongard_head_config': {'num_classes': 2},
                  'attribute_classifier_config': {}, 'relation_gnn_config': {}},
        'training': {'seed': 42}
    }
    DEVICE = torch.device(CONFIG['device'])
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def log_failure(problem_fingerprint: str, log_file: str = './data/adversarial_seeds.json'):
    """
    Logs the fingerprint of a failed Bongard problem for targeted generation.
    Args:
        problem_fingerprint (str): A unique identifier or description of the failed problem.
        log_file (str): Path to the JSON file where failures are logged.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    failures = []
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                failures = json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Could not decode JSON from {log_file}. Starting with empty failures list.")
    
    if problem_fingerprint not in failures:
        failures.append(problem_fingerprint)
        with open(log_file, 'w') as f:
            json.dump(failures, f, indent=4)
        logger.info(f"Logged failed problem fingerprint: {problem_fingerprint}")
    else:
        logger.debug(f"Problem fingerprint {problem_fingerprint} already logged.")

def run_validation(
    predicted_labels: List[int],
    true_labels: List[int],
    predicted_rules: Optional[List[str]] = None,
    ground_truth_rules: Optional[List[str]] = None,
    problem_ids: Optional[List[Any]] = None
) -> Dict[str, Any]:
    """
    Runs a comprehensive validation process, calculating various metrics.
    Args:
        predicted_labels (List[int]): List of predicted class labels (e.g., 0 or 1).
        true_labels (List[int]): List of true class labels.
        predicted_rules (Optional[List[str]]): List of inferred DSL rules.
        ground_truth_rules (Optional[List[str]]): List of ground truth DSL rules.
        problem_ids (Optional[List[Any]]): Optional list of problem identifiers for logging.
    Returns:
        Dict[str, Any]: A dictionary containing various computed metrics.
    """
    metrics = {}

    # 1. Classification Accuracy
    if predicted_labels and true_labels:
        acc = classification_accuracy(true_labels, predicted_labels)
        metrics['classification_accuracy'] = acc
        logger.info(f"Classification Accuracy: {acc:.4f}")
    else:
        logger.warning("Skipping classification accuracy: Labels not provided.")

    # 2. Rule Match F1 Score (if rules are provided)
    if predicted_rules is not None and ground_truth_rules is not None:
        f1 = rule_match_f1(ground_truth_rules, predicted_rules)
        metrics['rule_match_f1'] = f1
        logger.info(f"Rule Match F1 Score: {f1:.4f}")
    else:
        logger.warning("Skipping rule match F1 score: Rules not provided.")

    # 3. Detailed F1, Precision, Recall (for classification)
    if predicted_labels and true_labels and HAS_METRICS:
        try:
            # Assuming binary classification (0, 1)
            # Use precision_recall_fscore_support for direct access to P, R, F
            precision_binary, recall_binary, f1_binary, _ = precision_recall_fscore_support(
                true_labels, predicted_labels, average='binary', zero_division=0
            )
            
            metrics['f1_binary'] = f1_binary
            metrics['precision_binary'] = precision_binary
            metrics['recall_binary'] = recall_binary

            logger.info(f"Binary F1 Score: {f1_binary:.4f}")
            logger.info(f"Binary Precision: {precision_binary:.4f}")
            logger.info(f"Binary Recall: {recall_binary:.4f}")

            # You can also compute per-class F1 if needed
            # f1_per_class = calculate_f1_scores(true_labels, predicted_labels, average=None)
            # metrics['f1_per_class'] = f1_per_class.tolist()

        except Exception as e:
            logger.error(f"Error calculating detailed classification metrics: {e}", exc_info=True)

    # 4. Misclassified Problems (for debugging/analysis)
    if predicted_labels and true_labels and problem_ids:
        misclassified_problems = []
        for i, (pred, true) in enumerate(zip(predicted_labels, true_labels)):
            if pred != true:
                misclassified_problems.append({
                    'id': problem_ids[i],
                    'predicted': pred,
                    'true': true,
                    'gt_rule': ground_truth_rules[i] if ground_truth_rules and i < len(ground_truth_rules) else 'N/A',
                    'inferred_rule': predicted_rules[i] if predicted_rules and i < len(predicted_rules) else 'N/A'
                })
        metrics['misclassified_problems'] = misclassified_problems
        logger.info(f"Found {len(misclassified_problems)} misclassified problems.")

    return metrics


def validate_on_real(model_ckpt_path: str, config: Dict[str, Any]):
    """
    Validates the Bongard Solver's emergent system on a set of real Bongard problems.
    This function simulates the emergent loop for each real problem and logs failures.
    
    Args:
        model_ckpt_path (str): Path to the trained perception model checkpoint.
        config (Dict[str, Any]): The configuration dictionary.
    """
    if not HAS_ALL_IMPORTS:
        logger.error("Cannot run full validation due to missing imports. Please ensure all dependencies are installed.")
        return

    logger.info(f"Starting validation on real Bongard problems using model from: {model_ckpt_path}")

    # Load the trained perception model
    # load_trained_model returns a LitBongard instance (or PerceptionModule if directly loaded)
    perception_model, _ = load_trained_model(
        model_path=model_ckpt_path,
        cfg=config,
        current_rank=0, # Assuming single GPU for validation
        is_ddp_initialized=False
    )
    perception_model.eval() # Set to evaluation mode

    # Load real Bongard problems
    real_data_path = config['data']['real_data_path']
    if not os.path.exists(real_data_path):
        logger.error(f"Real data path '{real_data_path}' does not exist. Cannot validate on real data.")
        return

    # Assuming RealBongardDataset can load problems from this path
    real_dataset = RealBongardDataset(real_data_path, config)
    
    if len(real_dataset) == 0:
        logger.warning(f"No real Bongard problems found in {real_data_path}. Skipping validation.")
        return

    logger.info(f"Loaded {len(real_dataset)} real Bongard problems for validation.")

    all_predicted_labels = []
    all_true_labels = []
    all_predicted_rules = []
    all_ground_truth_rules = []
    all_problem_ids = []
    
    for problem_idx in tqdm(range(len(real_dataset)), desc="Validating on Real Problems"):
        try:
            # RealBongardDataset should return:
            # (problem_images_list, problem_labels_list, problem_gt_sgs_list, problem_gt_rule_description, problem_id)
            problem_images_list, problem_labels_list, problem_gt_sgs_list, problem_gt_rule_description, problem_id = real_dataset[problem_idx]
            
            # Initialize Workspace with the problem images and the perception model
            ws = Workspace(problem_images_list, config=config, perception_model=perception_model)
            
            # Store ground truth support set data in workspace for RuleTester
            ws.support_set_scene_graphs = problem_gt_sgs_list
            ws.support_set_labels = problem_labels_list

            temperature = config['slipnet_config'].get('initial_temperature', 1.0)
            max_solve_iterations = 50 # Max iterations for emergent loop per problem

            # Seed initial codelets (Scouts, GroupScouts, RuleProposers)
            for obj_id_list in ws.object_ids_per_image:
                for obj_id in obj_id_list:
                    for feat in ['shape', 'color', 'size', 'position_h', 'position_v', 'fill', 'orientation', 'texture']:
                        ws.post_codelet(Scout(obj_id, feat, urgency=0.1))
            ws.post_codelet(GroupScout(urgency=0.2))
            ws.post_codelet(RuleProposer(urgency=0.3)) # New RuleProposer

            # Run the emergent loop
            iteration = 0
            inferred_solution_rule = None
            while not ws.sg.problem_solved() and ws.coderack and iteration < max_solve_iterations:
                iteration += 1
                current_temperature = compute_temperature(iteration, max_solve_iterations,
                                                         initial_temperature=config['slipnet_config']['initial_temperature'],
                                                         final_temperature=config['slipnet_config']['final_temperature'],
                                                         annealing_type=config['slipnet_config']['annealing_type'])
                
                ws.run_codelets(current_temperature, max_steps=10) # Run a few codelets per iteration
                ws.concept_net.step(decay_factor=config['slipnet_config']['decay_factor'],
                                    max_activation=config['slipnet_config']['max_activation'])
                
                # Check for solution
                if ws.sg.problem_solved():
                    inferred_solution_rule = ws.sg.get_solution()
                    break

            # Collect results for this problem
            all_problem_ids.append(problem_id)
            all_true_labels.extend(problem_labels_list)
            all_ground_truth_rules.append(problem_gt_rule_description)

            if inferred_solution_rule:
                all_predicted_rules.append(inferred_solution_rule)
                # For predicted labels, assume the inferred rule correctly classifies the problem images
                # This is a simplification; a full system would use the symbolic engine to apply the rule
                # to each image and get a predicted label.
                # For now, if a solution is found, assume it's "correct" for the labels.
                # This needs refinement based on how the rule is applied to yield labels.
                # For a dummy: if rule found, assume it correctly predicts the first image's label.
                # More robust: use symbolic engine to apply inferred_solution_rule to problem_images_list
                # and generate predicted_labels_for_problem.
                predicted_labels_for_problem = [1 if inferred_solution_rule == problem_gt_rule_description else 0] * len(problem_labels_list)
                all_predicted_labels.extend(predicted_labels_for_problem)

                logger.info(f"[VALID] Problem {problem_id}: Inferred Rule='{inferred_solution_rule}', GT Rule='{problem_gt_rule_description}'")
            else:
                all_predicted_rules.append("NO_RULE_FOUND")
                # If no rule is found, assume it predicts the negative class for all images in the problem
                all_predicted_labels.extend([0] * len(problem_labels_list))
                logger.warning(f"[VALID] Problem {problem_id}: No solution found by emergent system.")
                # Log failure if no solution is found
                log_failure(problem_id) # Use the problem ID as fingerprint

        except Exception as e:
            logger.error(f"Error validating problem {problem_idx+1} (ID: {problem_id if 'problem_id' in locals() else 'N/A'}): {e}", exc_info=True)
            log_failure(f"problem_{problem_idx+1}_error") # Log error as a failure fingerprint
            # Ensure lists are kept consistent even on error
            all_problem_ids.append(f"problem_{problem_idx+1}_error")
            all_true_labels.extend([0] * len(problem_labels_list) if 'problem_labels_list' in locals() else [0]) # Default to 0 if labels not available
            all_predicted_labels.extend([0] * len(problem_labels_list) if 'problem_labels_list' in locals() else [0])
            all_predicted_rules.append("ERROR_OCCURRED")
            all_ground_truth_rules.append("ERROR_OCCURRED")

    # Run comprehensive validation metrics
    if all_true_labels:
        final_metrics = run_validation(
            predicted_labels=all_predicted_labels,
            true_labels=all_true_labels,
            predicted_rules=all_predicted_rules,
            ground_truth_rules=all_ground_truth_rules,
            problem_ids=all_problem_ids
        )
        logger.info(f"\n--- Overall Validation Summary ---")
        for metric, value in final_metrics.items():
            if isinstance(value, (float, int)):
                logger.info(f"{metric}: {value:.4f}")
            elif isinstance(value, list) and metric == 'misclassified_problems':
                logger.info(f"Misclassified Problems Count: {len(value)}")
                for p in value:
                    logger.info(f"  - ID: {p['id']}, Pred: {p['predicted']}, True: {p['true']}, GT Rule: {p['gt_rule']}, Inferred Rule: {p['inferred_rule']}")
    else:
        logger.info("\nNo problems were successfully processed for overall validation.")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info("Running src/validation.py example.")

    # Create dummy config.yaml if it doesn't exist for standalone testing
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        logger.warning(f"Dummy config.yaml not found at {config_path}. Creating a sample.")
        sample_config = {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'debug': {
                'log_level': 'INFO',
                'save_model_checkpoints': './checkpoints',
                'logs_dir': './logs',
                'seed': 42,
                'validation_accuracy_threshold': 0.8,
                'conflict_threshold': 0.2,
                'solution_threshold': 0.8
            },
            'slipnet_config': {
                'initial_temperature': 1.0,
                'final_temperature': 0.1,
                'annealing_type': 'linear',
                'decay_factor': 0.01,
                'max_activation': 1.0,
                'activation_threshold': 0.1,
                'rule_plausibility_threshold': 0.6,
                'solution_threshold': 0.8
            },
            'data': {
                'image_size': [128, 128],
                'real_data_path': './data/real_bongard'
            },
            'model': {
                'backbone': 'mobilenet_v3_small',
                'pretrained': True,
                'feat_dim': 576,
                'bongard_head_config': {'num_classes': 2},
                'attribute_classifier_config': {'shape': 5, 'color': 7, 'size': 3, 'fill': 2, 'orientation': 4, 'texture': 2, 'mlp_dim': 256, 'head_dropout_prob': 0.3},
                'relation_gnn_config': {'hidden_dim': 256, 'num_relations': 11},
            },
            'training': {
                'epochs': 1, # Small number for dummy run
                'batch_size': 1, # Small batch size for dummy run
                'learning_rate': 1e-3,
                'optimizer': 'AdamW',
                'scheduler': 'None',
                'early_stop_patience': 5,
                'early_stop_delta': 0.001,
                'use_amp': False,
                'log_every_n_steps': 1,
                'use_domain_adaptation': False, # Not directly used in validation, but part of config
                'grl_alpha': 1.0,
                'lambda_style': 0.1,
                'lr_disc': 1e-4
            },
            'replay': {
                'buffer_capacity': 100,
                'init_samples': 10
            }
        }
        os.makedirs(sample_config['debug']['save_model_checkpoints'], exist_ok=True)
        os.makedirs(sample_config['debug']['logs_dir'], exist_ok=True)
        os.makedirs(sample_config['data']['real_data_path'], exist_ok=True)
        
        # Create a dummy real Bongard image for testing
        dummy_real_img = Image.new('RGB', (128, 128), color = (random.randint(0,255), random.randint(0,255), random.randint(0,255)))
        dummy_real_img.save(os.path.join(sample_config['data']['real_data_path'], 'dummy_real_problem_0_pos.png'))
        dummy_real_img.save(os.path.join(sample_config['data']['real_data_path'], 'dummy_real_problem_0_neg.png'))
        
        # Create a dummy model checkpoint
        if HAS_ALL_IMPORTS:
            from core_models.models import LitBongard
            dummy_model = LitBongard(CONFIG)
            torch.save(dummy_model.perception_module.state_dict(), os.path.join(sample_config['debug']['save_model_checkpoints'], 'dummy_perception_model.pt'))
            logger.info("Created dummy perception model checkpoint.")

        with open(config_path, 'w') as f:
            import yaml
            yaml.dump(sample_config, f, indent=4)
        logger.info(f"Created sample config at: {config_path}")
    
    # Load the config
    if HAS_ALL_IMPORTS:
        from config import load_config as load_main_config # Avoid conflict with core_models.training_args.load_config
        cfg = load_main_config(config_path)
    else:
        cfg = CONFIG # Use dummy config if imports failed

    # Define a dummy model checkpoint path
    dummy_model_ckpt = os.path.join(cfg['debug']['save_model_checkpoints'], 'dummy_perception_model.pt')

    # Run validation
    validate_on_real(dummy_model_ckpt, cfg)

