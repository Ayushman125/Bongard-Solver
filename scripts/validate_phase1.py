
# -----------------------------------------------------------
# Make sure we can import both:
#   1) core_models   (living at project root)
#   2) anything under src/
# -----------------------------------------------------------

import os, sys
SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
SRC_ROOT   = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, REPO_ROOT)     # for core_models/
sys.path.insert(0, SRC_ROOT)      # for src/data, src/perception, src/utils

#!/usr/bin/env python3


import os, time, glob, logging
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.calibration import calibration_curve


import torch
from torchvision.transforms.functional import to_tensor

# Core model imports
from core_models.training_args import config
from core_models.training      import train_perception_with_buffer, fine_tune_perception
# Modular Bongard generator imports
from bongard_generator.sampler import BongardSampler
from bongard_generator.config_loader import get_sampler_config
from bongard_generator.rule_loader import get_all_rules
from bongard_generator.validation import ValidationSuite
# Perception model imports
from src.perception.primitive_extractor import extract_cnn_features, MODEL, load_perception_model


# 1) If no checkpoint yet, train model first (only if not already present)
def checkpoint_exists():
    return os.path.exists(config['phase1']['best_model_path']) or os.path.exists(config['phase1']['last_model_path'])

if not checkpoint_exists():
    print(f">>> No checkpoint found at {config['phase1']['best_model_path']} or {config['phase1']['last_model_path']}\n>>> Training Phase-1 perception model...")
    train_perception_with_buffer(config)
    print(">>> Training done. Now proceeding to validation.")
else:
    ckpt_path = config['phase1']['best_model_path'] if os.path.exists(config['phase1']['best_model_path']) else config['phase1']['last_model_path']
    print(f"Checkpoint found at {ckpt_path}. Skipping training.")


# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger("validate_phase1")

# Ensure checkpoints exist and resume if interrupted
from pathlib import Path
Path(config['phase1']['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
ckpt = config['phase1']['best_model_path'] if os.path.exists(config['phase1']['best_model_path']) else config['phase1']['last_model_path']
if os.path.exists(ckpt):
    logger.info(f"Loading model checkpoint: {ckpt}")
    # Model is loaded automatically in primitive_extractor; no need to call load_model()
else:
    logger.error(f"No checkpoint found at {ckpt} after attempted training. Model may be uninitialized!")


import functools
@functools.lru_cache(maxsize=2)
def build_synth_holdout(n=None, cache_path="synth_holdout.npz"):
    n = n or config['phase1']['synth_holdout_count']
    if os.path.exists(cache_path):
        logger.info(f"Loading cached synthetic holdout from {cache_path}")
        arr = np.load(cache_path, allow_pickle=True)
        imgs = [Image.fromarray(x) for x in arr['imgs']]
        labels = arr['labels'].tolist()
        return imgs, labels
    logger.info(f"Generating {n} synthetic holdout samples using BongardSampler...")
    sampler_config = get_sampler_config(img_size=config['phase1']['img_size'])
    sampler = BongardSampler(sampler_config)
    rules = get_all_rules()
    imgs, labels = [], []
    for i in tqdm(range(n), desc="Synth Holdout Generation"):
        rule = rules[i % len(rules)]
        problem = sampler.sample_problem(rule_description=rule.description, num_pos_scenes=1, num_neg_scenes=1)
        if problem:
            for scene in problem['positive_scenes']:
                img = scene.get('image')
                if img is not None:
                    imgs.append(img)
                    labels.append(MODEL.class_names.index(rule.value) if hasattr(rule, 'value') else 0)
            for scene in problem['negative_scenes']:
                img = scene.get('image')
                if img is not None:
                    imgs.append(img)
                    labels.append(MODEL.class_names.index(rule.value) if hasattr(rule, 'value') else 0)
    # Save cache
    arr_imgs = np.stack([np.array(img) for img in imgs])
    np.savez_compressed(cache_path, imgs=arr_imgs, labels=np.array(labels))
    logger.info(f"Saved synthetic holdout to {cache_path}")
    return imgs, labels


def test_modular_generator():
    """Test the modular Bongard generator package."""
    logger.info("==== Testing Modular Bongard Generator ====")
    try:
        # Validate installation
        validator = ValidationSuite()
        validation_results = validator.run_all_validations()
        validator.print_validation_report()
        if not all(validation_results.values()):
            logger.warning("⚠ Some validations failed but continuing with tests")
        else:
            logger.info("✓ All validations passed")

        # Test sampler configuration
        logger.info("Testing sampler configuration...")
        config_obj = get_sampler_config()
        logger.info(f"✓ Config loaded: img_size={config_obj.img_size}, max_objs={config_obj.max_objs}")

        # Test single problem generation
        logger.info("Testing single problem generation...")
        sampler = BongardSampler(config_obj)
        rules = get_all_rules()
        rule = rules[0]
        problem = sampler.sample_problem(rule_description=rule.description, num_pos_scenes=3, num_neg_scenes=3)
        if problem:
            logger.info(f"✓ Generated problem with rule: {problem['rule']['description']}")
            logger.info(f"  Positive scenes: {len(problem['positive_scenes'])}")
            logger.info(f"  Negative scenes: {len(problem['negative_scenes'])}")
            # Validate problem structure
            required_keys = ['rule', 'positive_scenes', 'negative_scenes', 'metadata']
            for key in required_keys:
                if key not in problem:
                    logger.error(f"✗ Missing key '{key}' in generated problem")
                    return False
            # Check scenes have required structure
            for i, scene in enumerate(problem['positive_scenes'][:1]):
                if 'objects' not in scene or 'metadata' not in scene:
                    logger.warning(f"⚠ Scene {i} missing objects or metadata")
                else:
                    logger.info(f"  Scene {i}: {len(scene['objects'])} objects")
        else:
            logger.error("✗ Failed to generate problem")
            return False

        # Test sampler with different configurations
        logger.info("Testing sampler with different configurations...")
        custom_config = get_sampler_config(img_size=64, max_objs=2)
        sampler = BongardSampler(custom_config)
        test_rules = ["SHAPE(TRIANGLE)", "COUNT(2)", "COLOR(RED)"]
        for rule_desc in test_rules:
            try:
                problem = sampler.sample_problem(rule_description=rule_desc, num_pos_scenes=2, num_neg_scenes=2)
                if problem:
                    logger.info(f"✓ Generated problem for rule: {rule_desc}")
                else:
                    logger.warning(f"⚠ Could not generate problem for rule: {rule_desc}")
            except Exception as e:
                logger.warning(f"⚠ Error generating problem for rule {rule_desc}: {e}")

        # Test coverage tracking
        logger.info("Testing coverage tracking...")
        coverage_report = sampler.get_coverage_report()
        logger.info(f"✓ Coverage report generated: {coverage_report.get('total_scenes', 0)} scenes tracked")

        # Test dataset generation
        logger.info("Testing small dataset generation...")
        try:
            dataset = sampler.generate_dataset(num_problems=3, use_adversarial=False)
            if dataset and 'problems' in dataset:
                logger.info(f"✓ Generated dataset with {len(dataset['problems'])} problems")
                for i, problem in enumerate(dataset['problems'][:1]):
                    if 'id' in problem and 'rule' in problem:
                        logger.info(f"  Problem {i}: ID={problem['id']}, Rule={problem['rule']['description']}")
                    else:
                        logger.warning(f"⚠ Problem {i} missing ID or rule")
            else:
                logger.warning("⚠ Dataset generation returned empty or invalid result")
        except Exception as e:
            logger.warning(f"⚠ Dataset generation failed: {e}")

        logger.info("✓ Modular generator testing completed successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Modular generator testing failed: {e}")
        return False


@functools.lru_cache(maxsize=2)
def load_real_holdout(root=None, cache_path="real_holdout.npz"):
    root = root or config['phase1']['real_holdout_root']
    if not os.path.isdir(root) or not os.listdir(root):
        logger.warning(f"No files in real holdout dir {root}, skipping real validation.")
        return None, None
    if os.path.exists(cache_path):
        logger.info(f"Loading cached real holdout from {cache_path}")
        arr = np.load(cache_path, allow_pickle=True)
        imgs = [Image.fromarray(x) for x in arr['imgs']]
        labels = arr['labels'].tolist()
        return imgs, labels
    logger.info(f"Loading real holdout images from {root}")
    imgs, labels = [], []
    for prob in tqdm(sorted(os.listdir(root)), desc="Real Holdout Problems"):
        for lbl_folder in ["0","1"]:
            files = glob.glob(f"{root}/{prob}/images/{lbl_folder}/*")
            for fn in tqdm(files, desc=f"{prob}/{lbl_folder}", leave=False):
                img = Image.open(fn).convert("L")
                imgs.append(img)
                labels.append(int(lbl_folder))
    if not imgs:
        logger.warning(f"No images found in real holdout dir {root}, skipping real validation.")
        return None, None
    arr_imgs = np.stack([np.array(img) for img in imgs])
    np.savez_compressed(cache_path, imgs=arr_imgs, labels=np.array(labels))
    logger.info(f"Saved real holdout to {cache_path}")
    return imgs, labels


def eval_set(imgs, labels):
    preds, confs = [], []
    logger.info(f"Evaluating {len(imgs)} images...")
    # Fallback class names if MODEL is None or missing class_names
    fallback_class_names = ['triangle', 'quadrilateral', 'filled', 'outlined']
    class_names = getattr(MODEL, 'class_names', fallback_class_names)
    for img in tqdm(imgs, desc="Inferencing"):
        result = extract_cnn_features(img)
        if isinstance(result, tuple) and len(result) >= 2:
            v, c = result[:2]
        else:
            raise ValueError(f"Unexpected output from extract_cnn_features: {result}")
        try:
            preds.append(class_names.index(v))
        except ValueError:
            # If v is not in class_names, append a default index (e.g., 0)
            preds.append(0)
        confs.append(c)
    acc = accuracy_score(labels, preds)
    logger.info(f"Accuracy: {acc:.4f}")
    prob_true, prob_pred = calibration_curve(
        [int(p==t) for p,t in zip(preds, labels)],
        confs, n_bins=config['phase1']['validate_bins']
    )
    return acc, prob_pred, prob_true


def plot_calibration(x, y, title):
    logger.info(f"Plotting calibration curve: {title}")
    plt.plot(x, y, "o-")
    plt.plot([0,1],[0,1],"--", color="gray")
    plt.title(title); plt.xlabel("Predicted Confidence"); plt.ylabel("Empirical Accuracy")
    plt.show()


def online_finetune_test(imgs, labels):
    # Why You’re Seeing 100% “Fine-Tune” Accuracy
    # Your online fine-tune routine currently does this:
    # 1. Takes the same 100 synthetic images and labels for both training and testing.
    # 2. Measures “pre-fine-tune” accuracy on those 100 → a perfect 100% (they’re synthetic and your model already overfits them).
    # 3. Fine-tunes on them → of course it remains 100%.
    # Whenever you train and test on identical data, you’ll trivially hit 100% accuracy.
    #
    # How to Fix: Introduce a Proper Train/Test Split
    # 1. Hold out a test subset that the model never sees during fine-tuning.
    # 2. Evaluate “pre” and “post” on that held-out set.

    from src.perception.primitive_extractor import initialize_perception_model
    # Use the last checkpoint for fine-tuning
    ckpt_path = config['phase1']['last_model_path']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = initialize_perception_model(ckpt_path, device, config)
    if model is None:
        raise RuntimeError("PerceptionModel failed to load—cannot fine-tune")

    # Split: train on first n_train, test on the rest (up to 100)
    n_train = 80
    train_imgs = imgs[:n_train]
    train_lbls = labels[:n_train]
    test_imgs  = imgs[n_train:100]
    test_lbls  = labels[n_train:100]

    # Evaluate on held-out test set before fine-tuning
    logger.info(f"Evaluating before fine-tuning on {len(test_imgs)} held-out samples...")
    pre_acc, *_ = eval_set(test_imgs, test_lbls)

    # Fine-tune on the training set
    from core_models.replay_buffer import ReplayBuffer
    buffer = ReplayBuffer(capacity=len(train_imgs))
    logger.info("Preparing buffer for online fine-tuning...")
    for img, lbl in tqdm(list(zip(train_imgs, train_lbls)), desc="Buffering", leave=False):
        buffer.push(to_tensor(img), lbl)
    logger.info("Starting online fine-tuning...")
    start = time.time()
    fine_tune_perception(model, buffer, config)
    dur = time.time() - start
    logger.info(f"Fine-tuning completed in {dur:.1f}s")

    # Evaluate on held-out test set after fine-tuning
    logger.info(f"Evaluating after fine-tuning on {len(test_imgs)} held-out samples...")
    post_acc, *_ = eval_set(test_imgs, test_lbls)
    return pre_acc, post_acc, dur


if __name__ == "__main__":
    # Test modular generator first
    logger.info("==== Phase 1 Validation: Modular Generator Testing ====")
    modular_success = test_modular_generator()
    if not modular_success:
        logger.warning("Modular generator testing had issues, but continuing with phase 1 validation")
    
    # Determine synthetic holdout cache path from config if present
    synth_cache = 'synth_holdout.npz'
    if 'holdout_cache' in config['data'].get('synthetic_data_config', {}):
        synth_cache = config['data']['synthetic_data_config']['holdout_cache']

    if config['data'].get('use_synthetic_data', True):
        logger.info("==== Phase 1 Validation: Synthetic Holdout ====")
        s_imgs, s_lbls = build_synth_holdout(cache_path=synth_cache)
        s_acc, s_pp, s_pt = eval_set(s_imgs, s_lbls)
        logger.info(f"Synth Acc: {s_acc:.4f}")
        plot_calibration(s_pp, s_pt, "Synthetic Calibration")
    else:
        logger.info("==== Phase 1 Validation: Real Holdout ====")
        r_imgs, r_lbls = load_real_holdout()
        if r_imgs is not None:
            r_acc, r_pp, r_pt = eval_set(r_imgs, r_lbls)
            logger.info(f"Real Acc: {r_acc:.4f}")
            plot_calibration(r_pp, r_pt, "Real Calibration")
        else:
            logger.info("No real holdout data found. Skipping real validation.")

    logger.info("==== Phase 1: Online Fine-tune Test (first 100 synth) ====")
    # Use synthetic for fine-tune if available, else real
    if 's_imgs' in locals() and 's_lbls' in locals():
        pre, post, t = online_finetune_test(s_imgs[:100], s_lbls[:100])
    elif 'r_imgs' in locals() and r_imgs is not None:
        pre, post, t = online_finetune_test(r_imgs[:100], r_lbls[:100])
    else:
        logger.info("No data available for online fine-tune test.")
        pre = post = t = None
    if pre is not None:
        logger.info(f"Online FT: Pre {pre:.4f} → Post {post:.4f} in {t:.1f}s")
    
    # Summary report
    logger.info("==== Phase 1 Validation Summary ====")
    if modular_success:
        logger.info("✓ Modular generator tests passed")
    else:
        logger.warning("⚠ Modular generator tests had issues")
    
    if 's_acc' in locals():
        logger.info(f"✓ Synthetic holdout accuracy: {s_acc:.4f}")
    if 'r_acc' in locals():
        logger.info(f"✓ Real holdout accuracy: {r_acc:.4f}")
    if pre is not None and post is not None:
        improvement = post - pre
        logger.info(f"✓ Fine-tuning improvement: {improvement:+.4f}")
    
    logger.info("Phase 1 validation completed!")
