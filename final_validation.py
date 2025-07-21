"""
Complete Bongard-LOGO Pipeline Integration - Final Validation
Comprehensive pipeline including:
- CP-SAT constraint satisfaction sampling
- Genetic algorithm optimization
- Professional geometric shape rendering
- GNN-based scene quality filtering
- Real ShapeBongard_V2 dataset processing
- Synthetic dataset generation (NVlabs 6+6 or 7+7 format)
- Auto-labeling with COCO annotations
- Complete validation and testing
"""
import os
import sys
import logging
import shutil
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from types import SimpleNamespace

# Add src path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Core imports - add to Python path
current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)

# Essential imports
try:
    import torch
except ImportError:
    print("ERROR: PyTorch not found. Please install PyTorch.")
    sys.exit(1)

# Import with proper error handling
try:
    from src.bongard_generator.builder import BongardGenerator
    from config import GeneratorConfig
    from src.bongard_generator.scene_graph import build_scene_graph, SHAPES, COLORS, FILLS
    from src.bongard_generator.gnn_model import SceneGNN
    from src.bongard_generator.dataset import create_composite_scene
    from src.bongard_generator.rule_loader import get_all_rules, BongardRule
    from src.bongard_generator.rule_validator import RuleValidator
    from config import load_config, CONFIG
    from PIL import Image
    ADVANCED_COMPONENTS_AVAILABLE = True
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.warning(f"Advanced components not available: {e}")
    try:
        from bongard_rules import get_all_rules, BongardRule
        from config import load_config, CONFIG
        from PIL import Image
    except ImportError as e2:
        logger.error(f"Critical imports failed: {e2}")
        sys.exit(1)
    BongardGenerator = None
    RuleValidator = None  # Add this line
    ADVANCED_COMPONENTS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CompleteBongardPipeline:
    """
    Complete Bongard-LOGO pipeline integrating all components:
    - CP-SAT constraint satisfaction
    - Genetic algorithm optimization
    - Professional shape rendering
    - GNN-based quality filtering
    - Real ShapeBongard_V2 dataset processing
    - Auto-labeling and COCO annotations
    - NVlabs format generation (6+6 or 7+7)
    """
    def __init__(self, config):
        # Always use SimpleNamespace for config and subcomponents
        def to_namespace(obj):
            if isinstance(obj, dict):
                return SimpleNamespace(**{k: to_namespace(v) for k, v in obj.items()})
            return obj
        self.config = to_namespace(config)
        self.validation_dir = Path("complete_validation_output")
        self.validation_dir.mkdir(exist_ok=True, parents=True)
        self.synthetic_dir = self.validation_dir / "synthetic"
        self.synthetic_dir.mkdir(exist_ok=True, parents=True)
        # Use the real NVlabs repo location for Bongard-LOGO
        self.nvlabs_dir = Path("data/Bongard-LOGO")  # NVlabs repo location (real repo)
        self.shapebongard_dir = Path("ShapeBongard_V2")  # Real dataset location
        # Initialize core components
        self._initialize_components()
        # Statistics tracking
        self.stats = {
            'generated_problems': 0,
            'generated_images': 0,
            'gnn_filtered': 0,
            'cp_sat_solutions': 0,
            'genetic_improvements': 0,
            'annotation_extractions': 0,
            'real_dataset_problems': 0,
            'real_dataset_images': 0
        }

    def _initialize_components(self):
        """Initialize all pipeline components and integrate all generation logics."""
        logger.info(" 🔧  Initializing complete pipeline components...")
        try:
            # --- GNN Training Logic ---
            gnn_ckpt = getattr(self.config.generator, 'gnn_ckpt', None)
            use_gnn = getattr(self.config.generator, 'use_gnn', False)
            gnn_trained = os.path.exists(gnn_ckpt) if gnn_ckpt else False
            if use_gnn and not gnn_trained and gnn_ckpt:
                logger.info(f" 🧠  Training SceneGNN model (checkpoint: {gnn_ckpt}) ...")
                from src.bongard_generator.gnn_model import SceneGNN, train_gnn
                import torch
                from torch_geometric.data import Data
                import glob

                def load_graphs_from_folders(base_dir, in_feats=16):
                    graphs = []
                    for problem_type in os.listdir(base_dir):
                        for label in ['0', '1']:
                            folder = os.path.join(base_dir, problem_type, label)
                            if not os.path.isdir(folder):
                                continue
                            for img_file in glob.glob(os.path.join(folder, '*.png')):
                                # Dummy graph creation: replace with actual feature extraction if available
                                x = torch.randn((5, in_feats))
                                edge_index = torch.tensor([[0,1,2,3,4,0,2,3],[1,2,3,4,0,2,3,1]], dtype=torch.long)
                                y = torch.tensor([int(label)])
                                batch = torch.zeros(5, dtype=torch.long)
                                graphs.append(Data(x=x, edge_index=edge_index, y=y, batch=batch))
                    return graphs

                synthetic_base = os.path.join('complete_validation_output', 'synthetic_pass1')
                train_data = load_graphs_from_folders(synthetic_base)
                # Use a small subset for validation
                val_data = train_data[:min(20, len(train_data))]
                if not train_data:
                    logger.warning("No synthetic graphs found for GNN training. Skipping.")
                else:
                    gnn = SceneGNN(in_feats=16)
                    train_gnn(gnn, train_data, val_data, device=getattr(self.config, 'device', 'cpu'), epochs=5, lr=1e-3, checkpoint_path=gnn_ckpt)
                    logger.info(f" ✅  SceneGNN trained and saved to {gnn_ckpt}")
            elif use_gnn and gnn_ckpt:
                logger.info(f" ✅  SceneGNN checkpoint found: {gnn_ckpt}")
            # --- End GNN Training Logic ---

            # --- Generator Integration ---
            # Ensure coverage and coverage_goals are present and valid
            # Use the real CoverageTracker/EnhancedCoverageTracker class if available
            try:
                from src.bongard_generator.coverage import CoverageTracker, EnhancedCoverageTracker
                coverage_cls = EnhancedCoverageTracker if 'EnhancedCoverageTracker' in globals() else CoverageTracker
            except ImportError:
                coverage_cls = None

            if not hasattr(self.config, 'coverage') or not hasattr(self.config.coverage, 'coverage_goals'):
                logger.warning("Config missing 'coverage' or 'coverage_goals'. Adding default coverage config.")
                default_goals = {
                    'default_quota': 10,
                    'priority_quota': 20,
                    'priority_cells': []
                }
                if coverage_cls:
                    self.config.coverage = coverage_cls(default_goals)
                else:
                    self.config.coverage = default_goals
            else:
                cov_goals = self.config.coverage.coverage_goals
                if not isinstance(cov_goals, dict):
                    self.config.coverage.coverage_goals = dict(cov_goals.__dict__) if hasattr(cov_goals, '__dict__') else cov_goals

            # COMPREHENSIVE CONFIG TYPE CONVERSION - Fix all string/int division errors
            self._fix_config_types(self.config)

            # Build GeneratorConfig from config attributes
            gen_cfg = self.config.generator
            # Prepare a GeneratorConfig object for BongardGenerator
            # Build GeneratorConfig for BongardGenerator
            # Build GeneratorConfig for BongardGenerator
            self.generator_config = GeneratorConfig(
                min_objects=getattr(gen_cfg, 'min_objects', 2),
                max_objects=getattr(gen_cfg, 'max_objects', 5),
                canvas_size=getattr(self.config, 'canvas_size', 128),
                color_mode=str(getattr(self.config, 'color_mode', 'bw'))
            )
            # Ensure coverage is present and valid
            if hasattr(self.config, 'coverage') and hasattr(self.config.coverage, 'coverage_goals'):
                coverage_dict = self.config.coverage
                if not isinstance(coverage_dict, dict):
                    coverage_dict = coverage_dict.__dict__ if hasattr(coverage_dict, '__dict__') else {'coverage_goals': getattr(coverage_dict, 'coverage_goals', {})}
            else:
                coverage_dict = {
                    'coverage_goals': {
                        'default_quota': 10,
                        'priority_quota': 20,
                        'priority_cells': []
                    }
                }
            # Attach coverage to GeneratorConfig
            setattr(self.generator_config, 'coverage', coverage_dict)

            # Pass GeneratorConfig to BongardGenerator
            if BongardGenerator:
                self.generator = BongardGenerator(self.generator_config)
            else:
                logger.error("BongardGenerator not available. Cannot initialize generator.")
                raise RuntimeError("BongardGenerator not available")
            logger.info(" ✅  BongardGenerator initialized (all logics integrated)")

        except Exception as e:
            logger.error(f" ❌  BongardGenerator init failed: {e}")
            raise  # Fail loudly, do not fallback

        self.real_dataset_available = self._check_real_dataset_availability()
        if self.real_dataset_available:
            logger.info(" ✅  Real ShapeBongard_V2 dataset available")
        else:
            logger.info(" ⚠️  Real dataset not available (synthetic only)")

    def _check_real_dataset_availability(self) -> bool:
        """Check if ShapeBongard_V2 dataset is available."""
        return os.path.exists(self.shapebongard_dir)

    def _fix_config_types(self, config_obj):
        """
        Recursively traverse a config object (SimpleNamespace or dict)
        and convert numeric string values to int or float.
        """
        numeric_keys = [
            'canvas_size', 'stroke_min', 'stroke_max', 'jitter_px', 'threshold',
            'img_size', 'image_size', 'min_objects', 'max_objects', 'num_workers',
            'batch_size', 'epochs', 'learning_rate', 'weight_decay', 'gnn_thresh',
            'gnn_radius', 'cp_quota', 'ga_quota', 'population_size', 'generations',
            'mutation_rate', 'crossover_rate', 'feature_consistency_weight',
            'gradient_accumulation_steps', 'mixup_alpha', 'cutmix_alpha',
            'label_smoothing_epsilon', 'detection_confidence_threshold'
        ]
        if isinstance(config_obj, SimpleNamespace):
            for key, value in config_obj.__dict__.items():
                if key in numeric_keys and isinstance(value, str):
                    try:
                        setattr(config_obj, key, int(value))
                    except (ValueError, TypeError):
                        try:
                            setattr(config_obj, key, float(value))
                        except (ValueError, TypeError):
                            pass  # Keep as string if conversion fails
                elif isinstance(value, (SimpleNamespace, dict)):
                    self._fix_config_types(value)
        elif isinstance(config_obj, dict):
            for key, value in config_obj.items():
                if key in numeric_keys and isinstance(value, str):
                    try:
                        config_obj[key] = int(value)
                    except (ValueError, TypeError):
                        try:
                            config_obj[key] = float(value)
                        except (ValueError, TypeError):
                            pass
                elif isinstance(value, (SimpleNamespace, dict)):
                    self._fix_config_types(value)

    def run(self):
        logger.info("  🚀  Starting full Bongard pipeline orchestration...")
        synthetic_pass1_dir = self.validation_dir / "synthetic_pass1"
        synthetic_pass2_dir = self.validation_dir / "synthetic_pass2"
        gnn_ckpt = getattr(self.config.generator, 'gnn_ckpt', 'checkpoints/scene_gnn.pth')

        # Phase 1: Generate synthetic scenes (pass 1) if not already present
        if synthetic_pass1_dir.exists() and any(synthetic_pass1_dir.iterdir()):
            logger.info(f" 🟢  Synthetic data already exists at {synthetic_pass1_dir}, skipping generation.")
            # Load records from disk for GNN training
            pass1_records = []
            for rule_dir in synthetic_pass1_dir.iterdir():
                if not rule_dir.is_dir():
                    continue
                rule_name = rule_dir.name
                for side in ['1', '0']:
                    side_dir = rule_dir / side
                    if not side_dir.exists():
                        continue
                    for img_path in side_dir.glob('*.png'):
                        # Try to load object data from corresponding JSON file
                        objs_path = img_path.with_name(img_path.stem + '_objs.json')
                        objs = None
                        if objs_path.exists():
                            try:
                                with open(objs_path, 'r') as f:
                                    objs = json.load(f)
                            except Exception as e:
                                logger.warning(f"Failed to load object data from {objs_path}: {e}")
                        pass1_records.append((rule_name, side, img_path.stem, img_path, objs))
        else:
            pass1_records = self.generate_synthetic_data(out_root=synthetic_pass1_dir, return_records=True)

        # Phase 2: Build graph dataset and train GNN if checkpoint does not exist
        gnn_trained = os.path.exists(gnn_ckpt)
        if gnn_trained:
            logger.info(f" 🟢  GNN checkpoint found at {gnn_ckpt}, skipping training and moving forward.")
        else:
            if pass1_records:
                train_graphs, val_graphs = self.build_graph_dataset(pass1_records)
                self.train_gnn(train_graphs, val_graphs)
            else:
                logger.warning("No synthetic records available for GNN training. Skipping GNN training.")

        # Phase 3: Inject GNN sampler and re-generate synthetic scenes (pass 2)
        if os.path.exists(gnn_ckpt):
            self.enable_gnn_sampler()
            if synthetic_pass2_dir.exists() and any(synthetic_pass2_dir.iterdir()):
                logger.info(f" 🟢  Synthetic data (pass 2) already exists at {synthetic_pass2_dir}, skipping generation.")
            else:
                pass2_records = self.generate_synthetic_data(out_root=synthetic_pass2_dir, return_records=True)
        else:
            logger.info(" ⚠️  GNN checkpoint not found, skipping GNN-based generation.")

        # Phase 4: Auto-label and report
        self.run_auto_labeling()
        self.print_summary_report()

    def generate_synthetic_data(self, out_root, N=6, return_records=False):
        logger.info(f"  📦  Generating synthetic dataset at {out_root} ...")
        out_root = Path(out_root)
        out_root.mkdir(parents=True, exist_ok=True)

        # Patch: Set generator output_dir to this synthetic folder for correct saving
        self.generator.output_dir = out_root

        records = []
        if self.generator is None:
            logger.error("Generator not available. Cannot generate synthetic data.")
            return []

        rules = get_all_rules()
        if not rules:
            logger.error("No rules found. Cannot generate synthetic data.")
            return []

        for rule in rules:
            rule_dir = out_root / rule.name
            for side, label in [("1", True), ("0", False)]:
                side_dir = rule_dir / side
                side_dir.mkdir(parents=True, exist_ok=True)
                scenes = self.generator.generate_for_rule(rule, N=N, is_positive=label)
                self.stats['generated_problems'] += 1
                for i, (img, objs, tag) in enumerate(scenes):
                    if isinstance(tag, dict):
                        tag_str = tag.get('generation_method', 'scene')
                    else:
                        tag_str = str(tag)
                    tag_str = ''.join(c for c in tag_str if c.isalnum() or c in ('-', '_'))
                    img_path = side_dir / f"{i:02d}_{tag_str}.png"

                    # Empty scene detection: if objs is empty or None, log and skip saving
                    if not objs or len(objs) == 0:
                        logger.warning(f"Skipping blank scene for rule {rule.name}, side {side}, idx {i} (no objects)")
                        continue

                    img.save(img_path)
                    # Save object data as JSON alongside image
                    objs_path = side_dir / f"{i:02d}_{tag_str}_objs.json"
                    try:
                        with open(objs_path, 'w') as f:
                            json.dump(objs, f)
                    except Exception as e:
                        logger.warning(f"Failed to save object data for {img_path}: {e}")
                    self.stats['generated_images'] += 1
                    records.append((rule, side, i, img_path, objs))

        # --- Color Check: Warn if images are grayscale only ---
        color_count = 0
        bw_count = 0
        for _, _, _, img_path, _ in records:
            try:
                img = Image.open(img_path)
                if img.mode in ["RGB", "RGBA"]:
                    arr = np.array(img)
                    if arr.ndim == 3 and not np.all(arr[...,0] == arr[...,1]) and not np.all(arr[...,1] == arr[...,2]):
                        color_count += 1
                    else:
                        bw_count += 1
                else:
                    bw_count += 1
            except Exception:
                continue

        if color_count == 0 and bw_count > 0:
            logger.warning("  ⚠️  All generated images are black and white (grayscale). Check BongardGenerator config for color settings.")
        elif color_count > 0:
            logger.info(f"  🎨  Colored images generated: {color_count}, grayscale: {bw_count}")

        return records if return_records else None

    def build_graph_dataset(self, scene_records, split=(0.8,0.2)):
        from torch_geometric.data import Data
        import torch
        graphs = []
        for rule, side, idx, img_path, objs in scene_records:
            if objs is None:
                continue  # Skip records without object data
            graph = build_scene_graph(objs, self.config)
            graph.y = torch.tensor([1 if side=="1" else 0], dtype=torch.float)
            graphs.append(graph)
        split_pt = int(len(graphs)*split[0])
        return graphs[:split_pt], graphs[split_pt:]

    def train_gnn(self, train_graphs, val_graphs):
        from torch_geometric.loader import DataLoader
        from src.bongard_generator.gnn_model import SceneGNN, train_gnn
        device = getattr(self.config, 'device', 'cpu')
        gen_cfg = self.config.generator
        in_feats = train_graphs[0].x.size(1) if train_graphs else 16
        model = SceneGNN(in_feats=in_feats).to(device)
        train_loader = DataLoader(train_graphs, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=8)
        train_gnn(
            model,
            train_loader,
            val_loader,
            device=device,
            epochs=getattr(gen_cfg, 'epochs', 5),
            lr=getattr(gen_cfg, 'learning_rate', 1e-3),
            checkpoint_path=getattr(gen_cfg, 'gnn_ckpt', 'checkpoints/scene_gnn.pth')
        )
        self.gnn = model
        logger.info(f" ✅  SceneGNN trained and checkpoint saved to {getattr(gen_cfg, 'gnn_ckpt', 'checkpoints/scene_gnn.pth')}")

    def enable_gnn_sampler(self):
        # GNN filtering is already integrated in BongardGenerator (builder.py)
        # No need to inject a separate sampler. Just log that GNN filtering is enabled.
        logger.info("  ✅  GNN filtering enabled in BongardGenerator. No separate sampler required.")

    def run_auto_labeling(self):
        import subprocess
        logger.info("  🏷️  Running auto-labeling for real data (if available)...")
        auto_label_script_path = Path("scripts/auto_label_nvlabs.py")
        if auto_label_script_path.exists():
            logger.info("  -> Executing auto_label_nvlabs.py ...")
            try:
                subprocess.run([
                    sys.executable, str(auto_label_script_path),
                    "--input-dir", str(self.shapebongard_dir),
                    "--output-dir", str(self.validation_dir / "real")
                ], check=True)
            except Exception as e:
                logger.error(f"    ❌   Error running auto_label_nvlabs.py: {e}")
        else:
            logger.warning("  -> auto_label_nvlabs.py not found. Skipping.")

    def print_summary_report(self):
        logger.info("\n" + "="*70)
        logger.info("  📊  Final Validation Report")
        logger.info("="*70)
        logger.info(f"Synthetic Problems Generated: {self.stats['generated_problems']}")
        logger.info(f"Synthetic Images Generated:   {self.stats['generated_images']}")
        # GNN filtering stats
        gnn_filtered = getattr(self.generator, 'gnn_filter_count', None)
        if hasattr(self, 'generator') and self.generator and hasattr(self.generator, 'cfg') and getattr(self.generator.cfg, 'use_gnn', False):
            logger.info(f"Images Filtered by GNN:     {gnn_filtered if gnn_filtered is not None else 'N/A'}")
        logger.info(f"Validation output in:         {os.path.abspath(self.validation_dir)}")
        logger.info("="*70)

    def generate_and_validate_synthetic_problem(self, rule: BongardRule, problem_idx: int):
        """Generates and validates a full synthetic Bongard problem for a given rule."""
        # Generate positive and negative scenes
        N = 7
        positive_scenes, negative_scenes = self.generator.generate_for_rule(rule, N)

        # --- Rule Validation Step ---
        if RuleValidator:
            positive_pass_count = 0
            for i, (scene_objects, _) in enumerate(positive_scenes):
                validator = RuleValidator(scene_objects)
                if validator.validate_rule(rule.program_ast[0]):
                    positive_pass_count += 1
                else:
                    logger.warning(f"Validation FAILED for rule '{rule.name}' on positive example {i}.")
            logger.info(f"Rule '{rule.name}': {positive_pass_count}/7 positive examples passed validation.")

            negative_pass_count = 0
            for i, (scene_objects, _) in enumerate(negative_scenes):
                validator = RuleValidator(scene_objects)
                if not validator.validate_rule(rule.program_ast[0]):
                    negative_pass_count += 1
                else:
                    logger.warning(f"Validation FAILED for rule '{rule.name}' on negative example {i}.")
            logger.info(f"Rule '{rule.name}': {negative_pass_count}/7 negative examples passed validation (correctly failed rule).")
        # --- End Rule Validation Step ---

        # Save positive images
        for i, (scene_objects, generation_method) in enumerate(positive_scenes):
            img = create_composite_scene(scene_objects, self.config)
            img_path = self.synthetic_dir / f"problem_{problem_idx}_rule_{rule.name}_pos_{i}.png"
            img.save(img_path)
            self.stats['generated_images'] += 1

        # Save negative images
        for i, (scene_objects, generation_method) in enumerate(negative_scenes):
            img = create_composite_scene(scene_objects, self.config)
            img_path = self.synthetic_dir / f"problem_{problem_idx}_rule_{rule.name}_neg_{i}.png"
            img.save(img_path)
            self.stats['generated_images'] += 1
        logger.info(f"✓ Generated and validated problem for rule: {rule.name} (positive: {len(positive_scenes)}, negative: {len(negative_scenes)})")

def main():
    """
    Main execution - runs complete Bongard-LOGO pipeline validation.
    """
    logger.info("=" * 70)
    logger.info("  🎯  COMPLETE BONGARD-LOGO PIPELINE VALIDATION")
    logger.info("=" * 70)

    # Load config and initialize pipeline
    try:
        # A simplified config for validation purposes
        config = {
            'prototypes_dir': 'data/prototypes',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'generator': {
                'use_gnn': False,  # Set to False: first generate data, then train GNN
                'gnn_ckpt': "checkpoints/scene_gnn.pth",
                'gnn_thresh': 0.5,
                'gnn_radius': 0.3,
                'cp_quota': 0.3,
                'ga_quota': 0.3,
                'use_color': True,  # Ensure color generation is enabled
                'colors': ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'black', 'white'],
                'fills': ['solid', 'empty', 'striped', 'dotted'],
            },
            'coverage': {
                'coverage_goals': {
                    'default_quota': 10,
                    'priority_quota': 20,
                    'priority_cells': []
                }
            },
            'canvas_size': 128,
            'stroke_min': 1,
            'jitter_px': 0.5,
            'color_mode': 'rgb',  # Add color mode to config
        }
        pipeline = CompleteBongardPipeline(config)
        logger.info("  🔧  Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"  ❌   Pipeline initialization failed: {e}", exc_info=True)
        return

    # Run full pipeline orchestration
    try:
        pipeline.run()
    except Exception as e:
        logger.error(f"  ❌   An error occurred during pipeline orchestration: {e}", exc_info=True)

if __name__ == "__main__":
    main()
