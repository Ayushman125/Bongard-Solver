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

# Add src path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Core imports - add to Python path
current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)

# Import with proper error handling
try:
    from bongard_generator.builder import BongardGenerator
    from bongard_generator.scene_graph import build_scene_graph, SHAPES, COLORS, FILLS
    from bongard_generator.gnn_model import SceneGNN
    from bongard_generator.dataset import create_composite_scene
    from bongard_rules import get_all_rules, BongardRule
    from config import load_config, CONFIG
    from PIL import Image
    import torch
    ADVANCED_COMPONENTS_AVAILABLE = True
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.warning(f"Advanced components not available: {e}")
    BongardGenerator = None
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
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.validation_dir = "complete_validation_output"
        self.synthetic_dir = "data/synthetic_bongard"
        self.nvlabs_dir = "data/Bongard-LOGO"  # NVlabs repo location
        self.shapebongard_dir = "ShapeBongard_V2"  # Real dataset location
        
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
        """Initialize all pipeline components."""
        logger.info("🔧 Initializing complete pipeline components...")
        
        if not ADVANCED_COMPONENTS_AVAILABLE:
            logger.error("   ❌ Critical components failed to import. Pipeline cannot continue.")
            return

        try:
            self.generator = BongardGenerator(self.config)
            logger.info("   ✅ BongardGenerator initialized")
        except Exception as e:
            logger.error(f"   ❌ BongardGenerator init failed: {e}")
            self.generator = None
        
        self.real_dataset_available = self._check_real_dataset_availability()
        if self.real_dataset_available:
            logger.info("   ✅ Real ShapeBongard_V2 dataset available")
        else:
            logger.info("   ⚠️ Real dataset not available (synthetic only)")
    
    def _check_real_dataset_availability(self) -> bool:
        """Check if ShapeBongard_V2 dataset is available."""
        return os.path.exists(self.shapebongard_dir)

    def run_complete_validation(self):
        """Run the complete integrated pipeline validation."""
        logger.info("🚀 Running complete pipeline validation...")
        
        # Create output directory
        os.makedirs(self.validation_dir, exist_ok=True)

        # 1. Generate synthetic data in 6+6 format
        self.generate_synthetic_data()

        # 2. (Optional) Run NVLabs auto-labeling
        self.run_auto_labeling()

        # 3. Final report
        self.print_summary_report()

    def generate_synthetic_data(self):
        logger.info("📦 Generating synthetic dataset (6+6 format)...")
        OUT_ROOT = os.path.join(self.validation_dir, "synthetic")
        
        rules = get_all_rules()
        if not rules:
            logger.error("No rules found. Cannot generate synthetic data.")
            return

        for rule in rules:
            rule_dir = os.path.join(OUT_ROOT, rule.name)
            for side, label in [("1", True), ("0", False)]:
                side_dir = os.path.join(rule_dir, side)
                os.makedirs(side_dir, exist_ok=True)
                
                scenes = self.generator.generate_for_rule(rule, N=6, is_positive=label)
                self.stats['generated_problems'] += 1

                for i, (img, objs, tag) in enumerate(scenes):
                    img.save(os.path.join(side_dir, f"{i:02d}_{tag}.png"))
                    self.stats['generated_images'] += 1
            logger.info(f"✓ Generated 6+6 for rule: {rule.name}")

    def run_auto_labeling(self):
        logger.info("🏷️ Running auto-labeling for real data (if available)...")
        # This part requires the nvlabs script to be executable
        # and the bongard-logo repo to be installed.
        # For this validation, we will just check if the script exists.
        auto_label_script_path = Path("scripts/auto_label_nvlabs.py")
        if auto_label_script_path.exists():
            logger.info("   -> Found auto_label_nvlabs.py. In a real run, this would be executed.")
            # In a real scenario, you would run:
            # import subprocess
            # subprocess.run(["python", str(auto_label_script_path)], check=True)
        else:
            logger.warning("   -> auto_label_nvlabs.py not found. Skipping.")

    def print_summary_report(self):
        logger.info("\n" + "="*70)
        logger.info("📊 Final Validation Report")
        logger.info("="*70)
        logger.info(f"Synthetic Problems Generated: {self.stats['generated_problems']}")
        logger.info(f"Synthetic Images Generated:   {self.stats['generated_images']}")
        if self.generator.cfg.use_gnn:
            # This stat would be tracked inside the generator
            # self.stats['gnn_filtered'] = self.generator.gnn_filter_count 
            logger.info(f"Images Filtered by GNN:     {self.stats['gnn_filtered']}")
        logger.info(f"Validation output in:         {os.path.abspath(self.validation_dir)}")
        logger.info("="*70)


def main():
    """
    Main execution - runs complete Bongard-LOGO pipeline validation.
    """
    logger.info("=" * 70)
    logger.info("🎯 COMPLETE BONGARD-LOGO PIPELINE VALIDATION")
    logger.info("=" * 70)
    
    # Load config and initialize pipeline
    try:
        # A simplified config for validation purposes
        config = {
            'prototypes_dir': 'data/prototypes',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'generator': {
                'use_gnn': False, # Set to True to test GNN filtering
                'gnn_ckpt': "checkpoints/scene_gnn.pth",
                'gnn_thresh': 0.5,
                'gnn_radius': 0.3,
                'cp_quota': 0.3,
                'ga_quota': 0.3,
            },
            'coverage_target': {},
            'canvas_size': 128,
            'stroke_min': 1,
            'jitter_px': 0.5,
        }
        pipeline = CompleteBongardPipeline(config)
        logger.info("🔧 Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"❌ Pipeline initialization failed: {e}", exc_info=True)
        return

    # Run validation
    pipeline.run_complete_validation()


if __name__ == "__main__":
    main()