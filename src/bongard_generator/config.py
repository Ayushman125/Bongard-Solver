from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

@dataclass
class GeneratorConfig:
    """A central configuration object for the Bongard problem generator."""
    
    # General settings
    img_size: int = 256
    
    # Shape generation
    min_shapes: int = 2
    max_shapes: int = 5
    allow_overlap: bool = False
    
    # Color and fill
    shape_color: Tuple[int, int, int] = (0, 0, 0)  # Black
    bg_color: Tuple[int, int, int] = (255, 255, 255) # White
    fill_type: str = 'solid' # 'solid', 'striped', 'dotted'
    
    # Domain randomization
    enable_jitter: bool = True
    jitter_strength: float = 0.02
    enable_rotation: bool = True
    
    # Advanced Textures & Backgrounds
    bg_texture: str = "none"  # "none", "noise", "checker"
    noise_level: float = 0.1
    noise_opacity: float = 0.5
    checker_size: int = 20
    
    # Prototype Shapes
    prototype_path: str = "shapebongordV2" # Relative to project root

    # Style-Transfer Integration (GAN)
    use_gan_stylization: bool = False
    gan_model_path: str = "checkpoints/gan_generator.pth" # Example path

    # Meta-Controller & Rule Complexity
    use_meta_controller: bool = False
    rule_paths: List[str] = field(default_factory=lambda: ["src/bongard_generator/rules"])
    # In a real system, this might point to a file with solver performance data
    solver_feedback_path: Optional[str] = None
    
    # Coverage and Sampling
    use_coverage_tracker: bool = True
    use_hybrid_sampler: bool = True
    
    # For action-based generation
    enable_actions: bool = True
    action_library: List[str] = field(default_factory=lambda: ["line", "arc", "circle", "rectangle"])

@dataclass
class RuleConfig:
    """Placeholder for rule-specific configurations if needed."""
    pass

    """Configuration for a specific Bongard rule."""
    name: str
    parameters: Dict[str, Any]

@dataclasses.dataclass
class BongardProblemConfig:
    """Overall configuration for generating a Bongard problem."""
    positive_config: GeneratorConfig
    negative_config: GeneratorConfig
    rule_config: RuleConfig
