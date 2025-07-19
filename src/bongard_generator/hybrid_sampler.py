import math
import random
from typing import List, Tuple, Any
from .rule_loader import get_all_rules
from .dataset import BongardDataset
from .genetic_generator import GeneticSceneGenerator

class HybridSampler:
    """Hybrid sampler that combines CP-SAT seeding with genetic diversification."""
    
    def __init__(self, cfg, tester=None):
        self.cfg = cfg
        
        # Handle both dict and SamplerConfig object
        if hasattr(cfg, 'data'):
            # YAML config structure
            total = cfg['data']['total']
            hybrid_split = cfg['data'].get('hybrid_split', {'cp': 0.7, 'ga': 0.3})
        else:
            # SamplerConfig object or different structure
            total = getattr(cfg, 'total', 1000)
            hybrid_split = getattr(cfg, 'hybrid_split', {'cp': 0.7, 'ga': 0.3})
        
        self.cp_quota = int(total * hybrid_split['cp'])
        self.ga_quota = total - self.cp_quota  # Ensure exact total
        self.rules = get_all_rules()
        
        # Two samplers with unified API
        self.cp_sampler = BongardSampler(cfg, generator_mode="cp_sat")
        self.ga_sampler = BongardSampler(cfg, generator_mode="genetic")
        
        print(f"→ Hybrid build: total={total}, cp={self.cp_quota}, ga={self.ga_quota}")
        print("→ Rules:", [r.description for r in self.rules])

    def build_synth_holdout(self, n=None) -> Tuple[List[Any], List[int]]:
        """Build synthetic holdout dataset using hybrid CP-SAT + genetic approach."""
        imgs, lbls = [], []
        
        # Phase A: CP-SAT Seeding
        per_rule_cp = math.ceil(self.cp_quota / len(self.rules))
        for rule in self.rules:
            for _ in range(per_rule_cp):
                pos, neg = self.cp_sampler.sample_problem(
                    rule_description=rule.description, num_pos=1, num_neg=1
                )
                imgs += pos + neg
                lbls += [1]*len(pos) + [0]*len(neg)
                print(f"  • CP pass: rule={rule.description}, got pos={len(pos)}, neg={len(neg)}")
        
        # Phase B: Genetic Diversification  
        per_rule_ga = math.ceil(self.ga_quota / len(self.rules))
        for rule in self.rules:
            for _ in range(per_rule_ga):
                pos, neg = self.ga_sampler.sample_problem(
                    rule_description=rule.description, num_pos=1, num_neg=1
                )
                imgs += pos + neg
                lbls += [1]*len(pos) + [0]*len(neg)
                print(f"  • GA pass: rule={rule.description}, got pos={len(pos)}, neg={len(neg)}")
        
        # Final Assembly & Shuffle
        combined = list(zip(imgs, lbls))
        random.shuffle(combined)
        imgs, lbls = zip(*combined)
        
        return list(imgs), list(lbls)

class BongardSampler:
    """Unified sampler interface for both CP-SAT and genetic modes."""
    
    def __init__(self, cfg, generator_mode="cp_sat"):
        self.cfg = cfg
        self.mode = generator_mode
        if generator_mode == "genetic":
            self.ga_generator = GeneticSceneGenerator(cfg, None)
    
    def sample_problem(self, rule_description: str, num_pos: int = 1, num_neg: int = 1):
        """Sample a problem with positive and negative scenes."""
        if self.mode == "cp_sat":
            return self._sample_cp_sat(rule_description, num_pos, num_neg)
        elif self.mode == "genetic":
            return self._sample_genetic(rule_description, num_pos, num_neg)
        else:
            raise ValueError(f"Unknown generator mode: {self.mode}")
    
    def _sample_cp_sat(self, rule_description: str, num_pos: int, num_neg: int):
        """Sample using CP-SAT/dataset approach."""
        try:
            # Find rule by description
            rules = get_all_rules()
            rule = next((r for r in rules if r.description == rule_description), None)
            if not rule:
                rule = next((r for r in rules if getattr(r, 'name', '') == rule_description), None)
            
            if rule:
                # Create dataset with small quota for sampling
                ds = BongardDataset(target_quota=max(num_pos, num_neg, 2), rule_list=[getattr(rule, 'name', rule.description)])
                pos_imgs, neg_imgs = [], []
                
                # Collect examples from dataset
                for sample in ds:
                    if sample['label'] == 1 and len(pos_imgs) < num_pos:
                        pos_imgs.append(sample['image'])
                    elif sample['label'] == 0 and len(neg_imgs) < num_neg:
                        neg_imgs.append(sample['image'])
                    
                    # Stop once we have enough
                    if len(pos_imgs) >= num_pos and len(neg_imgs) >= num_neg:
                        break
                
                # Fill missing with mock images if needed
                from PIL import Image
                while len(pos_imgs) < num_pos:
                    pos_imgs.append(Image.new('L', (128, 128), color=255))
                while len(neg_imgs) < num_neg:
                    neg_imgs.append(Image.new('L', (128, 128), color=255))
                
                return pos_imgs, neg_imgs
        except Exception as e:
            print(f"CP-SAT sampling failed for {rule_description}: {e}")
            
        # Fallback: return mock images
        from PIL import Image
        pos_imgs = [Image.new('L', (128, 128), color=255) for _ in range(num_pos)]
        neg_imgs = [Image.new('L', (128, 128), color=255) for _ in range(num_neg)]
        return pos_imgs, neg_imgs
    
    def _sample_genetic(self, rule_description: str, num_pos: int, num_neg: int):
        """Sample using genetic generator."""
        try:
            # Find rule by description  
            rules = get_all_rules()
            rule = next((r for r in rules if r.description == rule_description), None)
            if not rule:
                rule = next((r for r in rules if getattr(r, 'name', '') == rule_description), None)
                
            if rule and hasattr(self.ga_generator, 'generate_problem'):
                pos_imgs, neg_imgs = [], []
                
                # Generate positive samples
                for _ in range(num_pos):
                    img, lbl = self.ga_generator.generate_problem(rule, is_positive=True)
                    pos_imgs.append(img)
                    
                # Generate negative samples  
                for _ in range(num_neg):
                    img, lbl = self.ga_generator.generate_problem(rule, is_positive=False)
                    neg_imgs.append(img)
                    
                return pos_imgs, neg_imgs
        except Exception as e:
            print(f"Genetic sampling failed for {rule_description}: {e}")
            
        # Fallback: return mock images
        from PIL import Image
        pos_imgs = [Image.new('L', (128, 128), color=255) for _ in range(num_pos)]
        neg_imgs = [Image.new('L', (128, 128), color=255) for _ in range(num_neg)]
        return pos_imgs, neg_imgs
