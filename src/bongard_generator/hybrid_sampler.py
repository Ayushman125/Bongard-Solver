import math
from bongard_generator.rule_loader import get_all_rules
from bongard_generator.dataset import BongardDataset
from bongard_generator.genetic_generator import GeneticSceneGenerator

class HybridSampler:
    def __init__(self, cfg, tester):
        self.cfg = cfg
        self.cp_quota = int(cfg['data']['total'] * cfg['data']['hybrid_split']['cp'])
        self.ga_quota = int(cfg['data']['total'] * cfg['data']['hybrid_split']['ga'])
        self.rules = get_all_rules()
        self.cp_sampler = CPSatSampler(cfg)
        self.ga_sampler = GeneticSceneGenerator(cfg, tester)

    def build_holdout(self):
        imgs, lbls = [], []
        cp_per_rule = math.ceil(self.cp_quota / len(self.rules))
        ga_per_rule = math.ceil(self.ga_quota / len(self.rules))

        # Phase A: CP-SAT seeding
        for rule in self.rules:
            ds = BongardDataset(target_quota=cp_per_rule, rule_list=[rule.name])
            for sample in ds:
                imgs.append(sample['image'])
                lbls.append(sample['label'])

        # Phase B: Genetic diversification
        for rule in self.rules:
            for _ in range(ga_per_rule):
                img, lbl = self.ga_sampler.generate_problem(rule)
                imgs.append(img)
                lbls.append(lbl)

        return imgs, lbls

class CPSatSampler:
    def __init__(self, cfg):
        self.cfg = cfg
    def sample_problem(self, rule):
        ds = BongardDataset(target_quota=1, rule_list=[rule.name])
        sample = ds[0]
        return sample['image'], sample['label']
