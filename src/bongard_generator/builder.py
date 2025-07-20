# src/bongard_generator/builder.py (master generator)
from .cp_sampler       import CPSampler
from .genetic_sampler import GeneticSampler
from .actions         import PrototypeAction
from .dataset         import create_composite_scene
from .coverage        import CoverageTracker
from .scene_graph import build_scene_graph
from .gnn_model  import SceneGNN
import torch

class BongardGenerator:
    def __init__(self, cfg):
        # self.cp    = CPSampler(cfg.cp)
        # self.ga    = GeneticSampler(cfg.ga, self.coverage)
        self.proto = PrototypeAction(cfg.prototypes_dir)
        self.cfg   = cfg.generator
        self.coverage = CoverageTracker(cfg.coverage_target)
        
        if self.cfg.use_gnn:
            self._gnn = self._lazy_load_gnn(cfg)

    def _lazy_load_gnn(self, cfg):
        # lazy-load once
        # These SHAPES, COLORS, FILLS should be imported from a central place
        SHAPES = ["circle","square","triangle","pentagon","star","arc","zigzag","prototype"]
        COLORS = ["black","white","red","blue","green"]
        FILLS  = ["solid","hollow","striped","dotted"]
        feat_dim = len(SHAPES)+len(COLORS)+len(FILLS)+3
        gnn = SceneGNN(feat_dim).to(cfg.device)
        gnn.load_state_dict(torch.load(cfg.generator.gnn_ckpt))
        gnn.eval()
        return gnn

    def generate_for_rule(self, rule, N, is_positive=True):
        # This is a simplified generation logic.
        # In a real implementation, you would call the appropriate sampler.
        # For now, we'll just create some dummy objects to test the pipeline.
        
        # cp_n  = int(N * self.cfg.cp_quota)
        # ga_n  = int(N * self.cfg.ga_quota)
        # pt_n  = N - cp_n - ga_n
        scenes=[]

        # # CP-SAT branch
        # for _ in range(cp_n):
        #     objs = self.cp.sample_scene(rule)
        #     scenes.append((objs,'cp_sat'))

        # # Genetic branch
        # for _ in range(ga_n):
        #     objs = self.ga.sample_scene(rule)
        #     scenes.append((objs,'genetic'))

        # # Prototype branch
        # for _ in range(pt_n):
        #     objs = self.proto.inject(rule)  # returns objs with 'prototype' slots
        #     scenes.append((objs,'prototype'))

        # Dummy object generation for demonstration
        for i in range(N):
            num_objs = random.randint(1, 5)
            objs = []
            for _ in range(num_objs):
                obj = {
                    'shape': 'circle' if is_positive else 'square',
                    'x': random.randint(20, 108),
                    'y': random.randint(20, 108),
                    'size': random.randint(10, 30),
                    'fill': 'solid',
                    'color': 'black'
                }
                # The rule would modify the objects here
                # rule.apply(obj) 
                objs.append(obj)
            scenes.append((objs, 'dummy'))


        # Render + record coverage
        out=[]
        for objs,tag in scenes:
            # GNN filtering
            if self.cfg.use_gnn:
                data = build_scene_graph(objs, self.cfg)
                data = data.to(self.cfg.device)
                with torch.no_grad():
                    score = self._gnn(data).item()
                if score < self.cfg.generator.gnn_thresh:
                    continue   # discard low-scoring scene

            img = create_composite_scene(objs, self.cfg)
            self.coverage.record(rule, objs)
            out.append((img,objs,tag))
        return out
