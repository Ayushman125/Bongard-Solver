from bongard_generator.dataset import BongardDataset, create_composite_scene
from bongard_generator.visualize import show_mosaic, plot_rule_distribution

# 1) instantiate with your full rule set
rules = [
    ('SHAPE(circle)', 1),
    ('SHAPE(triangle)', 2),
    ('SHAPE(square)', 3),
    ('FILL(solid)', 1),
    ('FILL(outline)', 2),
    ('COUNT(2)', 2),
    ('COUNT(3)', 3),
    ('RELATION(overlap)', 2),
    ('RELATION(near)', 2)
]

class SyntheticBongardDataset:
    def __init__(self, rules, img_size=128, grayscale=True):
        self.dataset = BongardDataset(canvas_size=img_size)
        self.examples = []
        for rule_desc, count in rules:
            rule = self.dataset._select_rule_for_generation()
            for i in range(count):
                scene = self.dataset._generate_single_scene(rule, num_objects=2, is_positive=True)
                if scene:
                    img = create_composite_scene(scene['objects'], img_size)
                    self.examples.append({'image': img, 'rule': rule_desc, 'label': 1, 'scene_graph': scene['scene_graph']})
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        return self.examples[idx]

if __name__ == '__main__':
    ds = SyntheticBongardDataset(rules=rules, img_size=128, grayscale=True)
    show_mosaic(ds, n=16, cols=4)
    plot_rule_distribution(ds)
