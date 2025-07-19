import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def show_mosaic(dataset, n=16, cols=4):
    idxs = np.random.choice(len(dataset), size=n, replace=False)
    rows = int(np.ceil(n/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    for ax, idx in zip(axes.flatten(), idxs):
        sample = dataset[idx]
        img = sample['image'].squeeze().numpy() if hasattr(sample['image'], 'numpy') else np.array(sample['image'])
        rule_title = str(sample.get('rule', ''))
        if len(rule_title) > 20:
            rule_title = rule_title[:20] + '...'
        ax.imshow(img, cmap='gray')
        ax.set_title(rule_title + f" / L{sample.get('label', '')}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def plot_rule_distribution(dataset):
    cnt = Counter(sample['rule'] for sample in dataset)
    rules, freqs = zip(*cnt.most_common())
    plt.figure(figsize=(10,4))
    plt.bar(rules, freqs)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Count")
    plt.title("Frequency of Each Rule in Dataset")
    plt.show()

def plot_attr_distribution(dataset, attr):
    cnt = Counter(o[attr] for sample in dataset for o in sample['scene_graph']['objects'])
    items, freqs = zip(*cnt.items())
    plt.figure()
    plt.bar(items, freqs)
    plt.title(f"{attr} Distribution")
    plt.show()
