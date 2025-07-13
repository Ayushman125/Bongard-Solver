import subprocess
import yaml

def run_experiment(name, params):
    conf = yaml.safe_load(open('config.yaml'))
    conf.update(params)
    with open('config_temp.yaml', 'w') as f:
        yaml.dump(conf, f)
    subprocess.run(['dvc', 'repro', '--pipeline', 'prepare'])
    subprocess.run(['python', 'training/train.py', '--config', 'config_temp.yaml'])
    # record metrics
    # parse logs, save to experiments/results.csv

if __name__ == '__main__':
    experiments = [
        ('base', {}),
        ('mixup_only', {'augment': {'mixup_prob': 0.5, 'cutmix_prob': 0.0}}),
        ('cutmix_only', {'augment': {'mixup_prob': 0.0, 'cutmix_prob': 0.5}}),
        ('dali_only', {'pipeline': {'use_dali': True}}),
    ]
    for name, params in experiments:
        run_experiment(name, params)
