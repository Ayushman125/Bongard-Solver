# Genetic algorithm configuration for Bongard-LOGO generator
# Add this section to your main config.py or load from YAML if preferred
GENETIC_CONFIG = {
    'population_size': 50,           # Number of candidate scenes per generation
    'num_generations': 30,           # Number of generations to evolve
    'mutation_rate': 0.2,            # Probability of mutation per object
    'crossover_rate': 0.5,           # Probability of crossover between scenes
    'diversity_weight': 0.5,         # Weight for diversity in fitness
    'tester_weight': 0.5,            # Weight for neural tester confidence in fitness
    'coverage_weight': 1.0,          # Weight for coverage in fitness
    'elitism': 2,                    # Number of top scenes to carry over each generation
    'max_attempts': 100,             # Max attempts per scene generation
    'cache_enabled': True,           # Enable cache for coverage tracking
    'seed': 42,                      # Random seed for reproducibility
    # Add more hyperparameters as needed
}

# To use: import GENETIC_CONFIG in GeneticSceneGenerator and BongardSampler
