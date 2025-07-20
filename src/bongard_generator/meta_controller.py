"""
MetaController for adaptive rule selection in Bongard-LOGO problem generation.
Uses a bandit model to dynamically adjust rule selection probabilities based on
generator success/failure, ensuring a diverse and high-quality dataset.
"""
import random
from typing import List, Dict, Any
from .rule_loader import AbstractRule, load_rules
from .bandit_model import BanditModel

class MetaController:
    """Manages adaptive rule selection for the Bongard problem generator."""
    
    def __init__(self, rule_paths: List[str], initial_epsilon: float = 0.1):
        """
        Initialize the MetaController.
        
        Args:
            rule_paths: List of paths to rule directories.
            initial_epsilon: Initial exploration factor for the bandit model.
        """
        self.all_rules = load_rules(rule_paths)
        self.rule_names = [rule.name for rule in self.all_rules]
        
        if not self.all_rules:
            raise ValueError("No rules loaded. Check rule paths.")
            
        self.bandit_model = BanditModel(arms=self.rule_names, epsilon=initial_epsilon)
        print(f"MetaController initialized with {len(self.all_rules)} rules.")

    def select_rules(self, batch_size: int) -> List[AbstractRule]:
        """
        Select a batch of rules using the bandit model.
        
        Args:
            batch_size: The number of rules to select.
            
        Returns:
            A list of selected rule instances.
        """
        if batch_size > len(self.all_rules):
            print(f"Warning: batch_size {batch_size} is larger than available rules {len(self.all_rules)}. Returning all rules.")
            return self.all_rules

        # Use bandit model to sample rule names
        selected_rule_names = self.bandit_model.sample([r.name for r in self.all_rules], batch_size)
        
        # Map names back to rule instances
        selected_rules = [rule for rule in self.all_rules if rule.name in selected_rule_names]
        
        # Ensure correct batch size if there are duplicates or issues
        if len(selected_rules) < batch_size:
            # Fallback to random sampling if bandit selection fails
            additional_rules = random.sample(self.all_rules, batch_size - len(selected_rules))
            selected_rules.extend(additional_rules)
            
        return selected_rules

    def update_rule_feedback(self, rule_name: str, reward: float):
        """
        Update the bandit model with feedback on a rule's performance.
        
        Args:
            rule_name: The name of the rule to update.
            reward: The reward (1 for success, 0 for failure).
        """
        if rule_name in self.rule_names:
            self.bandit_model.update(rule_name, reward)
        else:
            print(f"Warning: Attempted to update non-existent rule '{rule_name}'")

    def get_rule_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Get the current performance statistics for all rules.
        
        Returns:
            A dictionary with rule names as keys and their stats as values.
        """
        return self.bandit_model.stats

# Factory function
def create_meta_controller(rule_paths: List[str], initial_epsilon: float = 0.1) -> MetaController:
    """
    Factory function to create and initialize the MetaController.
    
    Args:
        rule_paths: List of paths to rule directories.
        initial_epsilon: Initial exploration factor for the bandit model.
        
    Returns:
        An instance of MetaController.
    """
    return MetaController(rule_paths, initial_epsilon)
