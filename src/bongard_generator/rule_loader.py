"""
Base classes and loader for Bongard problem rules.
"""
import abc
import os
import importlib
import inspect
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

class AbstractRule(abc.ABC):
    """Abstract base class for all Bongard rules."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """A unique, machine-readable name for the rule (e.g., 'SHAPE_COUNT')."""
        pass

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """A human-readable description of the rule."""
        pass

    @abc.abstractmethod
    def apply(self, objects: List[Dict[str, Any]], is_positive: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Modifies a list of objects to either satisfy or violate the rule.

        Args:
            objects: The initial list of objects to be modified.
            is_positive: If True, the modified objects must satisfy the rule.
                         If False, they must violate it.

        Returns:
            A tuple containing:
            - The modified list of objects.
            - A dictionary of features specific to this rule application.
        """
        pass

# Alias for backward compatibility
BongardRule = AbstractRule

class RuleLoader:
    """Dynamically loads all rule classes from the 'rules' directory."""

    def __init__(self, rule_directory="src/bongard_generator/rules"):
        self.rule_directory = rule_directory
        self.rules = self._load_rules()

    def _load_rules(self) -> List[AbstractRule]:
        """Scans the rule directory and instantiates all AbstractRule subclasses."""
        loaded_rules = []
        # Correctly resolve the rule directory path relative to the project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        absolute_rule_dir = os.path.join(project_root, self.rule_directory)

        if not os.path.isdir(absolute_rule_dir):
            logger.error(f"Rule directory not found: {absolute_rule_dir}")
            return []

        for filename in os.listdir(absolute_rule_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                module_name = f"src.bongard_generator.rules.{filename[:-3]}"
                try:
                    module = importlib.import_module(module_name)
                    for name, cls in inspect.getmembers(module, inspect.isclass):
                        if issubclass(cls, AbstractRule) and cls is not AbstractRule:
                            loaded_rules.append(cls())
                            logger.info(f"Successfully loaded rule: {name}")
                except Exception as e:
                    logger.error(f"Failed to load rule from {filename}: {e}")
        return loaded_rules

    def get_rules(self) -> List[AbstractRule]:
        """Returns the list of loaded rule instances."""
        return self.rules

def get_all_rules() -> List[AbstractRule]:
    """
    Global function to get all available rules.
    Creates a RuleLoader instance and returns all loaded rules.
    """
    loader = RuleLoader()
    return loader.get_rules()

def get_rule_by_description(description: str) -> AbstractRule:
    """
    Get a rule by its description.
    """
    rules = get_all_rules()
    for rule in rules:
        if rule.description.lower() == description.lower():
            return rule
    return None
