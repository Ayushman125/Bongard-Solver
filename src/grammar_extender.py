"""
Meta-Grammar Generator Stub with Deterministic Sampling
Phase 1 Module
"""

import random
import yaml
from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from integration.task_profiler import TaskProfiler
from integration.data_validator import DataValidator

class GrammarExtender:
    """
    Dynamically extends a Domain-Specific Language (DSL) grammar by proposing new operators.

    This module analyzes the existing grammar and a set of contextual keywords to propose
    relevant new operators. It uses TF-IDF vectorization to find operators that are
    semantically similar to the provided context, ensuring that proposals are both novel
and relevant.

    Attributes:
        tau (float): A temperature parameter controlling the diversity of proposals.
                     Higher values lead to more diverse but potentially less relevant operators.
        grammar (Dict): The loaded grammar configuration, typically from a YAML file.
        profiler (TaskProfiler): An instance of the TaskProfiler for performance monitoring.
        dv (DataValidator): An instance of the DataValidator for schema validation.
        vectorizer (TfidfVectorizer): A TF-IDF vectorizer fitted on the grammar's operator descriptions.
        operator_matrix (np.ndarray): A matrix of TF-IDF vectors for all operators in the grammar.
    """
    def __init__(self, tau: float = 0.3, grammar_path: str = 'resources/grammar_base.yaml') -> None:
        self.tau = tau
        self.profiler = TaskProfiler()
        self.dv = DataValidator()
        try:
            with open(grammar_path, 'r', encoding='utf-8') as f:
                self.grammar = yaml.safe_load(f)
            
            # Assuming a schema validation step is desired.
            # self.dv.validate(self.grammar, 'grammar_config.schema.json')

            self._build_semantic_model()

        except FileNotFoundError:
            raise RuntimeError(f"Grammar file not found at: {grammar_path}")
        except Exception as e:
            raise RuntimeError(f"Failed loading or processing grammar: {e}")

    def _build_semantic_model(self):
        """Builds a TF-IDF vectorizer from the operator descriptions in the grammar."""
        with self.profiler.profile("build_semantic_model"):
            operator_docs = [op.get('description', '') for op in self.grammar.get('operators', [])]
            if not any(operator_docs):
                # Handle case where no descriptions are available
                self.vectorizer = None
                self.operator_matrix = None
                return

            self.vectorizer = TfidfVectorizer(stop_words='english')
            self.operator_matrix = self.vectorizer.fit_transform(operator_docs)

    def propose(self, known_operators: List[str], context: Dict[str, Any], topk: int = 5) -> List[str]:
        """
        Proposes new operators based on semantic similarity to the given context.

        Args:
            known_operators (List[str]): A list of operators already in use, to be excluded from proposals.
            context (Dict[str, Any]): A dictionary containing contextual information, such as keywords
                                     or descriptions of the task requiring new operators.
            topk (int): The maximum number of new operators to propose.

        Returns:
            List[str]: A list of proposed operator names, sorted by relevance.
        """
        if not self.vectorizer:
            return []

        with self.profiler.profile("propose_operators"):
            # Use 'keywords' if present, else any string value in context
            if 'keywords' in context and context['keywords']:
                context_text = " ".join(context['keywords'])
            else:
                # Fallback: concatenate all string values in context
                context_text = " ".join(str(v) for v in context.values() if isinstance(v, str))
            if not context_text:
                return []

            context_vec = self.vectorizer.transform([context_text])
            
            # Compute cosine similarity between the context and all operators
            sim_scores = cosine_similarity(context_vec, self.operator_matrix).flatten()

            # Apply temperature scaling (softmax with tau) to influence diversity
            if self.tau > 0:
                prob_scores = np.exp(sim_scores / self.tau) / np.sum(np.exp(sim_scores / self.tau))
            else:
                prob_scores = sim_scores

            # Get all operators, filter out known ones
            all_operators = [op['name'] for op in self.grammar.get('operators', [])]
            
            # Create a list of (operator_name, score) tuples
            candidates = []
            for i, op_name in enumerate(all_operators):
                if op_name not in known_operators:
                    candidates.append((op_name, prob_scores[i]))

            # Sort candidates by score in descending order
            candidates.sort(key=lambda x: x[1], reverse=True)

            # If no candidates, fallback to all operators not in known_operators
            if not candidates:
                fallback_ops = [op for op in all_operators if op not in known_operators]
                # Return at least one operator if available
                return fallback_ops[:max(1, topk)]

            # Return the names of the top-k candidates
            return [name for name, score in candidates[:topk]]

    def estimate_coverage(self, puzzle_features: List[Dict]) -> float:
        """
        Estimates the grammar's coverage over a set of puzzle features.
        (This is a stub and would require a more complex implementation).

        Args:
            puzzle_features (List[Dict]): A list of feature dictionaries for held-out puzzles.

        Returns:
            float: An estimated coverage score between 0.0 and 1.0.
        """
        # Placeholder logic: a real implementation would attempt to parse or apply
        # grammar rules to the features.
        if not self.grammar.get('operators'):
            return 0.0
        
        # Simple stub: coverage is proportional to the number of operators.
        # A real implementation would be much more sophisticated.
        coverage = min(1.0, len(self.grammar['operators']) / 20.0) # Assume 20 operators is good coverage
        return coverage

