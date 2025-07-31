import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

class HierarchicalPredicatePredictor(nn.Module):
    """
    Hierarchical predicate learning with Bayesian inference and uncertainty estimation.
    """
    def __init__(self, feature_dim: int = 384):
        super().__init__()
        self.feature_dim = feature_dim
        self.predicate_hierarchy = self._create_predicate_hierarchy()
        self.super_category_to_predicates = self._build_category_mapping()
        # Feature encoder
        self.feature_encoder = nn.Linear(feature_dim * 2, feature_dim)
        # Knowledge fusion layer
        self.knowledge_fusion = KnowledgeFusionLayer(feature_dim)
        # Super-category classifier
        self.super_category_head = nn.Linear(feature_dim, len(self.super_category_to_predicates))
        # Predicate classifiers per super-category
        self.predicate_heads = nn.ModuleDict({
            cat: nn.Linear(feature_dim, len(preds))
            for cat, preds in self.super_category_to_predicates.items()
        })
        # Uncertainty estimation
        self.uncertainty_head = nn.Linear(feature_dim, 1)
    def forward(self, subject_features: torch.Tensor, object_features: torch.Tensor, knowledge_embeddings: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with Bayesian hierarchical prediction.
        Returns:
            Dictionary containing super-category probabilities, 
            predicate probabilities per category, and uncertainty estimates.
        """
        # Combine subject and object features
        combined_features = torch.cat([subject_features, object_features], dim=-1)
        # Encode features
        encoded_features = self.feature_encoder(combined_features)
        # Integrate knowledge embeddings if available
        if knowledge_embeddings is not None:
            encoded_features = self.knowledge_fusion(encoded_features, knowledge_embeddings)
        # Predict super-categories with uncertainty
        super_category_logits = self.super_category_head(encoded_features)
        super_category_probs = F.softmax(super_category_logits, dim=-1)
        # Predict predicates within each super-category
        predicate_predictions = {}
        uncertainty_estimates = {}
        for category in self.super_category_to_predicates.keys():
            # Get category-specific predictions
            category_logits = self.predicate_heads[category](encoded_features)
            category_probs = F.softmax(category_logits, dim=-1)
            predicate_predictions[category] = category_probs
            # Estimate uncertainty for this category
            uncertainty = self.uncertainty_head(encoded_features)
            uncertainty_estimates[category] = uncertainty
        return {
            'super_category_probs': super_category_probs,
            'predicate_predictions': predicate_predictions,
            'uncertainty_estimates': uncertainty_estimates,
            'encoded_features': encoded_features
        }
    def predict_with_bayesian_inference(self, subject_features: torch.Tensor, object_features: torch.Tensor, knowledge_embeddings: Optional[torch.Tensor] = None, top_k: int = 5) -> List[Dict]:
        """
        Perform Bayesian inference to get top-k predicate predictions with confidence.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(subject_features, object_features, knowledge_embeddings)
            super_category_probs = outputs['super_category_probs']
            predicate_predictions = outputs['predicate_predictions']
            uncertainty_estimates = outputs['uncertainty_estimates']
            # Bayesian marginalization over super-categories
            final_predictions = []
            for category_idx, category in enumerate(self.super_category_to_predicates.keys()):
                super_prob = super_category_probs[0, category_idx].item()
                predicate_probs = predicate_predictions[category][0]
                uncertainty = uncertainty_estimates[category][0].item()
                predicates = self.super_category_to_predicates[category]
                for pred_idx, predicate in enumerate(predicates):
                    # Marginal probability: P(predicate|data) = P(predicate|category) * P(category|data)
                    marginal_prob = predicate_probs[pred_idx].item() * super_prob
                    # Confidence with uncertainty consideration
                    confidence = marginal_prob * (1 - uncertainty)
                    final_predictions.append({
                        'predicate': predicate,
                        'probability': marginal_prob,
                        'confidence': confidence,
                        'super_category': category,
                        'super_category_prob': super_prob,
                        'uncertainty': uncertainty
                    })
            # Sort by confidence and return top-k
            final_predictions.sort(key=lambda x: x['confidence'], reverse=True)
            return final_predictions[:top_k]
    def _create_predicate_hierarchy(self) -> Dict:
        """Create sophisticated predicate hierarchy for Bongard problems."""
        hierarchy = {
            'spatial_static': {
                'description': 'Static spatial relationships',
                'predicates': ['on', 'under', 'beside', 'near', 'far', 'inside', 'outside', 
                              'above', 'below', 'left_of', 'right_of', 'in_front_of', 'behind'],
                'weight': 0.9
            },
            'spatial_dynamic': {
                'description': 'Dynamic spatial relationships',
                'predicates': ['moving_towards', 'moving_away', 'following', 'chasing',
                              'approaching', 'departing', 'orbiting', 'crossing'],
                'weight': 0.85
            },
            'functional': {
                'description': 'Functional relationships',
                'predicates': ['supports', 'contains', 'uses', 'operates', 'controls',
                              'activates', 'powers', 'connects', 'attaches'],
                'weight': 0.8
            },
            'categorical': {
                'description': 'Categorical relationships',
                'predicates': ['is_a', 'part_of', 'similar_to', 'type_of', 'instance_of',
                              'belongs_to', 'member_of', 'category_of'],
                'weight': 0.75
            },
            'physical': {
                'description': 'Physical interaction relationships',
                'predicates': ['touches', 'collides', 'pushes', 'pulls', 'blocks',
                              'penetrates', 'overlaps', 'intersects'],
                'weight': 0.88
            },
            'abstract': {
                'description': 'Abstract conceptual relationships',
                'predicates': ['represents', 'symbolizes', 'implies', 'suggests',
                              'indicates', 'signifies', 'denotes', 'exemplifies'],
                'weight': 0.7
            }
        }
        return hierarchy
    def _build_category_mapping(self) -> Dict[str, List[str]]:
        """Build mapping from super-categories to predicates."""
        mapping = {}
        for category, info in self.predicate_hierarchy.items():
            mapping[category] = info['predicates']
        return mapping

class KnowledgeFusionLayer(nn.Module):
    """Advanced knowledge fusion layer for integrating prior knowledge."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Multi-head attention for knowledge fusion
        self.knowledge_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1
        )
        # Knowledge transformation layers
        self.knowledge_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
    def forward(self, 
                visual_features: torch.Tensor,
                knowledge_embeddings: torch.Tensor) -> torch.Tensor:
        """Fuse visual features with knowledge embeddings."""
        # Transform knowledge embeddings
        transformed_knowledge = self.knowledge_transform(knowledge_embeddings)
        # Apply attention to align knowledge with visual features
        attended_knowledge, _ = self.knowledge_attention(
            query=visual_features.unsqueeze(0),
            key=transformed_knowledge.unsqueeze(0),
            value=transformed_knowledge.unsqueeze(0)
        )
        attended_knowledge = attended_knowledge.squeeze(0)
        # Compute fusion gate
        combined = torch.cat([visual_features, attended_knowledge], dim=-1)
        gate = self.fusion_gate(combined)
        # Gated fusion
        fused_features = gate * visual_features + (1 - gate) * attended_knowledge
        return fused_features

class BayesianPredicateLoss(nn.Module):
    """Advanced loss function for hierarchical predicate learning."""
    def __init__(self, 
                 super_category_weight: float = 0.3,
                 predicate_weight: float = 0.5,
                 uncertainty_weight: float = 0.2,
                 label_smoothing: float = 0.1):
        super().__init__()
        self.super_category_weight = super_category_weight
        self.predicate_weight = predicate_weight
        self.uncertainty_weight = uncertainty_weight
        # Loss functions
        self.super_category_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.predicate_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.uncertainty_loss = nn.MSELoss()
    def forward(self, 
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute hierarchical loss with uncertainty regularization.
        """
        # Super-category loss
        super_loss = self.super_category_loss(
            predictions['super_category_probs'],
            targets['super_category']
        )
        # Category-specific predicate losses
        predicate_losses = []
        for category in predictions['predicate_predictions']:
            if category in targets['category_specific_targets']:
                pred_loss = self.predicate_loss(
                    predictions['predicate_predictions'][category],
                    targets['category_specific_targets'][category]
                )
                predicate_losses.append(pred_loss)
        avg_predicate_loss = torch.mean(torch.stack(predicate_losses)) if predicate_losses else torch.tensor(0.0)
        # Uncertainty regularization
        uncertainty_target = targets.get('uncertainty', torch.zeros_like(predictions['uncertainty_estimates'][list(predictions['uncertainty_estimates'].keys())[0]]))
        uncertainty_losses = []
        for category in predictions['uncertainty_estimates']:
            unc_loss = self.uncertainty_loss(
                predictions['uncertainty_estimates'][category],
                uncertainty_target
            )
            uncertainty_losses.append(unc_loss)
        avg_uncertainty_loss = torch.mean(torch.stack(uncertainty_losses))
        # Total loss
        total_loss = (
            self.super_category_weight * super_loss +
            self.predicate_weight * avg_predicate_loss +
            self.uncertainty_weight * avg_uncertainty_loss
        )
        return {
            'total_loss': total_loss,
            'super_category_loss': super_loss,
            'predicate_loss': avg_predicate_loss,
            'uncertainty_loss': avg_uncertainty_loss
        }
