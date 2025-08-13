
import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(f"[DEBUG] Adding PROJECT_ROOT to sys.path: {PROJECT_ROOT}")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import argparse
import json
# Add src to sys.path for imports
try:
    from src.data_pipeline.data_loader import load_action_programs
    from src.data_pipeline.logo_parser import ComprehensiveNVLabsParser, OneStrokeShape, ensure_all_strings
    from src.Derive_labels.features import extract_topological_features, extract_multiscale_features, extract_relational_features, extract_problem_level_features
    from src.Derive_labels.compositional_features import _calculate_composition_features
    from src.Derive_labels.contextual_features import positive_negative_contrast_score, label_consistency_ratio
    from src.Derive_labels.context_features import compute_discriminative_features
    from src.Derive_labels.validation import validate_features
    from src.Derive_labels.emergence import ConceptMemoryBank
    from src.Derive_labels.scene_graph import SceneGraphFormatter
    from src.Derive_labels.tsv_validation import TSVValidator
except Exception as e:
    print(f"[IMPORT ERROR] Could not import required modules: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

def split_support_holdout(examples):
    """
    Split 7 examples into 6 support and 1 hold-out.
    Returns: support_examples, holdout_example
    """
    return examples[:6], examples[6]

def assess_problem_level_concept(problem_id, positive_examples, negative_examples, logger, context_memory):
    """
    Assess the concept at the problem level using Derive_labels modules for symbolic, compositional, and contextual features.
    Returns: dict with induced concept, hold-out validation, and logs.
    """
    if len(positive_examples) != 7 or len(negative_examples) != 7:
        logger.warning(f"Problem {problem_id} does not have 7+7 examples, skipping.")
        return None
    support_pos, holdout_pos = positive_examples[:6], positive_examples[6]
    support_neg, holdout_neg = negative_examples[:6], negative_examples[6]

    # Feature extraction for support sets with context memory
    support_pos_features = [extract_topological_features(ex, context_memory=context_memory) for ex in support_pos]
    support_neg_features = [extract_topological_features(ex, context_memory=context_memory) for ex in support_neg]
    support_pos_compositional = [_calculate_composition_features(ex, context=context_memory) for ex in support_pos]
    support_neg_compositional = [_calculate_composition_features(ex, context=context_memory) for ex in support_neg]

    # Phase 2: Context-Dependent Perception
    from src.Derive_labels.contextual_features import contextual_concept_hypotheses
    support_pos_feats = [torch.tensor(list(feat.values()), dtype=torch.float) for feat in support_pos_features]
    support_neg_feats = [torch.tensor(list(feat.values()), dtype=torch.float) for feat in support_neg_features]
    query_feat = torch.tensor(list(support_pos_features[0].values()), dtype=torch.float)
    context_hypotheses = contextual_concept_hypotheses(support_pos_feats, support_neg_feats, query_feat)
    logger.info(f"[{problem_id}] Context hypotheses: {context_hypotheses.tolist()}")

    # Discriminative features
    discriminative = compute_discriminative_features(support_pos_features, support_neg_features)
    induced = extract_problem_level_features(support_pos, support_neg)
    # Combine: emergent + compositional + contextual
    combined_concepts = {
      'emergent': induced,
      'compositional': support_pos_compositional,
      'contextual': context_hypotheses.tolist()
    }

    # Phase 3: Meta-Learning Episode
    from src.Derive_labels.meta_learning import MAML, MetaLearnerWrapper, construct_episode
    from src.Derive_labels.models import FeatureExtractorDNN, AdaptiveClassifierDNN
    input_dim = support_pos_feats[0].shape[0] if support_pos_feats else 128
    base_feature_net = FeatureExtractorDNN(input_dim=input_dim, hidden_dims=[256,128])
    base_classifier = AdaptiveClassifierDNN(input_dim=128, output_dim=1)
    meta_model = MetaLearnerWrapper(base_feature_net, base_classifier)
    maml = MAML(meta_model, inner_lr=0.01, outer_lr=0.001, inner_steps=1)

    # Prepare support and query examples for meta-learning
    support_feats = support_pos_feats + support_neg_feats
    support_labels = [1]*len(support_pos_feats) + [0]*len(support_neg_feats)
    query_feats = [torch.tensor(list(extract_topological_features(holdout_pos, context_memory).values()), dtype=torch.float)]
    query_labels = [1]

    # Construct and run meta-learning
    sup_x, sup_y, qry_x, qry_y = construct_episode(support_feats, support_labels, query_feats, query_labels)
    fast_weights = maml.inner_update(sup_x, sup_y)
    meta_loss = maml.outer_update(qry_x, qry_y, fast_weights)
    logger.info(f"[{problem_id}] Meta-learning loss: {meta_loss:.4f}")

    # Generate final adapted prediction
    final_logit = meta_model(qry_x, params=fast_weights)
    final_prob = torch.sigmoid(final_logit).item()
    combined_concepts['meta_prob'] = final_prob

    # Contextual statistics
    pos_contrast = positive_negative_contrast_score(
        [f.get('stroke_diversity', 0) for f in support_pos_compositional],
        [f.get('stroke_diversity', 0) for f in support_neg_compositional]
    )
    label_consistency = label_consistency_ratio(
        [f.get('dominant_stroke_type', '') for f in support_pos_compositional + support_neg_compositional]
    )

    # Hold-out validation
    holdout_results = []
    for ex, label in zip([holdout_pos, holdout_neg], ['positive', 'negative']):
        feats = extract_topological_features(ex, context_memory=context_memory)
        comp_feats = _calculate_composition_features(ex, context=context_memory)
        valid = validate_features(feats)
        match = any(v for k, v in feats.items() if k in induced and induced[k])
        logger.info(f"[Hold-out] {problem_id} [{label}]: features={feats}, comp={comp_feats}, induced={induced}, match={match}, valid={valid}")
        holdout_results.append({
            'problem_id': problem_id,
            'label': label,
            'features': feats,
            'compositional': comp_feats,
            'induced_concept': induced,
            'match': match,
            'validation': valid
        })
    return {
        'problem_id': problem_id,
        'induced_concept': induced,
        'discriminative_features': discriminative,
        'contrast_score': pos_contrast,
        'label_consistency': label_consistency,
        'holdout_results': holdout_results,
        'combined_concepts': combined_concepts
    }
def main():
    parser = argparse.ArgumentParser(description="Extract derived labels for Bongard-LOGO problems.")
    parser.add_argument('--input-dir', required=True, help='Input directory containing ShapeBongard_V2')
    parser.add_argument('--output', required=True, help='Output JSON file for derived labels')
    parser.add_argument('--problems-list', required=True, help='File listing problem IDs to process')
    parser.add_argument('--tsv-file', required=True, help='TSV file for validation')
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("logo_to_shape")

    # Initialize concept memory
    ConceptMemoryBank.initialize()
    context_memory = ConceptMemoryBank.load()
    from src.Derive_labels.scene_graph import SceneGraphFormatter
    from src.Derive_labels.tsv_validation import TSVValidator
    sg_formatter = SceneGraphFormatter()
    validator = TSVValidator(tsv_path=args.tsv_file)

    logger.info(f"Loading action programs from {args.input_dir}")
    action_programs = load_action_programs(args.input_dir)
    with open(args.problems_list, 'r') as f:
        problem_ids = [line.strip() for line in f if line.strip()]

    derived_records = []
    for problem_id in problem_ids:
        if problem_id not in action_programs:
            logger.warning(f"No valid action program for problem_id: {problem_id}")
            continue
        positive_examples, negative_examples = action_programs[problem_id]
        # Emergent concepts
        support_pos_features = [extract_topological_features(ex, context_memory=context_memory) for ex in positive_examples[:6]]
        support_neg_features = [extract_topological_features(ex, context_memory=context_memory) for ex in negative_examples[:6]]
        emergent = support_pos_features + support_neg_features
        # Contextual hypotheses
        support_pos_feats = [torch.tensor(list(feat.values()), dtype=torch.float) for feat in support_pos_features]
        support_neg_feats = [torch.tensor(list(feat.values()), dtype=torch.float) for feat in support_neg_features]
        query_feat = torch.tensor(list(support_pos_features[0].values()), dtype=torch.float)
        from src.Derive_labels.contextual_features import contextual_concept_hypotheses
        context_hyps = contextual_concept_hypotheses(support_pos_feats, support_neg_feats, query_feat)
        # Meta-learning
        from src.Derive_labels.meta_learning import MAML, MetaLearnerWrapper, construct_episode
        from src.Derive_labels.models import FeatureExtractorDNN, AdaptiveClassifierDNN
        input_dim = support_pos_feats[0].shape[0] if support_pos_feats else 128
        base_feature_net = FeatureExtractorDNN(input_dim=input_dim, hidden_dims=[256,128])
        base_classifier = AdaptiveClassifierDNN(input_dim=128, output_dim=1)
        meta_model = MetaLearnerWrapper(base_feature_net, base_classifier)
        maml = MAML(meta_model, inner_lr=0.01, outer_lr=0.001, inner_steps=1)
        support_feats = support_pos_feats + support_neg_feats
        support_labels = [1]*len(support_pos_feats) + [0]*len(support_neg_feats)
        query_feats = [query_feat]
        query_labels = [1]
        sup_x, sup_y, qry_x, qry_y = construct_episode(support_feats, support_labels, query_feats, query_labels)
        fast_weights = maml.inner_update(sup_x, sup_y)
        maml.outer_update(qry_x, qry_y, fast_weights)
        meta_prob = float(torch.sigmoid(meta_model(qry_x, params=fast_weights)))
        # Compositional features
        comp_feats = [_calculate_composition_features(ex, context={'problem_id':problem_id}) for ex in positive_examples[:6]]
        # Combine all
        record = {
            'problem_id': problem_id,
            'emergent_concepts': {f'pos_{i}': v for i,v in enumerate(support_pos_features)} | {f'neg_{i}': v for i,v in enumerate(support_neg_features)},
            'contextual_hypotheses': context_hyps.tolist() if hasattr(context_hyps, 'tolist') else context_hyps,
            'compositional': {f'comp_{i}': v for i,v in enumerate(comp_feats)},
            'meta_prob': meta_prob
        }
        # Scene graph formatting
        record['scene_graph'] = sg_formatter.format(record)
        # TSV validation
        record['validation'] = validator.validate(problem_id, record)
        derived_records.append(record)

    logger.info(f"Writing derived labels to {args.output}")
    with open(args.output, 'w') as f:
        json.dump(derived_records, f, indent=2)
    logger.info("Finished writing derived labels.")

if __name__ == "__main__":
    main()