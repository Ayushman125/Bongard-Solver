import torch
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
    from src.Derive_labels.contextual_features import positive_negative_contrast_score, label_consistency_ratio, contextual_concept_hypotheses, discriminative_concepts
    ## compute_discriminative_features import removed; use embedding-based differentiation instead
    from src.Derive_labels.validation import validate_features
    from src.Derive_labels.emergence import ConceptMemoryBank
    from src.Derive_labels.scene_graph import SceneGraphFormatter
    from src.Derive_labels.tsv_validation import TSVValidator
    from src.Derive_labels.stroke_types import _calculate_stroke_type_differentiated_features
except Exception as e:
    print(f"[IMPORT ERROR] Could not import required modules: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Device selection for CUDA support
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    from src.Derive_labels.contextual_features import contextual_concept_hypotheses, discriminative_concepts
    support_pos_feats = [torch.tensor(list(feat.values()), dtype=torch.float) for feat in support_pos_features]
    support_neg_feats = [torch.tensor(list(feat.values()), dtype=torch.float) for feat in support_neg_features]
    query_feat = torch.tensor(list(support_pos_features[0].values()), dtype=torch.float)
    context_hypotheses = contextual_concept_hypotheses(support_pos_feats, support_neg_feats, query_feat)
    logger.info(f"[{problem_id}] Context hypotheses: {context_hypotheses.tolist()}")

    # Discriminative features
    # Use embedding-based differentiation for support sets
    # Ensure context is defined (example: can be empty or constructed from support sets)
    context = context if 'context' in locals() else {}
    pos_diff = _calculate_stroke_type_differentiated_features(support_pos_features, context)
    neg_diff = _calculate_stroke_type_differentiated_features(support_neg_features, context)
    # For cross-set discrimination, compare pos_diff and neg_diff embeddings/statistics as needed
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
    base_feature_net = FeatureExtractorDNN(input_dim=input_dim, hidden_dims=[256,128]).to(DEVICE)
    base_classifier = AdaptiveClassifierDNN(input_dim=128, output_dim=1).to(DEVICE)
    meta_model = MetaLearnerWrapper(base_feature_net, base_classifier)
    maml = MAML(meta_model, inner_lr=0.01, outer_lr=0.001, inner_steps=1)

    # Prepare support and query examples for meta-learning
    support_feats = support_pos_feats + support_neg_feats
    support_labels = [1]*len(support_pos_feats) + [0]*len(support_neg_feats)
    query_feats = [torch.tensor(list(extract_topological_features(holdout_pos, context_memory).values()), dtype=torch.float)]
    query_labels = [1]

    # Move tensors to device
    support_feats = [f.to(DEVICE) for f in support_feats]
    query_feats = [f.to(DEVICE) for f in query_feats]

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
    # 'discriminative_features': discriminative,  # Removed legacy reference
    'pos_differentiation': pos_diff,
    'neg_differentiation': neg_diff,
        'contrast_score': pos_contrast,
        'label_consistency': label_consistency,
        'holdout_results': holdout_results,
        'combined_concepts': combined_concepts
    }
def main():
    # Import ConceptEmbedder
    from src.Derive_labels.concept_embeddings import ConceptEmbedder
    concept_embed_dim = 512
    embedder = ConceptEmbedder(vocab_size=1000, emb_dim=concept_embed_dim)
    parser = argparse.ArgumentParser(description="Extract derived labels for Bongard-LOGO problems.")
    parser.add_argument('--input-dir', required=True, help='Input directory containing ShapeBongard_V2')
    parser.add_argument('--output', required=True, help='Output JSON file for derived labels')
    parser.add_argument('--problems-list', required=True, help='File listing problem IDs to process')
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("logo_to_shape")
    logger.info(f"Using device: {DEVICE}")

    # Initialize concept memory
    ConceptMemoryBank.initialize()
    context_memory = ConceptMemoryBank.load()
    from src.Derive_labels.scene_graph import SceneGraphFormatter
    from src.Derive_labels.tsv_validation import TSVValidator
    sg_formatter = SceneGraphFormatter()
    # Automate TSV file detection
    import glob
    tsv_files = glob.glob(os.path.join('data', '*.tsv'))
    validator1 = TSVValidator(tsv_path=tsv_files[0]) if len(tsv_files) > 0 else None
    validator2 = TSVValidator(tsv_path=tsv_files[1]) if len(tsv_files) > 1 else None

    logger.info(f"Loading action programs from {args.input_dir}")
    action_programs = load_action_programs(args.input_dir)
    with open(args.problems_list, 'r') as f:
        problem_ids = [line.strip() for line in f if line.strip()]

    derived_records = []
    def to_tensor(feat):
        logger.info(f"[to_tensor] Input: {repr(feat)} (type: {type(feat)})")
        import numbers
        if isinstance(feat, dict):
            vals = list(feat.values())
            logger.info(f"[to_tensor] Dict values: {vals}")
            logger.info(f"[to_tensor] Dict value types: {[type(v) for v in vals]}")
            return torch.tensor([v for v in vals if isinstance(v, numbers.Number)], dtype=torch.float)
        elif isinstance(feat, list):
            logger.info(f"[to_tensor] List length: {len(feat)}")
            logger.info(f"[to_tensor] List element types: {[type(f) for f in feat]}")
            # If feat is a list of dicts, flatten to numeric values
            if all(isinstance(f, dict) for f in feat):
                vals = []
                for f in feat:
                    vlist = list(f.values())
                    logger.info(f"[to_tensor] List[Dict] values: {vlist}")
                    logger.info(f"[to_tensor] List[Dict] value types: {[type(v) for v in vlist]}")
                    vals.extend([v for v in vlist if isinstance(v, numbers.Number)])
                logger.info(f"[to_tensor] Flattened numeric values: {vals}")
                return torch.tensor(vals, dtype=torch.float)
            else:
                logger.info(f"[to_tensor] List values: {feat}")
                logger.info(f"[to_tensor] List value types: {[type(v) for v in feat]}")
                non_numeric = [v for v in feat if not isinstance(v, numbers.Number)]
                if non_numeric:
                    logger.warning(f"[to_tensor] Non-numeric values found in list: {non_numeric}")
                return torch.tensor([v for v in feat if isinstance(v, numbers.Number)], dtype=torch.float)
        else:
            logger.error(f"[to_tensor] Invalid type: {type(feat)}")
            raise TypeError(f"Feature must be dict or list, got {type(feat)}")
    for problem_id in problem_ids:
        if problem_id not in action_programs:
            logger.warning(f"No valid action program for problem_id: {problem_id}")
            continue
        positive_examples, negative_examples = action_programs[problem_id]
        # Emergent concepts
        support_pos_features = [extract_topological_features(ex, context_memory=context_memory) for ex in positive_examples[:6]]
        support_neg_features = [extract_topological_features(ex, context_memory=context_memory) for ex in negative_examples[:6]]
        emergent = support_pos_features + support_neg_features

        # Gather all concept tokens for embedding (for this batch)
        all_tokens = []
        def extract_tokens(feat):
            if isinstance(feat, list):
                return [t for t in feat if isinstance(t, str)]
            elif isinstance(feat, str):
                return [feat]
            return []
        for feat in support_pos_features + support_neg_features:
            all_tokens.extend(extract_tokens(feat))
        embedder.register_tokens(all_tokens)

        # Get concept embeddings for each support example (only registered tokens)
        support_pos_embeds = [embedder(extract_tokens(feat)) for feat in support_pos_features]
        support_neg_embeds = [embedder(extract_tokens(feat)) for feat in support_neg_features]
        def numeric_tensor(feat):
            import numbers
            if isinstance(feat, dict):
                return torch.tensor([v for v in feat.values() if isinstance(v, numbers.Number)], dtype=torch.float)
            elif isinstance(feat, list):
                vals = [v for v in feat if isinstance(v, numbers.Number)]
                return torch.tensor(vals, dtype=torch.float) if vals else None
            return None
        support_pos_feats = []
        for feat, emb in zip(support_pos_features, support_pos_embeds):
            num_tensor = numeric_tensor(feat)
            if num_tensor is not None and num_tensor.numel() > 0:
                support_pos_feats.append(torch.cat([num_tensor, emb.squeeze(0)], dim=-1))
            else:
                support_pos_feats.append(emb.squeeze(0))
        support_neg_feats = []
        for feat, emb in zip(support_neg_features, support_neg_embeds):
            num_tensor = numeric_tensor(feat)
            if num_tensor is not None and num_tensor.numel() > 0:
                support_neg_feats.append(torch.cat([num_tensor, emb.squeeze(0)], dim=-1))
            else:
                support_neg_feats.append(emb.squeeze(0))
        # For query_feat, use the same logic
        num_tensor = numeric_tensor(support_pos_features[0])
        if num_tensor is not None and num_tensor.numel() > 0:
            query_feat = torch.cat([num_tensor, support_pos_embeds[0].squeeze(0)], dim=-1)
        else:
            query_feat = support_pos_embeds[0].squeeze(0)

        # Aggregate multi-token embeddings to a single vector before attention
        def aggregate_feat(f):
            if isinstance(f, torch.Tensor) and f.dim() == 2 and f.shape[1] == 512:
                return f.mean(dim=0)
            return f
        support_pos_feats = [aggregate_feat(f) for f in support_pos_feats]
        support_neg_feats = [aggregate_feat(f) for f in support_neg_feats]
        query_feat = aggregate_feat(query_feat)

        # Move tensors to device
        support_pos_feats = [f.to(DEVICE) for f in support_pos_feats]
        support_neg_feats = [f.to(DEVICE) for f in support_neg_feats]
        query_feat = query_feat.to(DEVICE)

        context_hyps = contextual_concept_hypotheses(support_pos_feats, support_neg_feats, query_feat)
        # Bongard-style concept reasoning: find discriminative concepts
        # Use raw concept token lists (not tensors/embeddings)
        disc_concepts = discriminative_concepts(
            [extract_tokens(feat) for feat in support_pos_features],
            [extract_tokens(feat) for feat in support_neg_features]
        )
        logger.info(f"[Bongard-style] Discriminative concepts (present in positive, absent in negative): {disc_concepts}")

        # Meta-learning
        from src.Derive_labels.meta_learning import MAML, MetaLearnerWrapper, construct_episode
        from src.Derive_labels.models import FeatureExtractorDNN, AdaptiveClassifierDNN
        input_dim = support_pos_feats[0].shape[0] if support_pos_feats else 128
        base_feature_net = FeatureExtractorDNN(input_dim=input_dim, hidden_dims=[256,128]).to(DEVICE)
        base_classifier = AdaptiveClassifierDNN(input_dim=128, output_dim=1).to(DEVICE)
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
        # Compositional features: extract stroke primitives for each example
        from src.Derive_labels.stroke_types import extract_modifier_from_stroke
        from src.Derive_labels.stroke_types import _calculate_stroke_specific_features
        comp_feats = []
        for ex in positive_examples[:6]:
            stroke_features = []
            for idx, cmd in enumerate(ex):
                # Optionally pass context and parent_shape_vertices/shape_obj if available
                context_dict = {'problem_id': problem_id}
                # If you have shape_info and it contains vertices, pass as parent_shape_vertices
                # primitive = _calculate_stroke_specific_features(cmd, idx, context=context_dict, parent_shape_vertices=shape_info['vertices'])
                # If you have shape_info and it contains a shape object, pass as shape_obj
                # primitive = _calculate_stroke_specific_features(cmd, idx, context=context_dict, shape_obj=shape_info['shape_obj'])
                # If no shape_info, just pass context
                primitive = _calculate_stroke_specific_features(cmd, idx, context=context_dict)
                stroke_features.append(primitive)
            logger.info(f"[STROKE FEATURE EXTRACTION] Problem {problem_id} Example: {ex}\nExtracted Features: {stroke_features}")
            comp_feat = _calculate_composition_features([cmd for cmd in ex], context={'problem_id': problem_id})
            comp_feat['stroke_features'] = stroke_features
            comp_feats.append(comp_feat)
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
        # TSV validation (dual, automated)
        record['validation'] = {}
        if validator1:
            record['validation']['primary'] = validator1.validate(problem_id, record)
        if validator2:
            record['validation']['secondary'] = validator2.validate(problem_id, record)
        derived_records.append(record)

    logger.info(f"Writing derived labels to {args.output}")
    with open(args.output, 'w') as f:
        json.dump(derived_records, f, indent=2)
    logger.info("Finished writing derived labels.")

if __name__ == "__main__":
    main()