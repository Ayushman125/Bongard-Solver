from .geometric_detectors import StackingEnsemble, DetectorAttention, detect_shapes
from .confidence_scoring import TemperatureScaler, PlattScaler, IsotonicScaler, score_confidence_advanced, initialize_confidence_calibrator
from .logo_processing import ContextEncoder
from .mcts_labeling import mc_dropout_predict
from .post_processing import crodlab_consensus, compute_kappa
from .validation import validate_batch, select_uncertain_samples, self_training, co_training

def derive_labels(batch, config, epoch):
    # 1. Feature extraction + detection
    labels = []
    for img in batch.images:
        label = StackingEnsemble(config['detectors']).predict(img)
        labels.append(label)

    # 2. Confidence calibration
    logits = batch.logits
    temp_probs = TemperatureScaler().calibrate(logits)
    if config.get('use_platt'):
        platt = PlattScaler(); platt.fit(logits, batch.true_labels)
        final_conf = platt.calibrate(logits)
    elif config.get('use_isotonic'):
        iso = IsotonicScaler(); iso.fit(temp_probs.max(axis=1), batch.true_labels)
        final_conf = iso.calibrate(temp_probs.max(axis=1))
    else:
        final_conf = temp_probs

    # 3. Context encoding (if Bongard)
    ctx_feats = ContextEncoder(config['feat_dim'], config['hidden_dim'])(batch.features)

    # 4. Uncertainty estimation
    mean_p, unc = mc_dropout_predict(config['model'], batch.tensors)
    review = validate_batch(labels, unc, epoch, config)

    # 5. Consensus & agreement
    if config.get('use_crowdlab'):
        consensus = crodlab_consensus([labels, batch.other_detector_labels])
    else:
        from .post_processing import dawid_skene
        consensus = dawid_skene([labels, batch.other_detector_labels])
    kappa, fleiss = compute_kappa([labels, batch.other_detector_labels])

    # 6. Active learning selection
    to_review = select_uncertain_samples(dict(zip(batch.ids, unc)), config['review_budget'])

    # 7. Semi-supervised update
    if epoch % config.get('self_train_freq', 5) == 0:
        self_training(config['model'], config['unlabeled_loader'], config.get('self_train_thresh', 0.95))
        co_training(config['model'], config['aux_model'], config['unlabeled_loader'])

    return consensus, final_conf, to_review
