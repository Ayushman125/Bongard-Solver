import numpy as np
import torch
from typing import List, Dict, Tuple, Set, Optional, Any
from collections import defaultdict, Counter
import logging
from dataclasses import dataclass
import json

@dataclass
class SceneGraphMetrics:
    recall_at_k: Dict[int, float]
    mean_recall_at_k: Dict[int, float]
    no_graph_constraint_recall: Dict[int, float]
    predicate_recall: Dict[str, Dict[int, float]]
    overall_statistics: Dict[str, Any]

class RecallAtKEvaluator:
    def __init__(self, 
                 predicate_classes: List[str],
                 k_values: List[int] = [20, 50, 100],
                 iou_threshold: float = 0.5):
        self.predicate_classes = predicate_classes
        self.k_values = k_values
        self.iou_threshold = iou_threshold
        self.predicate_frequency = Counter()
        self.evaluation_history = []
        logging.info(f"RecallAtKEvaluator initialized with {len(predicate_classes)} predicates")
    def evaluate_scene_graph(self,
                            predicted_triplets: List[Dict],
                            ground_truth_triplets: List[Dict],
                            predicted_objects: List[Dict],
                            ground_truth_objects: List[Dict]) -> SceneGraphMetrics:
        object_matching = self._match_objects(predicted_objects, ground_truth_objects)
        matched_gt_triplets = self._filter_triplets_by_matching(
            ground_truth_triplets, object_matching, is_ground_truth=True
        )
        matched_pred_triplets = self._filter_triplets_by_matching(
            predicted_triplets, object_matching, is_ground_truth=False
        )
        recall_at_k = self._calculate_recall_at_k(matched_pred_triplets, matched_gt_triplets)
        mean_recall_at_k = self._calculate_mean_recall_at_k(matched_pred_triplets, matched_gt_triplets)
        ng_recall_at_k = self._calculate_no_graph_constraint_recall(matched_pred_triplets, matched_gt_triplets)
        predicate_recall = self._calculate_predicate_specific_recall(matched_pred_triplets, matched_gt_triplets)
        overall_stats = self._calculate_overall_statistics(
            matched_pred_triplets, matched_gt_triplets, object_matching
        )
        return SceneGraphMetrics(
            recall_at_k=recall_at_k,
            mean_recall_at_k=mean_recall_at_k,
            no_graph_constraint_recall=ng_recall_at_k,
            predicate_recall=predicate_recall,
            overall_statistics=overall_stats
        )
    def _match_objects(self, 
                      predicted_objects: List[Dict], 
                      ground_truth_objects: List[Dict]) -> Dict[str, str]:
        matching = {}
        used_gt_objects = set()
        iou_matrix = np.zeros((len(predicted_objects), len(ground_truth_objects)))
        for i, pred_obj in enumerate(predicted_objects):
            for j, gt_obj in enumerate(ground_truth_objects):
                iou = self._calculate_bbox_iou(
                    pred_obj.get('bbox', [0, 0, 1, 1]),
                    gt_obj.get('bbox', [0, 0, 1, 1])
                )
                iou_matrix[i, j] = iou
        while True:
            max_iou_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            max_iou = iou_matrix[max_iou_idx]
            if max_iou < self.iou_threshold:
                break
            pred_idx, gt_idx = max_iou_idx
            if pred_idx in matching or gt_idx in used_gt_objects:
                iou_matrix[pred_idx, :] = -1
                iou_matrix[:, gt_idx] = -1
                continue
            matching[predicted_objects[pred_idx]['id']] = ground_truth_objects[gt_idx]['id']
            used_gt_objects.add(gt_idx)
            iou_matrix[pred_idx, :] = -1
            iou_matrix[:, gt_idx] = -1
        return matching
    def _calculate_bbox_iou(self, bbox1, bbox2) -> float:
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        x_left = max(x1_min, x2_min)
        y_top = max(y1_min, y2_min)
        x_right = min(x1_max, x2_max)
        y_bottom = min(y1_max, y2_max)
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0
    def _filter_triplets_by_matching(self,
                                   triplets: List[Dict],
                                   object_matching: Dict[str, str],
                                   is_ground_truth: bool) -> List[Dict]:
        filtered_triplets = []
        for triplet in triplets:
            subject_id = triplet.get('subject_id')
            object_id = triplet.get('object_id')
            if is_ground_truth:
                if subject_id and object_id:
                    filtered_triplets.append(triplet)
            else:
                if (subject_id and object_id and 
                    subject_id in object_matching and 
                    object_id in object_matching):
                    updated_triplet = triplet.copy()
                    updated_triplet['subject_id'] = object_matching[subject_id]
                    updated_triplet['object_id'] = object_matching[object_id]
                    filtered_triplets.append(updated_triplet)
        return filtered_triplets
    def _calculate_recall_at_k(self,
                             predicted_triplets: List[Dict],
                             ground_truth_triplets: List[Dict]) -> Dict[int, float]:
        if not ground_truth_triplets:
            return {k: 0.0 for k in self.k_values}
        gt_set = set()
        for triplet in ground_truth_triplets:
            gt_set.add((
                triplet['subject_id'],
                triplet['predicate'],
                triplet['object_id']
            ))
        sorted_predictions = sorted(
            predicted_triplets,
            key=lambda x: x.get('confidence', 1.0),
            reverse=True
        )
        recall_at_k = {}
        for k in self.k_values:
            top_k_predictions = sorted_predictions[:k]
            matches = 0
            for pred in top_k_predictions:
                pred_triplet = (
                    pred['subject_id'],
                    pred['predicate'],
                    pred['object_id']
                )
                if pred_triplet in gt_set:
                    matches += 1
            recall_at_k[k] = matches / len(gt_set)
        return recall_at_k
    def _calculate_mean_recall_at_k(self,
                                  predicted_triplets: List[Dict],
                                  ground_truth_triplets: List[Dict]) -> Dict[int, float]:
        gt_by_predicate = defaultdict(list)
        for triplet in ground_truth_triplets:
            predicate = triplet['predicate']
            gt_by_predicate[predicate].append(triplet)
        pred_by_predicate = defaultdict(list)
        for triplet in predicted_triplets:
            predicate = triplet['predicate']
            pred_by_predicate[predicate].append(triplet)
        mean_recall_at_k = {}
        for k in self.k_values:
            predicate_recalls = []
            for predicate in gt_by_predicate.keys():
                gt_triplets = gt_by_predicate[predicate]
                pred_triplets = pred_by_predicate.get(predicate, [])
                if not gt_triplets:
                    continue
                gt_set = set()
                for triplet in gt_triplets:
                    gt_set.add((
                        triplet['subject_id'],
                        triplet['predicate'],
                        triplet['object_id']
                    ))
                sorted_pred = sorted(
                    pred_triplets,
                    key=lambda x: x.get('confidence', 1.0),
                    reverse=True
                )
                top_k_pred = sorted_pred[:k]
                matches = 0
                for pred in top_k_pred:
                    pred_triplet = (
                        pred['subject_id'],
                        pred['predicate'],
                        pred['object_id']
                    )
                    if pred_triplet in gt_set:
                        matches += 1
                predicate_recall = matches / len(gt_set)
                predicate_recalls.append(predicate_recall)
            mean_recall_at_k[k] = np.mean(predicate_recalls) if predicate_recalls else 0.0
        return mean_recall_at_k
    def _calculate_no_graph_constraint_recall(self,
                                            predicted_triplets: List[Dict],
                                            ground_truth_triplets: List[Dict]) -> Dict[int, float]:
        if not ground_truth_triplets:
            return {k: 0.0 for k in self.k_values}
        gt_by_pair = defaultdict(set)
        for triplet in ground_truth_triplets:
            pair = (triplet['subject_id'], triplet['object_id'])
            gt_by_pair[pair].add(triplet['predicate'])
        pred_by_pair = defaultdict(list)
        for triplet in predicted_triplets:
            pair = (triplet['subject_id'], triplet['object_id'])
            pred_by_pair[pair].append(triplet)
        ng_recall_at_k = {}
        for k in self.k_values:
            total_gt_relations = sum(len(predicates) for predicates in gt_by_pair.values())
            if total_gt_relations == 0:
                ng_recall_at_k[k] = 0.0
                continue
            matched_relations = 0
            for pair, gt_predicates in gt_by_pair.items():
                if pair not in pred_by_pair:
                    continue
                pair_predictions = sorted(
                    pred_by_pair[pair],
                    key=lambda x: x.get('confidence', 1.0),
                    reverse=True
                )
                top_k_pair_pred = pair_predictions[:k]
                for pred in top_k_pair_pred:
                    if pred['predicate'] in gt_predicates:
                        matched_relations += 1
            ng_recall_at_k[k] = matched_relations / total_gt_relations
        return ng_recall_at_k
    def _calculate_predicate_specific_recall(self,
                                           predicted_triplets: List[Dict],
                                           ground_truth_triplets: List[Dict]) -> Dict[str, Dict[int, float]]:
        predicate_recall = {}
        gt_by_predicate = defaultdict(list)
        pred_by_predicate = defaultdict(list)
        for triplet in ground_truth_triplets:
            gt_by_predicate[triplet['predicate']].append(triplet)
        for triplet in predicted_triplets:
            pred_by_predicate[triplet['predicate']].append(triplet)
        for predicate in self.predicate_classes:
            gt_triplets = gt_by_predicate[predicate]
            pred_triplets = pred_by_predicate[predicate]
            if not gt_triplets:
                predicate_recall[predicate] = {k: 0.0 for k in self.k_values}
                continue
            pred_recall = self._calculate_recall_at_k(pred_triplets, gt_triplets)
            predicate_recall[predicate] = pred_recall
        return predicate_recall
    def _calculate_overall_statistics(self,
                                    predicted_triplets: List[Dict],
                                    ground_truth_triplets: List[Dict],
                                    object_matching: Dict[str, str]) -> Dict[str, Any]:
        for triplet in ground_truth_triplets:
            self.predicate_frequency[triplet['predicate']] += 1
        stats = {
            'num_ground_truth_triplets': len(ground_truth_triplets),
            'num_predicted_triplets': len(predicted_triplets),
            'num_matched_objects': len(object_matching),
            'predicate_distribution': dict(self.predicate_frequency),
            'unique_predicates_gt': len(set(t['predicate'] for t in ground_truth_triplets)),
            'unique_predicates_pred': len(set(t['predicate'] for t in predicted_triplets)),
            'average_confidence': np.mean([
                t.get('confidence', 1.0) for t in predicted_triplets
            ]) if predicted_triplets else 0.0
        }
        return stats
    def aggregate_metrics(self, individual_metrics: List[SceneGraphMetrics]) -> SceneGraphMetrics:
        if not individual_metrics:
            return SceneGraphMetrics(
                recall_at_k={k: 0.0 for k in self.k_values},
                mean_recall_at_k={k: 0.0 for k in self.k_values},
                no_graph_constraint_recall={k: 0.0 for k in self.k_values},
                predicate_recall={},
                overall_statistics={}
            )
        agg_recall_at_k = {}
        for k in self.k_values:
            k_values = [m.recall_at_k[k] for m in individual_metrics]
            agg_recall_at_k[k] = np.mean(k_values)
        agg_mean_recall_at_k = {}
        for k in self.k_values:
            k_values = [m.mean_recall_at_k[k] for m in individual_metrics]
            agg_mean_recall_at_k[k] = np.mean(k_values)
        agg_ng_recall = {}
        for k in self.k_values:
            k_values = [m.no_graph_constraint_recall[k] for m in individual_metrics]
            agg_ng_recall[k] = np.mean(k_values)
        all_predicates = set()
        for m in individual_metrics:
            all_predicates.update(m.predicate_recall.keys())
        agg_predicate_recall = {}
        for predicate in all_predicates:
            agg_predicate_recall[predicate] = {}
            for k in self.k_values:
                k_values = [
                    m.predicate_recall.get(predicate, {}).get(k, 0.0) 
                    for m in individual_metrics
                ]
                agg_predicate_recall[predicate][k] = np.mean(k_values)
        agg_stats = {
            'total_evaluations': len(individual_metrics),
            'average_gt_triplets': np.mean([
                m.overall_statistics['num_ground_truth_triplets'] 
                for m in individual_metrics
            ]),
            'average_pred_triplets': np.mean([
                m.overall_statistics['num_predicted_triplets'] 
                for m in individual_metrics
            ]),
            'total_predicate_frequency': dict(self.predicate_frequency)
        }
        return SceneGraphMetrics(
            recall_at_k=agg_recall_at_k,
            mean_recall_at_k=agg_mean_recall_at_k,
            no_graph_constraint_recall=agg_ng_recall,
            predicate_recall=agg_predicate_recall,
            overall_statistics=agg_stats
        )
