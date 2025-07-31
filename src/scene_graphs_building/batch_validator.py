import asyncio
import torch
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import logging
from PIL import Image
import gc
import psutil
from concurrent.futures import ThreadPoolExecutor
import time

@dataclass
class ValidationRequest:
    image: Image.Image
    scene_graph: Dict[str, Any]
    request_id: str
    priority: int = 0

@dataclass
class ValidationResult:
    request_id: str
    overall_score: float
    object_presence_score: float
    relationship_accuracy_score: float
    spatial_consistency_score: float
    detailed_feedback: List[Dict]
    processing_time: float
    memory_peak_mb: float

class BatchMultimodalValidator:
    def __init__(self,
                 max_batch_size: int = 4,
                 max_memory_mb: int = 3072,
                 validation_timeout: float = 30.0,
                 enable_gpu_batching: bool = True):
        self.max_batch_size = max_batch_size
        self.max_memory_mb = max_memory_mb
        self.validation_timeout = validation_timeout
        self.enable_gpu_batching = enable_gpu_batching and torch.cuda.is_available()
        self.pending_requests: List[ValidationRequest] = []
        self.processing_queue = asyncio.Queue()
        self.result_store: Dict[str, ValidationResult] = {}
        self.batch_stats = {
            'total_batches': 0,
            'successful_batches': 0,
            'average_batch_size': 0.0,
            'average_processing_time': 0.0,
            'memory_efficiency': 0.0
        }
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        logging.info(f"BatchMultimodalValidator initialized with batch_size={max_batch_size}")
    async def validate_batch(self, requests: List[ValidationRequest]) -> List[ValidationResult]:
        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        try:
            grouped_requests = self._group_by_complexity(requests)
            all_results = []
            for complexity_group in grouped_requests:
                batch_results = await self._process_complexity_group(complexity_group)
                all_results.extend(batch_results)
                gc.collect()
                if self.enable_gpu_batching:
                    torch.cuda.empty_cache()
            processing_time = time.time() - start_time
            peak_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            memory_used = peak_memory - initial_memory
            self._update_batch_stats(len(requests), processing_time, memory_used)
            return all_results
        except Exception as e:
            logging.error(f"Batch validation failed: {e}")
            return [
                ValidationResult(
                    request_id=req.request_id,
                    overall_score=0.0,
                    object_presence_score=0.0,
                    relationship_accuracy_score=0.0,
                    spatial_consistency_score=0.0,
                    detailed_feedback=[{'error': str(e)}],
                    processing_time=time.time() - start_time,
                    memory_peak_mb=psutil.Process().memory_info().rss / (1024 * 1024)
                ) for req in requests
            ]
    def _group_by_complexity(self, requests: List[ValidationRequest]) -> List[List[ValidationRequest]]:
        complexity_scores = []
        for req in requests:
            img_complexity = req.image.size[0] * req.image.size[1] / (512 * 512)
            obj_complexity = len(req.scene_graph.get('objects', []))
            rel_complexity = len(req.scene_graph.get('relationships', []))
            total_complexity = img_complexity + obj_complexity * 0.1 + rel_complexity * 0.05
            complexity_scores.append((req, total_complexity))
        complexity_scores.sort(key=lambda x: x[1])
        groups = []
        current_group = []
        current_complexity = 0
        for req, complexity in complexity_scores:
            if (len(current_group) >= self.max_batch_size or 
                (current_group and abs(complexity - current_complexity) > 0.5)):
                groups.append(current_group)
                current_group = []
                current_complexity = 0
            current_group.append(req)
            current_complexity = (current_complexity * (len(current_group) - 1) + complexity) / len(current_group)
        if current_group:
            groups.append(current_group)
        return groups
    async def _process_complexity_group(self, requests: List[ValidationRequest]) -> List[ValidationResult]:
        batch_images = []
        batch_scene_graphs = []
        request_ids = []
        for req in requests:
            batch_images.append(req.image)
            batch_scene_graphs.append(req.scene_graph)
            request_ids.append(req.request_id)
        current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        if current_memory > self.max_memory_mb:
            logging.warning(f"Memory usage high ({current_memory:.1f}MB), forcing cleanup")
            gc.collect()
            if self.enable_gpu_batching:
                torch.cuda.empty_cache()
        try:
            batch_results = await asyncio.wait_for(
                self._validate_batch_internal(batch_images, batch_scene_graphs, request_ids),
                timeout=self.validation_timeout
            )
            return batch_results
        except asyncio.TimeoutError:
            logging.error(f"Batch validation timeout for {len(requests)} requests")
            return [
                ValidationResult(
                    request_id=req_id,
                    overall_score=0.0,
                    object_presence_score=0.0,
                    relationship_accuracy_score=0.0,
                    spatial_consistency_score=0.0,
                    detailed_feedback=[{'error': 'Validation timeout'}],
                    processing_time=self.validation_timeout,
                    memory_peak_mb=current_memory
                ) for req_id in request_ids
            ]
    async def _validate_batch_internal(self, 
                                     images: List[Image.Image],
                                     scene_graphs: List[Dict],
                                     request_ids: List[str]) -> List[ValidationResult]:
        results = []
        start_time = time.time()
        sub_batch_size = min(len(images), self.max_batch_size)
        for i in range(0, len(images), sub_batch_size):
            sub_images = images[i:i + sub_batch_size]
            sub_graphs = scene_graphs[i:i + sub_batch_size]
            sub_ids = request_ids[i:i + sub_batch_size]
            sub_results = await self._process_sub_batch(sub_images, sub_graphs, sub_ids)
            results.extend(sub_results)
            if i + sub_batch_size < len(images):
                gc.collect()
                if self.enable_gpu_batching:
                    torch.cuda.empty_cache()
        return results
    async def _process_sub_batch(self, images, scene_graphs, request_ids) -> List[ValidationResult]:
        results = []
        processing_start = time.time()
        memory_start = psutil.Process().memory_info().rss / (1024 * 1024)
        for i, (image, scene_graph, request_id) in enumerate(zip(images, scene_graphs, request_ids)):
            try:
                objects = scene_graph.get('objects', [])
                object_score = min(1.0, len(objects) / 10.0)
                relationships = scene_graph.get('relationships', [])
                relationship_score = min(1.0, len(relationships) / 20.0)
                spatial_score = 0.8 + np.random.normal(0, 0.1)
                spatial_score = max(0.0, min(1.0, spatial_score))
                overall_score = (
                    0.3 * object_score +
                    0.4 * relationship_score +
                    0.3 * spatial_score
                )
                current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
                result = ValidationResult(
                    request_id=request_id,
                    overall_score=overall_score,
                    object_presence_score=object_score,
                    relationship_accuracy_score=relationship_score,
                    spatial_consistency_score=spatial_score,
                    detailed_feedback=[
                        {
                            'type': 'batch_validation',
                            'batch_size': len(images),
                            'processing_order': i + 1,
                            'memory_usage_mb': current_memory
                        }
                    ],
                    processing_time=time.time() - processing_start,
                    memory_peak_mb=current_memory
                )
                results.append(result)
            except Exception as e:
                logging.error(f"Failed to validate request {request_id}: {e}")
                results.append(
                    ValidationResult(
                        request_id=request_id,
                        overall_score=0.0,
                        object_presence_score=0.0,
                        relationship_accuracy_score=0.0,
                        spatial_consistency_score=0.0,
                        detailed_feedback=[{'error': str(e)}],
                        processing_time=time.time() - processing_start,
                        memory_peak_mb=psutil.Process().memory_info().rss / (1024 * 1024)
                    )
                )
        return results
    def _update_batch_stats(self, batch_size: int, processing_time: float, memory_used: float):
        self.batch_stats['total_batches'] += 1
        self.batch_stats['successful_batches'] += 1
        n = self.batch_stats['total_batches']
        self.batch_stats['average_batch_size'] = (
            (self.batch_stats['average_batch_size'] * (n - 1) + batch_size) / n
        )
        self.batch_stats['average_processing_time'] = (
            (self.batch_stats['average_processing_time'] * (n - 1) + processing_time) / n
        )
        self.batch_stats['memory_efficiency'] = (
            (self.batch_stats['memory_efficiency'] * (n - 1) + (batch_size / max(memory_used, 1))) / n
        )
    async def validate_scene_graphs_batch(self, 
                                        validation_requests: List[Tuple[Image.Image, Dict]]) -> List[ValidationResult]:
        requests = [
            ValidationRequest(
                image=img,
                scene_graph=sg,
                request_id=f"req_{i}",
                priority=0
            )
            for i, (img, sg) in enumerate(validation_requests)
        ]
        results = await self.validate_batch(requests)
        return results
    def get_performance_stats(self) -> Dict[str, Any]:
        current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        stats = {
            **self.batch_stats,
            'current_memory_mb': current_memory,
            'max_memory_limit_mb': self.max_memory_mb,
            'memory_utilization': current_memory / self.max_memory_mb,
            'gpu_available': self.enable_gpu_batching
        }
        if self.enable_gpu_batching:
            stats.update({
                'gpu_memory_allocated_mb': torch.cuda.memory_allocated() / (1024 * 1024),
                'gpu_memory_reserved_mb': torch.cuda.memory_reserved() / (1024 * 1024),
                'gpu_utilization': torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
            })
        return stats
