import torch
import numpy as np
import pickle
import hashlib
import os
import sqlite3
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Union
import psutil
import logging
from collections import OrderedDict
import gc

class MemoryEfficientFeatureCache:
    """
    Professional disk caching system optimized for RTX 3050 Ti (4GB VRAM).
    Uses SQLite + compressed features with smart memory management.
    """
    def __init__(self, 
                 cache_dir: str = "cache/features",
                 max_memory_cache_mb: int = 2048,  # 2GB for in-memory cache
                 compression_level: int = 6,
                 enable_gpu_cache: bool = True,
                 max_gpu_cache_mb: int = 512):  # 512MB for GPU cache (conservative)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "cache_metadata.db"
        self._init_database()
        self.max_memory_cache = max_memory_cache_mb * 1024 * 1024  # Convert to bytes
        self.memory_cache = OrderedDict()  # LRU cache
        self.cache_lock = threading.RLock()
        self.enable_gpu_cache = enable_gpu_cache and torch.cuda.is_available()
        self.max_gpu_cache = max_gpu_cache_mb * 1024 * 1024
        self.gpu_cache = OrderedDict() if self.enable_gpu_cache else None
        self.compression_level = compression_level
        logging.info(f"MemoryEfficientFeatureCache initialized: {cache_dir}")
        logging.info(f"Memory cache: {max_memory_cache_mb}MB, GPU cache: {max_gpu_cache_mb}MB")
    def _init_database(self):
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    cache_key TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    feature_shape TEXT NOT NULL,
                    dtype TEXT NOT NULL,
                    compressed_size INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 1
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_last_accessed 
                ON cache_entries(last_accessed)
            """)
    def _generate_cache_key(self, 
                          image_path: str, 
                          mask_hash: str, 
                          model_config: Dict) -> str:
        key_data = f"{image_path}_{mask_hash}_{str(sorted(model_config.items()))}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]
    def _get_memory_usage(self) -> int:
        return sum(
            tensor.element_size() * tensor.nelement() 
            for tensor in self.memory_cache.values()
        )
    def _cleanup_memory_cache(self):
        current_usage = self._get_memory_usage()
        while current_usage > self.max_memory_cache and self.memory_cache:
            oldest_key, oldest_tensor = self.memory_cache.popitem(last=False)
            current_usage -= oldest_tensor.element_size() * oldest_tensor.nelement()
            logging.debug(f"Evicted cache entry: {oldest_key}")
    def _cleanup_gpu_cache(self):
        if not self.enable_gpu_cache:
            return
        gpu_usage = sum(
            tensor.element_size() * tensor.nelement() 
            for tensor in self.gpu_cache.values()
        )
        while gpu_usage > self.max_gpu_cache and self.gpu_cache:
            oldest_key, oldest_tensor = self.gpu_cache.popitem(last=False)
            gpu_usage -= oldest_tensor.element_size() * oldest_tensor.nelement()
            del oldest_tensor
            torch.cuda.empty_cache()
    def store_features(self, 
                      cache_key: str,
                      features: torch.Tensor,
                      metadata: Dict[str, Any]) -> None:
        with self.cache_lock:
            file_path = self.cache_dir / f"{cache_key}.npz"
            features_cpu = features.detach().cpu().numpy()
            np.savez_compressed(
                str(file_path),
                features=features_cpu,
                metadata=metadata
            )
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO cache_entries 
                    (cache_key, file_path, feature_shape, dtype, compressed_size, last_accessed)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    cache_key,
                    str(file_path),
                    str(features_cpu.shape),
                    str(features_cpu.dtype),
                    file_path.stat().st_size if file_path.exists() else 0
                ))
            self._cleanup_memory_cache()
            if self._get_memory_usage() + features.element_size() * features.nelement() < self.max_memory_cache:
                self.memory_cache[cache_key] = features.detach().cpu()
                self.memory_cache.move_to_end(cache_key)
            if self.enable_gpu_cache:
                self._cleanup_gpu_cache()
    def load_features(self, cache_key: str, device: Optional[str] = None) -> Optional[torch.Tensor]:
        with self.cache_lock:
            device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
            if self.enable_gpu_cache and cache_key in self.gpu_cache:
                tensor = self.gpu_cache[cache_key]
                self.gpu_cache.move_to_end(cache_key)
                logging.debug(f"Cache hit (GPU): {cache_key}")
                return tensor.to(device)
            if cache_key in self.memory_cache:
                tensor = self.memory_cache[cache_key]
                self.memory_cache.move_to_end(cache_key)
                if (self.enable_gpu_cache and 
                    str(device).startswith('cuda') and 
                    cache_key not in self.gpu_cache):
                    self._cleanup_gpu_cache()
                    gpu_tensor = tensor.to(device)
                    gpu_size = gpu_tensor.element_size() * gpu_tensor.nelement()
                    if gpu_size < self.max_gpu_cache:
                        self.gpu_cache[cache_key] = gpu_tensor
                        self.gpu_cache.move_to_end(cache_key)
                    return gpu_tensor
                return tensor.to(device)
            # Disk load
            with sqlite3.connect(str(self.db_path)) as conn:
                result = conn.execute("""
                    SELECT file_path FROM cache_entries WHERE cache_key = ?
                """, (cache_key,)).fetchone()
                if not result:
                    return None
                file_path = Path(result[0])
                if not file_path.exists():
                    conn.execute("""
                        DELETE FROM cache_entries WHERE cache_key = ?
                    """, (cache_key,))
                    return None
                conn.execute("""
                    UPDATE cache_entries 
                    SET last_accessed = CURRENT_TIMESTAMP,
                        access_count = access_count + 1
                    WHERE cache_key = ?
                """, (cache_key,))
            try:
                with np.load(str(file_path), allow_pickle=True) as data:
                    features_np = data['features']
                tensor = torch.from_numpy(features_np)
                self._cleanup_memory_cache()
                if self._get_memory_usage() + tensor.element_size() * tensor.nelement() < self.max_memory_cache:
                    self.memory_cache[cache_key] = tensor
                    self.memory_cache.move_to_end(cache_key)
                if self.enable_gpu_cache and str(device).startswith('cuda'):
                    self._cleanup_gpu_cache()
                    gpu_tensor = tensor.to(device)
                    gpu_size = gpu_tensor.element_size() * gpu_tensor.nelement()
                    if gpu_size < self.max_gpu_cache:
                        self.gpu_cache[cache_key] = gpu_tensor
                        self.gpu_cache.move_to_end(cache_key)
                    return gpu_tensor
                return tensor.to(device)
            except Exception as e:
                logging.error(f"Failed to load features from disk: {e}")
                return None
    def get_cache_stats(self) -> Dict[str, Any]:
        with sqlite3.connect(str(self.db_path)) as conn:
            total_entries = conn.execute("""
                SELECT COUNT(*) FROM cache_entries
            """).fetchone()[0]
            total_disk_size = conn.execute("""
                SELECT SUM(compressed_size) FROM cache_entries
            """).fetchone()[0] or 0
        memory_usage = self._get_memory_usage()
        gpu_usage = 0
        if self.enable_gpu_cache:
            gpu_usage = sum(
                t.element_size() * t.nelement() 
                for t in self.gpu_cache.values()
            )
        return {
            'total_entries': total_entries,
            'disk_size_mb': total_disk_size / (1024 * 1024),
            'memory_cache_entries': len(self.memory_cache),
            'memory_usage_mb': memory_usage / (1024 * 1024),
            'gpu_cache_entries': len(self.gpu_cache) if self.enable_gpu_cache else 0,
            'gpu_usage_mb': gpu_usage / (1024 * 1024),
            'system_memory_available_mb': psutil.virtual_memory().available / (1024 * 1024),
            'gpu_memory_available_mb': (
                torch.cuda.get_device_properties(0).total_memory - 
                torch.cuda.memory_allocated()
            ) / (1024 * 1024) if torch.cuda.is_available() else 0
        }
