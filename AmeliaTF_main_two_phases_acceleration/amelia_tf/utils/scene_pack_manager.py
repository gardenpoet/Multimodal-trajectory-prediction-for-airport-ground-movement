# scene_pack_manager.py
import tarfile
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import os
import time
from collections import OrderedDict
import threading
import concurrent.futures
from dataclasses import dataclass


@dataclass
class CacheEntry:
    """Cache entry"""
    data: Any
    access_time: float
    size: int  # Data size in bytes


class FastScenePackManager:
    """High-performance scene manager with batch prefetching"""

    def __init__(
        self,
        index_file_path: str,
        max_memory_cache=100,   # Max number of files in memory cache
        max_tar_cache=5,        # Number of cached tar files
        prefetch_size=10,       # Prefetch window size
        num_workers=2           # Number of prefetch worker threads
    ):
        self.index_file = Path(index_file_path)
        self.index_dir = self.index_file.parent
        self.airport = self.index_dir.name

        self.max_memory_cache = max_memory_cache
        self.max_tar_cache = max_tar_cache
        self.prefetch_size = prefetch_size
        self.num_workers = num_workers

        print(f"High-performance manager initialized for airport: {self.airport}")
        print(
            f"  Configuration: memory_cache={max_memory_cache}, "
            f"tar_cache={max_tar_cache}, prefetch={prefetch_size}"
        )

        # Load index
        self._load_index()
        self._build_index_mapping()

        # Cache system
        self._memory_cache = OrderedDict()   # index -> CacheEntry
        self._tar_cache = OrderedDict()      # tar_path -> tarfile handle
        self._cache_lock = threading.RLock()

        # Prefetch system
        self._prefetch_executor = None
        self._stop_prefetch = False
        self._prefetch_futures = {}

        # Statistics
        self._stats = {
            'total_loads': 0,
            'memory_hits': 0,
            'tar_hits': 0,
            'disk_loads': 0,
            'prefetch_hits': 0,
            'total_open_time': 0.0,
            'total_load_time': 0.0,
        }

        # Start prefetch executor
        self._start_prefetch_executor()

    def _load_index(self):
        """Load scene index"""
        with open(self.index_file, 'r') as f:
            self.scene_index = json.load(f)

    def _build_index_mapping(self):
        """Build index mapping"""
        self.index_map = []
        self._tar_distribution = {}  # tar_path -> number of files

        for scene_name in sorted(self.scene_index.keys()):
            info = self.scene_index[scene_name]
            tar_path = info['tar_file']
            if not os.path.isabs(tar_path):
                tar_path = os.path.join(self.index_dir, tar_path)

            file_count = 0
            for pkl_info in sorted(info['pkl_files'], key=lambda x: x['arcname']):
                self.index_map.append({
                    'tar_path': tar_path,
                    'scene_name': scene_name,
                    'arcname': pkl_info['arcname'],
                    'agents': pkl_info['agents'],
                    'filename': pkl_info.get('name', '')
                })
                file_count += 1

            self._tar_distribution[tar_path] = file_count

        print(f"  Index mapping contains {len(self.index_map)} files")
        print(f"  Distributed across {len(self._tar_distribution)} tar files")

        sorted_tars = sorted(
            self._tar_distribution.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for tar_path, count in sorted_tars[:3]:
            print(f"    {os.path.basename(tar_path)}: {count} files")
        if len(sorted_tars) > 3:
            print(f"    ... and {len(sorted_tars) - 3} more tar files")

    def _start_prefetch_executor(self):
        """Start prefetch executor"""
        self._prefetch_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_workers,
            thread_name_prefix=f"Prefetch_{self.airport}"
        )

    def _get_tar_handle(self, tar_path: str) -> tarfile.TarFile:
        """Get tar handle with caching"""
        with self._cache_lock:
            if tar_path in self._tar_cache:
                self._stats['tar_hits'] += 1
                self._tar_cache.move_to_end(tar_path)
                return self._tar_cache[tar_path]

            if len(self._tar_cache) >= self.max_tar_cache:
                _, old_tar = self._tar_cache.popitem(last=False)
                try:
                    old_tar.close()
                except Exception:
                    pass

            open_start = time.time()
            tar = tarfile.open(tar_path, 'r')
            self._stats['total_open_time'] += time.time() - open_start

            self._tar_cache[tar_path] = tar
            return tar

    def _load_from_disk(self, index: int) -> Optional[tuple]:
        """Load a single file from disk, return (data, size)"""
        info = self.index_map[index]
        tar_path = info['tar_path']
        arcname = info['arcname']

        try:
            tar = self._get_tar_handle(tar_path)
            member = tar.getmember(arcname)
            f = tar.extractfile(member)
            if f is None:
                return None

            data_bytes = f.read()
            f.close()
            data = pickle.loads(data_bytes)

            size = len(data_bytes) + len(pickle.dumps(data))
            self._stats['total_load_time'] += time.time()

            return data, size

        except Exception as e:
            print(f"Failed to load index={index}, file={arcname}: {e}")
            return None

    def _prefetch_worker(self, indices: List[int]):
        """Prefetch worker"""
        for idx in indices:
            if self._stop_prefetch:
                break

            with self._cache_lock:
                if idx in self._memory_cache:
                    self._stats['prefetch_hits'] += 1
                    continue

            result = self._load_from_disk(idx)
            if result is None:
                continue

            data, size = result
            with self._cache_lock:
                if len(self._memory_cache) >= self.max_memory_cache:
                    self._memory_cache.popitem(last=False)

                self._memory_cache[idx] = CacheEntry(
                    data=data,
                    access_time=time.time(),
                    size=size
                )

    def request_prefetch(self, indices: List[int]):
        """Request prefetch for a batch of indices"""
        if not indices or self._stop_prefetch:
            return

        with self._cache_lock:
            indices = [i for i in indices if i not in self._memory_cache]

        if not indices:
            return

        indices = indices[:self.prefetch_size * 2]
        future = self._prefetch_executor.submit(self._prefetch_worker, indices)
        self._prefetch_futures[id(future)] = future

    def load_by_index(self, index: int) -> Any:
        """Load data by index"""
        self._stats['total_loads'] += 1

        with self._cache_lock:
            if index in self._memory_cache:
                self._stats['memory_hits'] += 1
                entry = self._memory_cache.pop(index)
                entry.access_time = time.time()
                self._memory_cache[index] = entry
                return entry.data

        self._stats['disk_loads'] += 1
        result = self._load_from_disk(index)
        if result is None:
            return None

        data, size = result
        with self._cache_lock:
            if len(self._memory_cache) >= self.max_memory_cache:
                self._memory_cache.popitem(last=False)

            self._memory_cache[index] = CacheEntry(
                data=data,
                access_time=time.time(),
                size=size
            )

        next_indices = [
            i for i in range(index + 1, min(index + 1 + self.prefetch_size, len(self.index_map)))
        ]
        if next_indices:
            self.request_prefetch(next_indices)

        return data

    def get_total_files(self):
        """Return total number of pickle files"""
        return len(self.index_map)

    def get_file_info_by_index(self, index: int) -> Dict:
        """Get file metadata by index"""
        if 0 <= index < len(self.index_map):
            return self.index_map[index]
        return None

    def filter_by_agents_and_scenes(
        self, min_agents: int, max_agents: int, scene_names: List[str]
    ) -> List[int]:
        """Filter by agent count and scene names"""
        return [
            idx for idx, info in enumerate(self.index_map)
            if min_agents <= info['agents'] <= max_agents
            and info['scene_name'] in scene_names
        ]

    def filter_by_agents(self, min_agents: int, max_agents: int) -> List[int]:
        """Filter by agent count"""
        return [
            idx for idx, info in enumerate(self.index_map)
            if min_agents <= info['agents'] <= max_agents
        ]

    def load_pickle(self, scene_name: str, arcname: str) -> Any:
        """Backward-compatible interface"""
        for idx, info in enumerate(self.index_map):
            if info['scene_name'] == scene_name and info['arcname'] == arcname:
                return self.load_by_index(idx)
        return None

    def cleanup(self):
        """Clean up all resources"""
        self._stop_prefetch = True
        if self._prefetch_executor:
            self._prefetch_executor.shutdown(wait=False)

        with self._cache_lock:
            for tar in self._tar_cache.values():
                try:
                    tar.close()
                except Exception:
                    pass
            self._tar_cache.clear()
            self._memory_cache.clear()


class ScenePackManager(FastScenePackManager):
    """Simplified wrapper for backward compatibility"""

    def __init__(self, index_file_path: str, max_cached_tars=3, cache_timeout=300):
        super().__init__(
            index_file_path=index_file_path,
            max_memory_cache=50,
            max_tar_cache=max_cached_tars,
            prefetch_size=10,
            num_workers=1
        )
        print(
            f"Compatibility mode enabled: "
            f"max_cached_tars={max_cached_tars}, cache_timeout={cache_timeout}s"
        )
