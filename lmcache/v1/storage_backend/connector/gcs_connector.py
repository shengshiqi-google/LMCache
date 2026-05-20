# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Optional
from enum import IntEnum, auto
import asyncio
import gcsfs
from concurrent.futures import ThreadPoolExecutor

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.storage_backend.job_executor.pq_executor import AsyncPQExecutor

logger = init_logger(__name__)

class Priorities(IntEnum):
    PEEK = auto()
    PREFETCH = auto()
    GET = auto()
    PUT = auto()


class GCSConnector(RemoteConnector):
    """
    GCS Native Connector for LMCache, optimized for GCS Rapid Zonal Buckets.
    Uses standard gcsfs and ThreadPoolExecutor for loop-agnostic robustness.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        config: Optional[LMCacheEngineConfig],
        plugin_name: Optional[str] = None,
        bucket_name_str: Optional[str] = None,
    ):
        # initialize base class
        super().__init__(
            local_cpu_backend.config,
            local_cpu_backend.metadata,
        )

        self.loop = loop
        self.local_cpu_backend = local_cpu_backend

        # Extract bucket name
        bucket_name = bucket_name_str
        if bucket_name is None:
            if config and config.remote_url:
                bucket_name = config.remote_url.removeprefix("gs://")
            
            if bucket_name is None and config and config.extra_config:
                key_prefix = plugin_name or "gcs"
                bucket_name = config.extra_config.get(
                    "remote_storage_plugin.%s.bucket_name" % key_prefix
                )

        if not bucket_name:
            raise ValueError("GCS connector requires a valid bucket_name via URL (gs://...) or extra_config.")

        self.bucket_name = bucket_name
        
        gcsfs.GCSFileSystem.clear_instance_cache()
        self.fs = gcsfs.GCSFileSystem(consistency='none')

        # Resolve max_workers from extra_config with default of 4
        max_workers = 4
        if config is not None:
            max_workers = config.get_extra_config_value("gcs_max_workers", 4)

        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="gcs_connector")
        self.pq_executor = AsyncPQExecutor(loop, max_workers=max_workers)
        logger.info(f"Initialized GCSConnector with bucket name: {self.bucket_name}, max_workers: {max_workers}")

    def _get_object_path(self, key: CacheEngineKey) -> str:
        key_str = key.to_string()
        return f"{self.bucket_name}/{key_str}"

    async def _exists(self, key: CacheEngineKey) -> bool:
        def _exists():
            path = self._get_object_path(key)
            return self.fs.exists(path)
        return await self.loop.run_in_executor(self.executor, _exists)

    async def exists(self, key: CacheEngineKey) -> bool:
        return await self.pq_executor.submit_job(
            self._exists, key=key, priority=Priorities.PEEK
        )

    def exists_sync(self, key: CacheEngineKey) -> bool:
        path = self._get_object_path(key)
        return self.fs.exists(path)

    async def _get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        path = self._get_object_path(key)
        
        if not await self._exists(key):
            return None

        memory_obj = self.local_cpu_backend.allocate(
            self.meta_shapes, self.meta_dtypes, self.meta_fmt
        )
        if memory_obj is None:
            logger.debug("Memory allocation failed during GCS load.")
            return None

        def _download_data(memory_obj):
            # Strip the endian prefix for zero-copy safety
            buffer = memory_obj.byte_array.cast("B")
            with self.fs.open(path, 'rb', block_size=self.full_chunk_size_bytes) as f:
                num_read = f.readinto(buffer)
            return num_read

        try:
            num_read = await self.loop.run_in_executor(self.executor, _download_data, memory_obj)
            memory_obj = self.reshape_partial_chunk(memory_obj, num_read)
            return memory_obj
        except Exception as e:
            logger.error(f"Failed to read from GCS path {path}: {e}")
            if memory_obj is not None:
                memory_obj.ref_count_down()
            return None

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        return await self.pq_executor.submit_job(
            self._get, key=key, priority=Priorities.GET
        )

    async def _put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        path = self._get_object_path(key)
        buffer = memory_obj.byte_array

        def _upload_data():
            with self.fs.open(path, 'wb', block_size=self.full_chunk_size_bytes) as f:
                f.write(buffer)

        try:
            await self.loop.run_in_executor(self.executor, _upload_data)
        except Exception as e:
            logger.error(f"Failed to write to GCS path {path}: {e}")
            raise

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        await self.pq_executor.submit_job(
            self._put, key=key, memory_obj=memory_obj, priority=Priorities.PUT
        )

    async def list(self) -> List[str]:
        def _list():
            objects = self.fs.ls(self.bucket_name)
            return [o.removeprefix(f"{self.bucket_name}/") for o in objects]
        return await self.loop.run_in_executor(self.executor, _list)

    def support_batched_contains(self) -> bool:
        return True

    def batched_contains(self, keys: List[CacheEngineKey]) -> int:
        from concurrent.futures import ThreadPoolExecutor
        
        def _check_single(key):
            try:
                return self.exists_sync(key)
            except Exception as e:
                logger.warning(f"Error checking existence for key {key.to_string()}: {e}")
                return False

        # Query up to 64 GCS metadata exists requests in parallel
        with ThreadPoolExecutor(max_workers=64) as executor:
            results = list(executor.map(_check_single, keys))

        hit_chunks = 0
        for hit in results:
            if not hit:
                break
            hit_chunks += 1
        return hit_chunks

    async def close(self):
        await self.pq_executor.shutdown_async(wait=True)
        self.executor.shutdown(wait=False)
        logger.info("Closed the GCS connector")
