# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Optional
import asyncio
import gcsfs

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.config import LMCacheEngineConfig

logger = init_logger(__name__)

class GCSConnector(RemoteConnector):
    """
    Bare-minimum synchronous first-principles GCS Connector.
    No thread pools, no async complex executor. Just basic get/put.
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

        self.local_cpu_backend = local_cpu_backend

        # Extract bucket name
        bucket_name = bucket_name_str
        if bucket_name is None:
            # Resolve from extra_config or remote_url
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
        self.fs = gcsfs.GCSFileSystem()
        logger.info(f"Initialized bare-minimum GCSConnector with bucket name: {self.bucket_name}")

    def _get_object_path(self, key: CacheEngineKey) -> str:
        key_str = key.to_string()
        return f"{self.bucket_name}/{key_str}"

    async def exists(self, key: CacheEngineKey) -> bool:
        return self.exists_sync(key)

    def exists_sync(self, key: CacheEngineKey) -> bool:
        path = self._get_object_path(key)
        return self.fs.exists(path)

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        path = self._get_object_path(key)
        
        if not self.exists_sync(key):
            return None

        memory_obj = self.local_cpu_backend.allocate(
            self.meta_shapes, self.meta_dtypes, self.meta_fmt
        )
        if memory_obj is None:
            logger.debug("Memory allocation failed during GCS load.")
            return None

        try:
            buffer = memory_obj.byte_array.cast("B")
            with self.fs.open(path, 'rb', block_size=self.full_chunk_size_bytes) as f:
                num_read = f.readinto(buffer)
            memory_obj = self.reshape_partial_chunk(memory_obj, num_read)
            return memory_obj
        except Exception as e:
            logger.error(f"Failed to read from GCS path {path}: {e}")
            if memory_obj is not None:
                memory_obj.ref_count_down()
            return None

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        path = self._get_object_path(key)
        buffer = memory_obj.byte_array.cast("B")

        try:
            with self.fs.open(path, 'wb', block_size=self.full_chunk_size_bytes) as f:
                f.write(buffer)
        except Exception as e:
            logger.error(f"Failed to write to GCS path {path}: {e}")
            raise

    async def list(self) -> List[str]:
        objects = self.fs.ls(self.bucket_name)
        return [o.removeprefix(f"{self.bucket_name}/") for o in objects]

    def support_batched_contains(self) -> bool:
        return True

    def batched_contains(self, keys: List[CacheEngineKey]) -> int:
        hit_chunks = 0
        for key in keys:
            if not self.exists_sync(key):
                break
            hit_chunks += 1
        return hit_chunks

    async def close(self):
        logger.info("Closed bare-minimum GCS connector")

