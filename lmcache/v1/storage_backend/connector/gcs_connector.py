# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Optional
from urllib.parse import urlparse
import asyncio
import os

# Third Party
from google.cloud import storage as gcs_sync
from google.cloud.storage.asyncio.async_grpc_client import AsyncGrpcClient
from google.cloud.storage.asyncio.async_appendable_object_writer import AsyncAppendableObjectWriter
from google.cloud.storage.asyncio.async_multi_range_downloader import AsyncMultiRangeDownloader
import grpc

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend


logger = init_logger(__name__)

class MemoryViewSink:
    def __init__(self, buffer: memoryview):
        self.buffer: memoryview = memoryview(buffer).cast("B")
        self.offset: int = 0

    
    def writable(self) -> bool:
        return True
    
    def write(self, data: "memoryview | bytes | bytearray") -> int:
        view = memoryview(data).cast("B")
        n = len(view)
        end = self.offset + n
        if end > len(self.buffer):
            raise ValueError(
                f"MemoryViewSink overflow: writitng {n} bytes at offset "
                f"{self.offset} exceeds buffer size {len(self.buffer)}"
            )
        self.buffer[self.offset : end] = view
        self.offset = end
        return n
    
    def tell(self) -> int:
        return self.offset
    
    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.offset = offset
        elif whence == 1:
            self.offset += offset
        elif whence == 2:
            self.offset = len(self.buffer) + offset
        else:
            raise ValueError(f"Invalid whence: {whence}")
        return self.offset
    
    def __len__(self) -> int:
        return len(self.buffer)
    
class GcsConnector(RemoteConnector):

    def __init__(
        self,
        url: str,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
    ):
        super().__init__(config, metadata)

        if not url.startswith("gs://"):
            raise ValueError(f"GCS url must start with 'gs://': {url}")
        parsed = urlparse(url)
        bucket = parsed.netloc
        if not bucket:
            raise ValueError(f"GCS url missing bucket name: {url}")
        prefix = parsed.path.lstrip("/")

        self.url: str = url
        self.loop: asyncio.AbstractEventLoop = loop
        self.local_cpu_backend: LocalCPUBackend = local_cpu_backend
        self.bucket_name: str = bucket
        self.key_prefix: str = prefix

        self._grpc_client: Optional[AsyncGrpcClient] = None
        self._sync_client: Optional[gcs_sync.Client] = None
        self._init_lock: Optional[asyncio.Lock] = None
        self._bound_loop: Optional[asyncio.AbstractEventLoop] = None

        logger.info(
            "GcsConnector configured for bucket=%s prefix =%s",
            self.bucket_name,
            self.key_prefix,
        )
    
    async def _ensure_async_client(self) -> AsyncGrpcClient:
        current_loop = asyncio.get_running_loop()
        if self._init_lock is None:
            self._init_lock = asyncio.Lock()
        async with self._init_lock:
            if self._grpc_client is None:
                self._grpc_client = AsyncGrpcClient()
                self._bound_loop = current_loop
                logger.info(
                    "initialized AsyncGrpcClient for bucket=%s",
                    self.bucket_name
                )
            elif self._bound_loop is not current_loop:
                raise RuntimeError(
                    "GCSConnector AsyncGrpcClient is bound to a different "
                )
        return self._grpc_client
    
    def _ensure_sync_client(self) -> "gcs_sync.Client":
        if self._sync_client is None:
            self._sync_client = gcs_sync.Client()
            logger.info(
                "Initialized sync storage.Client for bucket=%s",
                self.bucket_name,
            )
        return self._sync_client
    
    def _object_name(self, key: CacheEngineKey) -> str:
        key_str = key.to_string()
        if self.key_prefix:
            return f"{self.key_prefix}/{key_str}"
        return key_str
    
    async def exists(self, key: CacheEngineKey) -> bool:
        return await asyncio.get_running_loop().run_in_executor(
            None, self.exists_sync, key
        )
    
    def exists_sync(self, key: CacheEngineKey) -> bool:
        client = self._ensure_sync_client()
        bucket = client.bucket(self.bucket_name)
        blob = bucket.blob(self._object_name(key))
        return blob.exists()
    
    def support_batched_contains(self) -> bool:
        return True

    def batched_contains(self, keys: List[CacheEngineKey]) -> int:
        if not keys:
            return 0
        client = self._ensure_sync_client()
        names = [self._object_name(k) for k in keys]
        common_prefix = os.path.commonprefix(names)
        found = {
            blob.name
            for blob in client.list_blobs(
                self.bucket_name, prefix = common_prefix
            )
        }
        count = 0
        for name in names:
            if name not in found:
                break
            count += 1
        return count
    
    
    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        grpc_client = await self._ensure_async_client()
        object_name = self._object_name(key)

        memory_obj = self.local_cpu_backend.allocate(
            self.meta_shapes,
            self.meta_dtypes,
            self.meta_fmt,
        )

        if memory_obj is None:
            return None
        
        expected_size = memory_obj.get_size()
        sink = MemoryViewSink(memoryview(memory_obj.byte_array))
        mrd = AsyncMultiRangeDownloader(
            grpc_client, self.bucket_name, object_name
        )

        success = False
        try:
            try:
                await mrd.open()
            except grpc.RpcError as e:
                if _is_not_found(e):
                    return None
                raise
            
            try:
                await mrd.download_ranges([(0, expected_size, sink)])
            finally:
                if mrd.is_stream_open:
                    try:
                        await mrd.close()
                    except Exception as ex:
                        logger.warning(
                            "Error closing MRD for %s: %s",
                            object_name,
                            ex,
                        )
            
            if sink.tell() != expected_size:
                logger.error(
                    "GCS object size mismatch for %s: got %d bytes, "
                    "expected %d. Partial chunks are not supported.",
                    object_name,
                    sink.tell(),
                    expected_size,
                )
                return None

            success = True
            return memory_obj
        
        except grpc.RpcError as e:
            if _is_not_found(e):
                return None
            logger.error(
                "GCS download failed for %s: %s", object_name, e
            )
            raise
        finally:
            if not success:
                memory_obj.ref_count_down()
        
    async def put(
        self, key: CacheEngineKey, memory_obj: MemoryObj
    ) -> None:
        grpc_client = await self._ensure_async_client()
        object_name = self._object_name(key)

        writer = AsyncAppendableObjectWriter(
            client=grpc_client,
            bucket_name=self.bucket_name,
            object_name=object_name,
            generation=0,
        )

        opened = False
        finalized = False
        try:
            try:
                await writer.open()
                opened = True
            except grpc.RpcError as e:
                if _is_already_exists(e):
                    logger.debug(
                        "GCS object %s already exists; skipping put.",
                        object_name,
                    )
                    return
                raise
            
            await writer.append(memoryview(memory_obj.byte_array))
            await writer.finalize()
            finalized = True
        except grpc.RpcError as e:
            logger.error(
                "GCS upload failed for %s: %s", object_name, e
            )
            raise
        finally:
            if opened and not finalized:
                try:
                    await writer.close()
                except Exception as ex:
                    logger.warning(
                        "Error closing writer for %s: %s",
                        object_name,
                        ex,
                    )
    
    async def list(self) -> List[str]:
        raise NotImplementedError(
            "GcsConnector.list() is not implemented"
        )
    
    async def close(self) -> None:
        if self._grpc_client is not None:
            try:
                await self._grpc_client.close()
            except Exception as e:
                logger.warning(
                    "Error closing AsyncGrpcClient: %s", e
                )
            self._grpc_client = None
        
        if self._sync_client is not None:
            try:
                self._sync_client.close()
            except Exception as e:
                logger.warning(
                    "Error closing sync storage.Client: %s", e
                )
            self._sync_client = None
        
def _is_not_found(error: grpc.RpcError) -> bool:
    code = getattr(error, "code", None)
    if callable(code):
        try:
            return code() == grpc.StatusCode.NOT_FOUND
        except Exception:
            return False
    return code == grpc.StatusCode.NOT_FOUND

def _is_already_exists(error: grpc.RpcError) -> bool:
    code = getattr(error, "code", None)
    if callable(code):
        try:
            actual = code()
        except Exception:
            return False
    else:
        actual = code
    return actual in (
        grpc.StatusCode.ALREADY_EXISTS,
        grpc.StatusCode.FAILED_PRECONDITION,
    )