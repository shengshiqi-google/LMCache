# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Optional
from urllib.parse import urlparse
import asyncio
import os

# Third Party
from google.cloud.storage.asyncio.async_grpc_client import AsyncGrpcClient
from google.cloud.storage.asyncio.async_appendable_object_writer import AsyncAppendableObjectWriter
from google.cloud.storage.asyncio.async_multi_range_downloader import AsyncMultiRangeDownloader
from google.api_core import exceptions
from google.cloud import _storage_v2 as storage_v2

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
    """
    Writable file-like wrapping a memoryview with a bounded write cursor.

    Used as a zero-copy download sink: bytes received from the GCS helper
    are written directly into a pre-allocated buffer (the byte_array of a
    MemoryObj) without an intermediate BytesIO round-trip.

    Writes that would exceed the buffer size raise ValueError, protecting
    against the case where the remote object is larger than the caller's
    pre-allocated buffer.
    """

    def __init__(self, buffer: memoryview):
        """
        Initalize the sink over a writable buffer.

        Args:
            buffer: A memoryview (or bytes-like supporting the buffer
                protocol) that the sink will write into. the view is
                recast to bytes (`'B'`) so writes are byte-addressable.
        """
        self.buffer: memoryview = memoryview(buffer).cast("B")
        self.offset: int = 0

    def writable(self) -> bool:
        """
        Indiate that the sink is writable.

        Returns:
            Always True.
        """
        return True
    
    def write(self, data: "memoryview | bytes | bytearray") -> int:
        """
        Copy `data` into the underlying buffer at the current offset.

        Args:
            data: A bytes-like object (bytes, bytearray, or memoryview).

        Returns:
            The number of bytes written.

        Raises:
            ValueError: If writing would exceed the buffer size.
        """
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
        """
        Return the current write offset.

        Returns:
            The byte offset at which the next write will begin.
        """
        return self.offset
    
    def seek(self, offset: int, whence: int = 0) -> int:
        """
        Set the write cursor.

        Args:
            offset: Byte offset (interpretation depends on `whence`).
            whence: 0 for absolute, 1 for relative to current, 2 for
                relative to end of buffer. Mirrors `io.IOBase`.
            
        Returns:
            The new absolute offset.
            
        Raises:
            ValueError: If `whence` is not 0, 1, or 2.
        """
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
    """
    Remote connector for Google Cloud Storage Rapid (zonal) buckets.

    Reads use `AsyncMultiRangeDownloader` and writes use
    `AsyncAppendableObjectWriter` (with `finalize()`), both from
    `google.cloud.storage.asyncio`. 

    Authentication uses Application Default Credentials on the host VM.

    URL format:
        ``gs://<bucket>[/<key_prefix>]``
    
    The async gRPC client is constructed lazily on first async use
    because it binds to the running event loop at construction time;
    creating it eagerly in `__init__` risks binding to a loop that is
    not yet (or no longer) running.
    """

    def __init__(
        self,
        url: str,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
    ):
        """
        Initialize the GCS connector.

        Args:
            url: A URL of the form ``gs://<bucket>[/<key_prefix>]``.
            loop: The asyncio event loop intended for this connector.
                Stored for parity with peer connectors; the async gRPC
                client is bound to whichever loop calls the first async
                method.
            local_cpu_backend: USed to allocated `MemoryObj` instances on
                the read path.
            config: LMCache engine config.
            metadata: LMCache engine metadata.
        
        Raises:
            ValueError: If `url` does not start with ``gs://`` or does
                not contain a bucket name.
        """
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

        # Lazy-initialized clients. The async gRPC client binds to the
        # running loop at construction; constructing it in __init__
        # before the caller's loop is live would bind it to the wrong
        # (or dead) loop.
        self._grpc_client: Optional[AsyncGrpcClient] = None
        self._init_lock: Optional[asyncio.Lock] = None
        self._bound_loop: Optional[asyncio.AbstractEventLoop] = None

        logger.info(
            "GcsConnector configured for bucket=%s prefix =%s",
            self.bucket_name,
            self.key_prefix,
        )
    
    async def _ensure_async_client(self) -> AsyncGrpcClient:
        """
        Lazy-initialize the AsyncGrpcClient on first async use.

        Captures the current running loop on first init; subsequent
        calls verify the loop has not changed. A mismatch indicates
        the connector is being reused across event loops, which is
        not supported.

        Returns:
            The lazily-initialized `AsyncGrpcClient`.
        
        Raises:
            RuntimeError: If called from a different event loop than
                the one this connector's client was first bound to.
        """
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
                    "GCSConnector AsyncGrpcClient is bound to a different event loop."
                )
        return self._grpc_client
    
    def _object_name(self, key: CacheEngineKey) -> str:
        """
        Build the GCS object name for a cache engine key.

        Args:
            key: The cache engine key.
        
        Returns:
            The object name, prefixed with `key_prefix` when one was
            configured in the URL.
        """
        key_str = key.to_string()
        if self.key_prefix:
            return f"{self.key_prefix}/{key_str}"
        return key_str
    
    async def exists(self, key: CacheEngineKey) -> bool:
        """
        Asynchronously check whether a key exists in the remove bucket.

        Args:
            key: The cache engine key.
        
        Returns:
            True if the object exists, False otherwise.
        """
        grpc_client = await self._ensure_async_client()
        try:
            await grpc_client.get_object(
                bucket_name=self.bucket_name, 
                object_name=self._object_name(key)
            )
            return True
        except exceptions.NotFound:
            return False
    
    def exists_sync(self, key: CacheEngineKey) -> bool:
        """
        Synchronously check whether a key exists in the remote bucket.

        Args:
            key: The cache engine key.
        
        Returns:
            True if the object exists, False otherwise.
        """
        # Execute the async method synchronously on the bound event loop
        future = asyncio.run_coroutine_threadsafe(self.exists(key), self.loop)
        return future.result()
    
    def support_batched_contains(self) -> bool:
        return True

    async def _batched_contains_async(self, keys: List[CacheEngineKey]) -> int:
        """Async implementation of batched_contains utilizing gRPC."""
        if not keys:
            return 0
        
        grpc_client = await self._ensure_async_client()
        names = [self._object_name(k) for k in keys]
        common_prefix = os.path.commonprefix(names)
        
        # gRPC requires the full bucket path format
        parent = f"projects/_/buckets/{self.bucket_name}"
        request = storage_v2.ListObjectsRequest(parent=parent, prefix=common_prefix)
        
        pager = await grpc_client.grpc_client.list_objects(request=request)
        found = {obj.name async for obj in pager}
        
        count = 0
        for name in names:
            if name not in found:
                break
            count += 1
        return count

    def batched_contains(self, keys: List[CacheEngineKey]) -> int:
        if not keys:
            return 0
        future = asyncio.run_coroutine_threadsafe(
            self._batched_contains_async(keys), self.loop
        )
        return future.result()
    
    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """
        Download an object into a freshly allocated `MemoryObj`.

        Allocates a `MemoryObj` sized to the LMCache full-chunk shape,
        wraps its buffer in a `MemoryViewSink`, and streams the GCS
        object directly into it with no intermediate copy. Returns
        `None` on miss or on a size mismatch (partial chunks are not
        supported on day-1).

        Args:
            key: The cache engine key.

        Returns:
            A populated `MemoryObj` on success, or `None` on miss or
            size mismatch.
        """
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
            except exceptions.NotFound:
                return None
            
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
        
        except exceptions.NotFound:
            return None
        except Exception as e:
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
        """
        Upload a `MemoryObj` as a finalized appendable object.

        Creates the object with ``generation=0`` (fail-if-exists),
        appends the buffer in a single call, and finalizes the object
        so it becomes immutable. If the object already exists, the put
        is treated as a no-op since cache values are deterministic
        given the key.

        Args:
            key: The cache engine key.
            memory_obj: The data to upload. Its `byte_array` is read
                directly via `memoryview` (no copy).
        """
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
            except (exceptions.AlreadyExists, exceptions.FailedPrecondition):
                logger.debug(
                    "GCS object %s already exists; skipping put.",
                    object_name,
                )
                return
            
            await writer.append(memoryview(memory_obj.byte_array))
            await writer.finalize()
            finalized = True
        except Exception as e:
            logger.error(
                "GCS upload failed for %s: %s", object_name, e
            )
            raise
        finally:
            # `finalize()` closes the bidi stream; calling `close()`
            # after `finalize()` is documented as undefined behavior.
            # Only close when we opened but did not finalize.
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
        """
        List operation is not implemented for day-1.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "GcsConnector.list() is not implemented"
        )
    
    async def close(self) -> None:
        """
        Close any lazily-initialized clients.

        Safe to call when no client was ever initialized, and safe to
        call multiple times.
        """
        if self._grpc_client is not None:
            try:
                await self._grpc_client.close()
            except Exception as e:
                logger.warning(
                    "Error closing AsyncGrpcClient: %s", e
                )
            self._grpc_client = None