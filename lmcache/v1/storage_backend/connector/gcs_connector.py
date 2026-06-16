# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Optional
from urllib.parse import urlparse
import asyncio
import os
import concurrent.futures
import multiprocessing
from multiprocessing import shared_memory

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

# --- Process Pool Worker Functions ---
# These must be at the module level to be picklable by multiprocessing.

# Global state for each worker process
_worker_loop = None
_worker_client = None

def _worker_init():
    """
    Initialize the worker process.
    Creates a dedicated event loop and AsyncGrpcClient for this worker.
    """
    os.environ["GRPC_DNS_RESOLVER"] = "native"
    import resource
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (65536, hard))
    except Exception:
        pass

    global _worker_loop, _worker_client
    _worker_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_worker_loop)
    _worker_client = AsyncGrpcClient()

def _worker_exists(bucket_name: str, object_name: str) -> bool:
    async def _do_exists():
        try:
            await _worker_client.get_object(
                bucket_name=bucket_name, 
                object_name=object_name
            )
            return True
        except exceptions.NotFound:
            return False
    return _worker_loop.run_until_complete(_do_exists())

def _worker_batched_contains(bucket_name: str, names: List[str]) -> int:
    async def _do_batched_contains():
        if not names:
            return 0
        common_prefix = os.path.commonprefix(names)
        parent = f"projects/_/buckets/{bucket_name}"
        request = storage_v2.ListObjectsRequest(parent=parent, prefix=common_prefix)
        
        pager = await _worker_client.grpc_client.list_objects(request=request)
        found = {obj.name async for obj in pager}
        
        count = 0
        for name in names:
            if name not in found:
                break
            count += 1
        return count
    return _worker_loop.run_until_complete(_do_batched_contains())

def _worker_get_ranges(bucket_name: str, object_name: str, expected_size: int, shm_name: str, ranges_info: list) -> bool:
    async def _do_get():
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        sinks = []
        try:
            mrd = AsyncMultiRangeDownloader(
                _worker_client, bucket_name, object_name
            )
            try:
                try:
                    await mrd.open()
                except exceptions.NotFound:
                    print(f"[{object_name}] NotFound during open")
                    return False
                
                download_ranges = []
                for start, size in ranges_info:
                    sink = MemoryViewSink(existing_shm.buf)
                    sink.offset = start
                    download_ranges.append((start, size, sink))
                    sinks.append(sink)
                    
                await mrd.download_ranges(download_ranges)
                
                for start, size, sink in download_ranges:
                    if sink.tell() != start + size:
                        print(f"[{object_name}] sink.tell() {sink.tell()} != expected {start + size}. start={start}, size={size}")
                        return False
                return True
            except exceptions.NotFound:
                print(f"[{object_name}] NotFound during download")
                return False
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[{object_name}] Exception during download: {e}")
                return False
            finally:
                if mrd.is_stream_open:
                    try:
                        await mrd.close()
                    except Exception:
                        pass
                for sink in sinks:
                    sink.buffer.release()
        finally:
            existing_shm.close()
    return _worker_loop.run_until_complete(_do_get())

def _worker_put(bucket_name: str, object_name: str, shm_name: str, size: int) -> None:
    async def _do_put():
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        try:
            view = existing_shm.buf[:size]
            writer = AsyncAppendableObjectWriter(
                client=_worker_client,
                bucket_name=bucket_name,
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
                    return
                
                await writer.append(view)
                await writer.finalize()
                finalized = True
            finally:
                if opened and not finalized:
                    try:
                        await writer.close()
                    except Exception:
                        pass
        finally:
            if 'view' in locals():
                view.release()
                del view
            existing_shm.close()
    _worker_loop.run_until_complete(_do_put())


class GcsConnector(RemoteConnector):
    """
    Remote connector for Google Cloud Storage Rapid (zonal) buckets.

    Uses a ProcessPoolExecutor with shared memory to distribute GCS 
    operations across multiple processes. This bypasses the Python GIL 
    and isolates gRPC connections from the main event loop, significantly 
    increasing throughput while minimizing memory copy overhead.

    URL format:
        ``gs://<bucket>[/<key_prefix>]``
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
            local_cpu_backend: Used to allocate `MemoryObj` instances on
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
        self.config: LMCacheEngineConfig = config

        # Configure and start the ProcessPoolExecutor
        # Read the max_workers from extra_config, default to 4
        num_workers = int(config.extra_config.get("gcs_max_workers", 4))
        self.num_workers = num_workers
        mp_context = multiprocessing.get_context("spawn")
        # We use separate executors for get and put to prevent 
        # a long backlog of background put tasks from starving the blocking get tasks!
        self.get_executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.num_workers,
            mp_context=mp_context,
            initializer=_worker_init
        )
        self.put_executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.num_workers,
            mp_context=mp_context,
            initializer=_worker_init
        )

        logger.info(
            "GcsConnector configured for bucket=%s prefix=%s with %d worker processes (shared memory, parallel mode)",
            self.bucket_name,
            self.key_prefix,
            self.num_workers
        )
        
        # Pre-warm the get_executor so child processes spawn and import torch in the background
        def _dummy_task():
            pass
        for _ in range(self.num_workers):
            self.get_executor.submit(_dummy_task)
    
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
        Asynchronously check whether a key exists in the remote bucket.
        """
        object_name = self._object_name(key)
        future = self.loop.run_in_executor(
            self.get_executor, _worker_exists, self.bucket_name, object_name
        )
        return await future
    
    def exists_sync(self, key: CacheEngineKey) -> bool:
        """
        Synchronously check whether a key exists in the remote bucket.
        """
        object_name = self._object_name(key)
        future = self.get_executor.submit(_worker_exists, self.bucket_name, object_name)
        return future.result()
    
    def support_batched_contains(self) -> bool:
        return True

    async def _batched_contains_async(self, keys: List[CacheEngineKey]) -> int:
        if not keys:
            return 0
        names = [self._object_name(k) for k in keys]
        future = self.loop.run_in_executor(
            self.get_executor, _worker_batched_contains, self.bucket_name, names
        )
        return await future

    def batched_contains(self, keys: List[CacheEngineKey]) -> int:
        if not keys:
            return 0
        names = [self._object_name(k) for k in keys]
        future = self.get_executor.submit(_worker_batched_contains, self.bucket_name, names)
        return future.result()
    
    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """
        Download an object into a freshly allocated `MemoryObj`.
        """
        object_name = self._object_name(key)

        memory_obj = self.local_cpu_backend.allocate(
            self.meta_shapes,
            self.meta_dtypes,
            self.meta_fmt,
        )

        if memory_obj is None:
            return None
        
        expected_size = memory_obj.get_size()

        # Don't split across workers if the chunk is very small (e.g., < 4MB per worker)
        workers_to_use = self.num_workers if expected_size >= self.num_workers * 4 * 1024 * 1024 else 1

        num_coros = int(self.config.extra_config.get("gcs_num_coros_per_worker", 4))
        total_chunks = workers_to_use * num_coros

        # Create shared memory
        shm = shared_memory.SharedMemory(create=True, size=expected_size)
        try:
            chunk_size = expected_size // total_chunks
            
            # Distribute chunks across workers
            worker_assignments = [[] for _ in range(workers_to_use)]
            for i in range(total_chunks):
                start = i * chunk_size
                size = expected_size - start if i == total_chunks - 1 else chunk_size
                worker_assignments[i % workers_to_use].append((start, size))
                
            futures = []
            for assignment in worker_assignments:
                if not assignment:
                    continue
                futures.append(
                    self.loop.run_in_executor(
                        self.get_executor, _worker_get_ranges, self.bucket_name, object_name, expected_size, shm.name, assignment
                    )
                )

            # Wait for all byte ranges to be downloaded
            results = await asyncio.gather(*futures)
            
            if not all(results):
                memory_obj.ref_count_down()
                return None
            
            # Copy from shared memory into memory_obj buffer
            memoryview(memory_obj.byte_array).cast("B")[:] = shm.buf
            return memory_obj
            
        except Exception as e:
            logger.error("GCS download failed for %s: %s", object_name, e)
            memory_obj.ref_count_down()
            raise
        finally:
            shm.close()
            shm.unlink()
        
    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        """
        Upload a `MemoryObj` as a finalized appendable object.
        """
        object_name = self._object_name(key)
        
        data_view = memoryview(memory_obj.byte_array).cast("B")
        size = len(data_view)
        
        # Create shared memory
        shm = shared_memory.SharedMemory(create=True, size=size)
        try:
            # Copy data into shared memory
            shm.buf[:size] = data_view
            
            future = self.loop.run_in_executor(
                self.put_executor, _worker_put, self.bucket_name, object_name, shm.name, size
            )
            
            # Fire-and-forget callback to clean up the shared memory
            # This allows the background upload task to proceed without blocking the caller
            def _cleanup(f):
                try:
                    f.result()
                except Exception as e:
                    logger.error("GCS upload failed for %s: %s", object_name, e)
                finally:
                    shm.close()
                    shm.unlink()
                    
            future.add_done_callback(_cleanup)
        except Exception as e:
            # If an error happens before run_in_executor successfully launches
            shm.close()
            shm.unlink()
            logger.error("GCS upload failed for %s: %s", object_name, e)
            raise
    
    async def list(self) -> List[str]:
        """
        List operation is not implemented for day-1.
        """
        raise NotImplementedError("GcsConnector.list() is not implemented")
    
    async def close(self) -> None:
        """
        Close the process pool executor.
        """
        if getattr(self, "get_executor", None) is not None:
            self.get_executor.shutdown(wait=False)
            self.get_executor = None
        if getattr(self, "put_executor", None) is not None:
            self.put_executor.shutdown(wait=False)
            self.put_executor = None