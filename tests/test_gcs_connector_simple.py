import asyncio
import torch
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.connector.gcs_connector import GCSConnector

async def main():
    print("Initializing GCS Connector test...")
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_cpu=True,
        max_local_cpu_size=1.0,
        remote_url="gs://shiqi-rapid-west4-agent"
    )
    metadata = LMCacheMetadata(
        model_name="test-model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(64, 2, 256, 8, 128),
        use_mla=False,
        role="worker"
    )
    
    loop = asyncio.get_running_loop()
    
    # 1. Create LocalCPUBackend
    cpu_backend = LocalCPUBackend(config, metadata, "cpu")
    
    # 2. Instantiate GCSConnector
    print("Creating GCSConnector...")
    connector = GCSConnector(
        loop=loop,
        local_cpu_backend=cpu_backend,
        config=config,
        bucket_name_str="shiqi-rapid-west4-agent"
    )
    print("GCSConnector created successfully!")
    
    # 3. Test exists
    key = CacheEngineKey("test-model", 1, 0, 42, torch.bfloat16)
    print("Testing exists...")
    has_key = await connector.exists(key)
    print(f"Exists result: {has_key}")
    
    # 4. Test put/get
    print("Allocating MemoryObj...")
    mem_obj = cpu_backend.allocate(connector.meta_shapes, connector.meta_dtypes, connector.meta_fmt)
    # fill raw_tensor with dummy values
    mem_obj.raw_tensor.fill_(3.14)
    
    print("Testing put...")
    await connector.put(key, mem_obj)
    print("Put complete!")
    
    print("Testing exists after put...")
    has_key_after = await connector.exists(key)
    print(f"Exists result after put: {has_key_after}")
    
    print("Testing get...")
    retrieved_obj = await connector.get(key)
    if retrieved_obj is not None:
        print("Get successful!")
        print(f"Is equal: {torch.equal(mem_obj.raw_tensor, retrieved_obj.raw_tensor)}")
        retrieved_obj.ref_count_down()
    else:
        print("Get failed (None)")
        
    mem_obj.ref_count_down()
    print("Done!")

if __name__ == "__main__":
    asyncio.run(main())
