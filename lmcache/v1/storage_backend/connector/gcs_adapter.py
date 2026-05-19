# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.connector import (
    ConnectorAdapter,
    ConnectorContext,
    extract_plugin_type,
    parse_remote_url,
)
from lmcache.v1.storage_backend.connector.base_connector import (
    RemoteConnector,
)

logger = init_logger(__name__)

PLUGIN_TYPE = "gcs"

class GCSConnectorAdapter(ConnectorAdapter):
    """Adapter for Google Cloud Storage connectors."""

    def __init__(self) -> None:
        super().__init__("gs://")

    def can_parse(self, url: str) -> bool:
        if url.startswith(self.schema):
            return True
        if url.startswith("plugin://"):
            pname = url[len("plugin://") :]
            return extract_plugin_type(pname) == PLUGIN_TYPE
        return False

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        # Local
        from .gcs_connector import GCSConnector

        logger.info("Creating GCS connector")

        bucket_name_str = None
        if context.plugin_name is None:
            # Remote URL starts with gs://
            bucket_name_str = context.url.removeprefix("gs://")

        return GCSConnector(
            context.loop,
            context.local_cpu_backend,
            context.config,
            plugin_name=context.plugin_name,
            bucket_name_str=bucket_name_str,
        )
