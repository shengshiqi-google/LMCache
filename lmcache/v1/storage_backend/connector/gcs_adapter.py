# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.connector import (
    ConnectorAdapter,
    ConnectorContext,
)
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.connector.gcs_connector import GcsConnector

logger = init_logger(__name__)


class GcsConnectorAdapter(ConnectorAdapter):
    """
    Adapter that registers the ``gs://`` URL scheme to construct a 
    `GcsConnector`.

    Auto-discovered by `ConnectorManager` because this module is named
    ``*_adapter.py`` and this class ubsclasses `ConnectorAdapter`.
    """

    def __init__(self) -> None:
        """
        Register the adapter under the ``gs://`` URL scheme.
        """
        super().__init__(schema="gs://")

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        """
        Construct a `GcsConnector` from a `ConnectorContext`.

        Args:
            context: The connector context. Must carry a non-None
            `config` and `metadata`; both are required by the
            `RemoteConnector` base class.
        
        Returns:
            A configured `GcsConnector`.
        
        Raises:
            ValueError: If `context.config` or `context.metadata` is
                None.
        """
        if context.config is None:
            raise ValueError(
                "GcsConnector requires a non-None config"
            )

        if context.metadata is None:
            raise ValueError(
                "GcsConnector requires a non-None metadata"
            )

        return GcsConnector(
            url=context.url,
            loop=context.loop,
            local_cpu_backend=context.local_cpu_backend,
            config=context.config,
            metadata=context.metadata,
        )