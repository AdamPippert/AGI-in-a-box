"""
Connector Registry

Central registry for connector types and factory methods.
Supports plugin-style connector registration.
"""

from __future__ import annotations

from typing import Type

from .base import Connector, ConnectorConfig


class ConnectorRegistry:
    """
    Registry for connector implementations.

    Allows dynamic registration of connector types and
    factory-based instantiation.
    """

    _connectors: dict[str, Type[Connector]] = {}

    @classmethod
    def register(cls, platform: str, connector_class: Type[Connector]) -> None:
        """Register a connector class for a platform."""
        cls._connectors[platform.lower()] = connector_class

    @classmethod
    def unregister(cls, platform: str) -> None:
        """Remove a connector registration."""
        cls._connectors.pop(platform.lower(), None)

    @classmethod
    def get(cls, platform: str) -> Type[Connector] | None:
        """Get connector class for a platform."""
        return cls._connectors.get(platform.lower())

    @classmethod
    def create(cls, config: ConnectorConfig) -> Connector:
        """
        Create a connector instance from config.

        Args:
            config: Connector configuration with platform specified

        Returns:
            Instantiated connector

        Raises:
            ValueError: If platform is not registered
        """
        connector_class = cls.get(config.platform)
        if connector_class is None:
            available = list(cls._connectors.keys())
            raise ValueError(
                f"Unknown platform: {config.platform}. Available: {available}"
            )
        return connector_class(config)

    @classmethod
    def list_platforms(cls) -> list[str]:
        """List all registered platforms."""
        return list(cls._connectors.keys())

    @classmethod
    def is_registered(cls, platform: str) -> bool:
        """Check if a platform is registered."""
        return platform.lower() in cls._connectors


def register_connector(platform: str):
    """Decorator to register a connector class."""

    def decorator(cls: Type[Connector]) -> Type[Connector]:
        ConnectorRegistry.register(platform, cls)
        return cls

    return decorator
