"""
Connector Manager

Manages multiple connector instances, routing messages,
and coordinating lifecycle across platforms.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from .base import (
    Connector,
    ConnectorConfig,
    ConnectorState,
    Message,
    Channel,
    MessageHandler,
)
from .registry import ConnectorRegistry


@dataclass
class RoutingRule:
    """Rule for routing outbound messages."""

    name: str
    platform: str
    channel_id: str
    condition: Callable[[Message], bool] | None = None
    priority: int = 0


class ConnectorManager:
    """
    Manages multiple connectors with unified message handling.

    Features:
    - Lifecycle management for all connectors
    - Unified message dispatch
    - Cross-platform routing
    - Health monitoring
    """

    def __init__(self):
        self._connectors: dict[str, Connector] = {}
        self._handlers: list[MessageHandler] = []
        self._routing_rules: list[RoutingRule] = []
        self._tasks: dict[str, asyncio.Task] = {}
        self._started = False

    @property
    def connectors(self) -> dict[str, Connector]:
        return self._connectors.copy()

    def add_connector(self, connector: Connector) -> None:
        """Add a connector to the manager."""
        if connector.connector_id in self._connectors:
            raise ValueError(f"Connector already exists: {connector.connector_id}")

        self._connectors[connector.connector_id] = connector
        connector.add_handler(self._on_message)

    def remove_connector(self, connector_id: str) -> Connector | None:
        """Remove a connector from the manager."""
        connector = self._connectors.pop(connector_id, None)
        if connector:
            connector.remove_handler(self._on_message)
        return connector

    def get_connector(self, connector_id: str) -> Connector | None:
        """Get a connector by ID."""
        return self._connectors.get(connector_id)

    def get_connectors_by_platform(self, platform: str) -> list[Connector]:
        """Get all connectors for a platform."""
        return [
            c for c in self._connectors.values()
            if c.platform.lower() == platform.lower()
        ]

    def create_connector(self, config: ConnectorConfig) -> Connector:
        """Create and add a connector from config."""
        connector = ConnectorRegistry.create(config)
        self.add_connector(connector)
        return connector

    def add_handler(self, handler: MessageHandler) -> None:
        """Add a global message handler."""
        self._handlers.append(handler)

    def remove_handler(self, handler: MessageHandler) -> None:
        """Remove a global message handler."""
        if handler in self._handlers:
            self._handlers.remove(handler)

    def add_routing_rule(self, rule: RoutingRule) -> None:
        """Add an outbound routing rule."""
        self._routing_rules.append(rule)
        self._routing_rules.sort(key=lambda r: -r.priority)

    async def _on_message(self, message: Message) -> None:
        """Internal handler for incoming messages."""
        for handler in self._handlers:
            try:
                result = handler(message)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                pass

    async def send(
        self,
        connector_id: str,
        channel_id: str,
        content: str,
        **kwargs: Any,
    ) -> Message:
        """Send a message through a specific connector."""
        connector = self._connectors.get(connector_id)
        if connector is None:
            raise ValueError(f"Connector not found: {connector_id}")

        channel = await connector.get_channel(channel_id)
        if channel is None:
            raise ValueError(f"Channel not found: {channel_id}")

        return await connector.send(channel, content, **kwargs)

    async def broadcast(
        self,
        content: str,
        platforms: list[str] | None = None,
        **kwargs: Any,
    ) -> list[Message]:
        """Broadcast a message to all connected platforms."""
        messages = []

        for connector in self._connectors.values():
            if platforms and connector.platform.lower() not in [p.lower() for p in platforms]:
                continue

            if not connector.is_connected:
                continue

            for rule in self._routing_rules:
                if rule.platform.lower() == connector.platform.lower():
                    try:
                        channel = await connector.get_channel(rule.channel_id)
                        if channel:
                            msg = await connector.send(channel, content, **kwargs)
                            messages.append(msg)
                    except Exception:
                        pass

        return messages

    async def start(self) -> None:
        """Start all connectors."""
        if self._started:
            return

        self._started = True

        for connector_id, connector in self._connectors.items():
            if connector.config.enabled:
                task = asyncio.create_task(
                    connector.run(),
                    name=f"connector_{connector_id}",
                )
                self._tasks[connector_id] = task

    async def stop(self) -> None:
        """Stop all connectors gracefully."""
        self._started = False

        for connector in self._connectors.values():
            await connector.shutdown()

        for task in self._tasks.values():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._tasks.clear()

    async def __aenter__(self) -> "ConnectorManager":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()

    def get_status(self) -> dict[str, Any]:
        """Get status of all connectors."""
        return {
            "started": self._started,
            "connector_count": len(self._connectors),
            "connectors": {
                cid: connector.get_status()
                for cid, connector in self._connectors.items()
            },
            "routing_rules": len(self._routing_rules),
        }

    def get_health(self) -> dict[str, Any]:
        """Get health status for monitoring."""
        connected = sum(
            1 for c in self._connectors.values()
            if c.state == ConnectorState.CONNECTED
        )
        total = len(self._connectors)

        return {
            "healthy": connected == total and total > 0,
            "connected": connected,
            "total": total,
            "platforms": list(set(c.platform for c in self._connectors.values())),
        }
