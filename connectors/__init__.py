"""
AGI-in-a-Box Connector Framework

Platform-agnostic connector abstraction for agent communication channels.
Supports Discord, Slack, Email, Telegram with a unified message interface.
"""

from .base import (
    Connector,
    ConnectorConfig,
    ConnectorState,
    Message,
    MessageType,
    Attachment,
    User,
    Channel,
)
from .registry import ConnectorRegistry
from .manager import ConnectorManager

__all__ = [
    "Connector",
    "ConnectorConfig",
    "ConnectorState",
    "Message",
    "MessageType",
    "Attachment",
    "User",
    "Channel",
    "ConnectorRegistry",
    "ConnectorManager",
]
