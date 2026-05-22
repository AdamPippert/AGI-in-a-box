"""
Base Connector Abstractions

Defines the unified interface for all platform connectors and
normalized message types for cross-platform communication.
"""

from __future__ import annotations

import asyncio
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Callable


class ConnectorState(Enum):
    """Lifecycle states for a connector."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class MessageType(Enum):
    """Types of messages that can flow through connectors."""

    TEXT = "text"
    COMMAND = "command"
    REPLY = "reply"
    REACTION = "reaction"
    EDIT = "edit"
    DELETE = "delete"
    SYSTEM = "system"
    FILE = "file"


@dataclass
class User:
    """Normalized user representation across platforms."""

    id: str
    username: str
    display_name: str = ""
    platform: str = ""
    is_bot: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.username


@dataclass
class Channel:
    """Normalized channel/conversation representation."""

    id: str
    name: str
    platform: str = ""
    is_private: bool = False
    is_group: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Attachment:
    """File or media attachment."""

    id: str
    filename: str
    content_type: str
    size_bytes: int
    url: str | None = None
    data: bytes | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    """
    Normalized message format across all platforms.

    This is the canonical representation that connectors translate
    to/from their platform-specific formats.
    """

    id: str
    content: str
    author: User
    channel: Channel
    timestamp: datetime
    message_type: MessageType = MessageType.TEXT
    reply_to_id: str | None = None
    attachments: list[Attachment] = field(default_factory=list)
    mentions: list[User] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: Any = None  # Original platform message

    @classmethod
    def create(
        cls,
        content: str,
        author: User,
        channel: Channel,
        **kwargs: Any,
    ) -> Message:
        """Factory method to create a new message."""
        return cls(
            id=kwargs.get("id", f"msg_{uuid.uuid4().hex[:12]}"),
            content=content,
            author=author,
            channel=channel,
            timestamp=kwargs.get("timestamp", datetime.now()),
            **{k: v for k, v in kwargs.items() if k not in ("id", "timestamp")},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "author": {
                "id": self.author.id,
                "username": self.author.username,
                "display_name": self.author.display_name,
            },
            "channel": {
                "id": self.channel.id,
                "name": self.channel.name,
            },
            "timestamp": self.timestamp.isoformat(),
            "message_type": self.message_type.value,
            "reply_to_id": self.reply_to_id,
            "attachments": [
                {"id": a.id, "filename": a.filename, "content_type": a.content_type}
                for a in self.attachments
            ],
        }


@dataclass
class ConnectorConfig:
    """Base configuration for connectors."""

    connector_id: str
    platform: str
    enabled: bool = True
    auto_reconnect: bool = True
    reconnect_delay_seconds: float = 5.0
    max_reconnect_attempts: int = 10
    rate_limit_per_second: float = 5.0
    metadata: dict[str, Any] = field(default_factory=dict)


MessageHandler = Callable[[Message], Any]


class Connector(ABC):
    """
    Abstract base class for platform connectors.

    Each connector translates between platform-specific protocols
    and the normalized Message format.
    """

    def __init__(self, config: ConnectorConfig):
        self.config = config
        self._state = ConnectorState.DISCONNECTED
        self._handlers: list[MessageHandler] = []
        self._reconnect_attempts = 0
        self._last_error: str | None = None

    @property
    def connector_id(self) -> str:
        return self.config.connector_id

    @property
    def platform(self) -> str:
        return self.config.platform

    @property
    def state(self) -> ConnectorState:
        return self._state

    @property
    def is_connected(self) -> bool:
        return self._state == ConnectorState.CONNECTED

    def add_handler(self, handler: MessageHandler) -> None:
        """Register a message handler."""
        self._handlers.append(handler)

    def remove_handler(self, handler: MessageHandler) -> None:
        """Remove a message handler."""
        if handler in self._handlers:
            self._handlers.remove(handler)

    async def _dispatch(self, message: Message) -> None:
        """Dispatch message to all registered handlers."""
        for handler in self._handlers:
            try:
                result = handler(message)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                pass  # Handler errors shouldn't break the connector

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the platform."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Gracefully disconnect from the platform."""
        pass

    @abstractmethod
    async def send(self, channel: Channel, content: str, **kwargs: Any) -> Message:
        """Send a message to a channel."""
        pass

    @abstractmethod
    async def reply(self, message: Message, content: str, **kwargs: Any) -> Message:
        """Reply to a specific message."""
        pass

    @abstractmethod
    async def edit(self, message: Message, new_content: str) -> Message:
        """Edit an existing message."""
        pass

    @abstractmethod
    async def delete(self, message: Message) -> None:
        """Delete a message."""
        pass

    @abstractmethod
    async def get_channel(self, channel_id: str) -> Channel | None:
        """Get channel by ID."""
        pass

    @abstractmethod
    async def get_user(self, user_id: str) -> User | None:
        """Get user by ID."""
        pass

    @abstractmethod
    async def listen(self) -> AsyncIterator[Message]:
        """Yield incoming messages as they arrive."""
        pass

    async def run(self) -> None:
        """Main run loop with auto-reconnect."""
        while self._state != ConnectorState.SHUTDOWN:
            try:
                self._state = ConnectorState.CONNECTING
                await self.connect()
                self._state = ConnectorState.CONNECTED
                self._reconnect_attempts = 0

                async for message in self.listen():
                    await self._dispatch(message)

            except Exception as e:
                self._last_error = str(e)
                self._state = ConnectorState.ERROR

                if not self.config.auto_reconnect:
                    raise

                self._reconnect_attempts += 1
                if self._reconnect_attempts > self.config.max_reconnect_attempts:
                    self._state = ConnectorState.SHUTDOWN
                    raise

                self._state = ConnectorState.RECONNECTING
                await asyncio.sleep(
                    self.config.reconnect_delay_seconds * self._reconnect_attempts
                )

    async def shutdown(self) -> None:
        """Shutdown the connector."""
        self._state = ConnectorState.SHUTDOWN
        await self.disconnect()

    def get_status(self) -> dict[str, Any]:
        """Get connector status for monitoring."""
        return {
            "connector_id": self.connector_id,
            "platform": self.platform,
            "state": self._state.value,
            "reconnect_attempts": self._reconnect_attempts,
            "last_error": self._last_error,
        }
