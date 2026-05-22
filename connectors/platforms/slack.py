"""
Slack Connector

Implements the connector interface for Slack using slack-bolt.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator

from ..base import (
    Connector,
    ConnectorConfig,
    ConnectorState,
    Message,
    MessageType,
    User,
    Channel,
    Attachment,
)
from ..registry import register_connector


@dataclass
class SlackConfig(ConnectorConfig):
    """Slack-specific configuration."""

    bot_token: str = ""
    app_token: str = ""
    signing_secret: str = ""
    socket_mode: bool = True

    def __post_init__(self):
        self.platform = "slack"


@register_connector("slack")
class SlackConnector(Connector):
    """
    Slack connector using slack-bolt.

    Requires: pip install slack-bolt
    """

    def __init__(self, config: SlackConfig):
        super().__init__(config)
        self._config: SlackConfig = config
        self._app = None
        self._handler = None
        self._client = None
        self._message_queue: asyncio.Queue[Message] = asyncio.Queue()
        self._user_cache: dict[str, User] = {}
        self._channel_cache: dict[str, Channel] = {}

    def _get_app(self):
        """Lazy initialization of Slack app."""
        if self._app is not None:
            return self._app

        try:
            from slack_bolt.async_app import AsyncApp
            from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
        except ImportError:
            raise ImportError("slack-bolt required: pip install slack-bolt")

        self._app = AsyncApp(
            token=self._config.bot_token,
            signing_secret=self._config.signing_secret,
        )

        @self._app.message("")
        async def handle_message(message, say, client):
            normalized = await self._normalize_message(message, client)
            await self._message_queue.put(normalized)

        self._client = self._app.client
        return self._app

    async def _normalize_message(self, slack_message: dict, client) -> Message:
        """Convert Slack message to normalized format."""
        user_id = slack_message.get("user", "")
        channel_id = slack_message.get("channel", "")

        author = await self._get_or_fetch_user(user_id, client)
        channel = await self._get_or_fetch_channel(channel_id, client)

        attachments = []
        for file in slack_message.get("files", []):
            attachments.append(
                Attachment(
                    id=file.get("id", ""),
                    filename=file.get("name", ""),
                    content_type=file.get("mimetype", "application/octet-stream"),
                    size_bytes=file.get("size", 0),
                    url=file.get("url_private", ""),
                )
            )

        timestamp = datetime.fromtimestamp(float(slack_message.get("ts", 0)))

        return Message(
            id=slack_message.get("ts", ""),
            content=slack_message.get("text", ""),
            author=author,
            channel=channel,
            timestamp=timestamp,
            message_type=MessageType.TEXT,
            reply_to_id=slack_message.get("thread_ts"),
            attachments=attachments,
            raw=slack_message,
        )

    async def _get_or_fetch_user(self, user_id: str, client) -> User:
        """Get user from cache or fetch from Slack."""
        if user_id in self._user_cache:
            return self._user_cache[user_id]

        try:
            result = await client.users_info(user=user_id)
            user_data = result.get("user", {})
            user = User(
                id=user_id,
                username=user_data.get("name", user_id),
                display_name=user_data.get("real_name", ""),
                platform="slack",
                is_bot=user_data.get("is_bot", False),
            )
            self._user_cache[user_id] = user
            return user
        except Exception:
            return User(id=user_id, username=user_id, platform="slack")

    async def _get_or_fetch_channel(self, channel_id: str, client) -> Channel:
        """Get channel from cache or fetch from Slack."""
        if channel_id in self._channel_cache:
            return self._channel_cache[channel_id]

        try:
            result = await client.conversations_info(channel=channel_id)
            channel_data = result.get("channel", {})
            channel = Channel(
                id=channel_id,
                name=channel_data.get("name", channel_id),
                platform="slack",
                is_private=channel_data.get("is_private", False),
                is_group=channel_data.get("is_group", False),
            )
            self._channel_cache[channel_id] = channel
            return channel
        except Exception:
            return Channel(id=channel_id, name=channel_id, platform="slack")

    async def connect(self) -> None:
        """Connect to Slack."""
        if not self._config.bot_token:
            raise ValueError("Slack bot token required")

        app = self._get_app()

        if self._config.socket_mode:
            if not self._config.app_token:
                raise ValueError("App token required for socket mode")

            from slack_bolt.adapter.socket_mode.async_handler import (
                AsyncSocketModeHandler,
            )

            self._handler = AsyncSocketModeHandler(app, self._config.app_token)
            asyncio.create_task(self._handler.start_async())

    async def disconnect(self) -> None:
        """Disconnect from Slack."""
        if self._handler:
            await self._handler.close_async()
            self._handler = None

    async def send(self, channel: Channel, content: str, **kwargs: Any) -> Message:
        """Send a message to a Slack channel."""
        if not self._client:
            raise RuntimeError("Not connected")

        result = await self._client.chat_postMessage(
            channel=channel.id,
            text=content,
            **kwargs,
        )

        return Message(
            id=result.get("ts", ""),
            content=content,
            author=User(id="bot", username="bot", platform="slack", is_bot=True),
            channel=channel,
            timestamp=datetime.now(),
            message_type=MessageType.TEXT,
        )

    async def reply(self, message: Message, content: str, **kwargs: Any) -> Message:
        """Reply to a Slack message (in thread)."""
        if not self._client:
            raise RuntimeError("Not connected")

        thread_ts = message.reply_to_id or message.id

        result = await self._client.chat_postMessage(
            channel=message.channel.id,
            text=content,
            thread_ts=thread_ts,
            **kwargs,
        )

        return Message(
            id=result.get("ts", ""),
            content=content,
            author=User(id="bot", username="bot", platform="slack", is_bot=True),
            channel=message.channel,
            timestamp=datetime.now(),
            message_type=MessageType.REPLY,
            reply_to_id=thread_ts,
        )

    async def edit(self, message: Message, new_content: str) -> Message:
        """Edit a Slack message."""
        if not self._client:
            raise RuntimeError("Not connected")

        await self._client.chat_update(
            channel=message.channel.id,
            ts=message.id,
            text=new_content,
        )

        message.content = new_content
        return message

    async def delete(self, message: Message) -> None:
        """Delete a Slack message."""
        if not self._client:
            raise RuntimeError("Not connected")

        await self._client.chat_delete(
            channel=message.channel.id,
            ts=message.id,
        )

    async def get_channel(self, channel_id: str) -> Channel | None:
        """Get a Slack channel by ID."""
        if not self._client:
            return None

        return await self._get_or_fetch_channel(channel_id, self._client)

    async def get_user(self, user_id: str) -> User | None:
        """Get a Slack user by ID."""
        if not self._client:
            return None

        return await self._get_or_fetch_user(user_id, self._client)

    async def listen(self) -> AsyncIterator[Message]:
        """Yield incoming Slack messages."""
        while self._state not in (ConnectorState.SHUTDOWN, ConnectorState.ERROR):
            try:
                message = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=1.0,
                )
                yield message
            except asyncio.TimeoutError:
                continue
