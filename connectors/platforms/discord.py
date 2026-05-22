"""
Discord Connector

Implements the connector interface for Discord using discord.py.
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
class DiscordConfig(ConnectorConfig):
    """Discord-specific configuration."""

    token: str = ""
    guild_ids: list[str] = field(default_factory=list)
    command_prefix: str = "!"
    intents_message_content: bool = True
    intents_members: bool = False

    def __post_init__(self):
        self.platform = "discord"


@register_connector("discord")
class DiscordConnector(Connector):
    """
    Discord connector using discord.py.

    Requires: pip install discord.py
    """

    def __init__(self, config: DiscordConfig):
        super().__init__(config)
        self._config: DiscordConfig = config
        self._client = None
        self._message_queue: asyncio.Queue[Message] = asyncio.Queue()
        self._ready_event = asyncio.Event()

    def _get_client(self):
        """Lazy initialization of Discord client."""
        if self._client is not None:
            return self._client

        try:
            import discord
        except ImportError:
            raise ImportError("discord.py required: pip install discord.py")

        intents = discord.Intents.default()
        if self._config.intents_message_content:
            intents.message_content = True
        if self._config.intents_members:
            intents.members = True

        self._client = discord.Client(intents=intents)

        @self._client.event
        async def on_ready():
            self._ready_event.set()

        @self._client.event
        async def on_message(discord_message):
            if discord_message.author == self._client.user:
                return

            message = self._normalize_message(discord_message)
            await self._message_queue.put(message)

        return self._client

    def _normalize_message(self, discord_message) -> Message:
        """Convert Discord message to normalized format."""
        author = User(
            id=str(discord_message.author.id),
            username=discord_message.author.name,
            display_name=discord_message.author.display_name,
            platform="discord",
            is_bot=discord_message.author.bot,
        )

        channel = Channel(
            id=str(discord_message.channel.id),
            name=getattr(discord_message.channel, "name", "DM"),
            platform="discord",
            is_private=not hasattr(discord_message.channel, "guild"),
            is_group=hasattr(discord_message.channel, "guild"),
        )

        attachments = [
            Attachment(
                id=str(a.id),
                filename=a.filename,
                content_type=a.content_type or "application/octet-stream",
                size_bytes=a.size,
                url=a.url,
            )
            for a in discord_message.attachments
        ]

        msg_type = MessageType.TEXT
        if discord_message.content.startswith(self._config.command_prefix):
            msg_type = MessageType.COMMAND

        return Message(
            id=str(discord_message.id),
            content=discord_message.content,
            author=author,
            channel=channel,
            timestamp=discord_message.created_at,
            message_type=msg_type,
            reply_to_id=str(discord_message.reference.message_id)
            if discord_message.reference
            else None,
            attachments=attachments,
            raw=discord_message,
        )

    async def connect(self) -> None:
        """Connect to Discord."""
        if not self._config.token:
            raise ValueError("Discord token required")

        client = self._get_client()
        asyncio.create_task(client.start(self._config.token))
        await asyncio.wait_for(self._ready_event.wait(), timeout=30)

    async def disconnect(self) -> None:
        """Disconnect from Discord."""
        if self._client:
            await self._client.close()
            self._client = None
            self._ready_event.clear()

    async def send(self, channel: Channel, content: str, **kwargs: Any) -> Message:
        """Send a message to a Discord channel."""
        if not self._client:
            raise RuntimeError("Not connected")

        discord_channel = self._client.get_channel(int(channel.id))
        if not discord_channel:
            discord_channel = await self._client.fetch_channel(int(channel.id))

        discord_message = await discord_channel.send(content)
        return self._normalize_message(discord_message)

    async def reply(self, message: Message, content: str, **kwargs: Any) -> Message:
        """Reply to a Discord message."""
        if not self._client or not message.raw:
            raise RuntimeError("Not connected or invalid message")

        discord_message = await message.raw.reply(content)
        return self._normalize_message(discord_message)

    async def edit(self, message: Message, new_content: str) -> Message:
        """Edit a Discord message."""
        if not message.raw:
            raise ValueError("Cannot edit message without raw reference")

        edited = await message.raw.edit(content=new_content)
        return self._normalize_message(edited)

    async def delete(self, message: Message) -> None:
        """Delete a Discord message."""
        if message.raw:
            await message.raw.delete()

    async def get_channel(self, channel_id: str) -> Channel | None:
        """Get a Discord channel by ID."""
        if not self._client:
            return None

        try:
            discord_channel = self._client.get_channel(int(channel_id))
            if not discord_channel:
                discord_channel = await self._client.fetch_channel(int(channel_id))

            return Channel(
                id=str(discord_channel.id),
                name=getattr(discord_channel, "name", "DM"),
                platform="discord",
                is_private=not hasattr(discord_channel, "guild"),
                is_group=hasattr(discord_channel, "guild"),
            )
        except Exception:
            return None

    async def get_user(self, user_id: str) -> User | None:
        """Get a Discord user by ID."""
        if not self._client:
            return None

        try:
            discord_user = await self._client.fetch_user(int(user_id))
            return User(
                id=str(discord_user.id),
                username=discord_user.name,
                display_name=discord_user.display_name,
                platform="discord",
                is_bot=discord_user.bot,
            )
        except Exception:
            return None

    async def listen(self) -> AsyncIterator[Message]:
        """Yield incoming Discord messages."""
        while self._state not in (ConnectorState.SHUTDOWN, ConnectorState.ERROR):
            try:
                message = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=1.0,
                )
                yield message
            except asyncio.TimeoutError:
                continue
