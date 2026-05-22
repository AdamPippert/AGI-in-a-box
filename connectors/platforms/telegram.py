"""
Telegram Connector

Implements the connector interface for Telegram using python-telegram-bot.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
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
class TelegramConfig(ConnectorConfig):
    """Telegram-specific configuration."""

    bot_token: str = ""
    allowed_user_ids: list[int] | None = None
    allowed_chat_ids: list[int] | None = None

    def __post_init__(self):
        self.platform = "telegram"


@register_connector("telegram")
class TelegramConnector(Connector):
    """
    Telegram connector using python-telegram-bot.

    Requires: pip install python-telegram-bot
    """

    def __init__(self, config: TelegramConfig):
        super().__init__(config)
        self._config: TelegramConfig = config
        self._app = None
        self._message_queue: asyncio.Queue[Message] = asyncio.Queue()

    def _get_app(self):
        """Lazy initialization of Telegram application."""
        if self._app is not None:
            return self._app

        try:
            from telegram.ext import Application, MessageHandler, filters
        except ImportError:
            raise ImportError(
                "python-telegram-bot required: pip install python-telegram-bot"
            )

        self._app = Application.builder().token(self._config.bot_token).build()

        async def handle_message(update, context):
            if update.message is None:
                return

            if self._config.allowed_user_ids:
                if update.message.from_user.id not in self._config.allowed_user_ids:
                    return

            if self._config.allowed_chat_ids:
                if update.message.chat.id not in self._config.allowed_chat_ids:
                    return

            normalized = self._normalize_message(update.message)
            await self._message_queue.put(normalized)

        self._app.add_handler(MessageHandler(filters.ALL, handle_message))
        return self._app

    def _normalize_message(self, tg_message) -> Message:
        """Convert Telegram message to normalized format."""
        author = User(
            id=str(tg_message.from_user.id),
            username=tg_message.from_user.username or str(tg_message.from_user.id),
            display_name=tg_message.from_user.full_name,
            platform="telegram",
            is_bot=tg_message.from_user.is_bot,
        )

        chat = tg_message.chat
        channel = Channel(
            id=str(chat.id),
            name=chat.title or chat.username or str(chat.id),
            platform="telegram",
            is_private=chat.type == "private",
            is_group=chat.type in ("group", "supergroup"),
        )

        content = tg_message.text or tg_message.caption or ""

        attachments = []
        if tg_message.document:
            attachments.append(
                Attachment(
                    id=tg_message.document.file_id,
                    filename=tg_message.document.file_name or "document",
                    content_type=tg_message.document.mime_type or "application/octet-stream",
                    size_bytes=tg_message.document.file_size or 0,
                )
            )
        if tg_message.photo:
            photo = tg_message.photo[-1]  # Largest size
            attachments.append(
                Attachment(
                    id=photo.file_id,
                    filename="photo.jpg",
                    content_type="image/jpeg",
                    size_bytes=photo.file_size or 0,
                )
            )

        msg_type = MessageType.TEXT
        if content.startswith("/"):
            msg_type = MessageType.COMMAND

        return Message(
            id=str(tg_message.message_id),
            content=content,
            author=author,
            channel=channel,
            timestamp=tg_message.date,
            message_type=msg_type,
            reply_to_id=str(tg_message.reply_to_message.message_id)
            if tg_message.reply_to_message
            else None,
            attachments=attachments,
            raw=tg_message,
        )

    async def connect(self) -> None:
        """Connect to Telegram."""
        if not self._config.bot_token:
            raise ValueError("Telegram bot token required")

        app = self._get_app()
        await app.initialize()
        await app.start()
        asyncio.create_task(app.updater.start_polling())

    async def disconnect(self) -> None:
        """Disconnect from Telegram."""
        if self._app:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
            self._app = None

    async def send(self, channel: Channel, content: str, **kwargs: Any) -> Message:
        """Send a message to a Telegram chat."""
        if not self._app:
            raise RuntimeError("Not connected")

        tg_message = await self._app.bot.send_message(
            chat_id=int(channel.id),
            text=content,
            **kwargs,
        )

        return self._normalize_message(tg_message)

    async def reply(self, message: Message, content: str, **kwargs: Any) -> Message:
        """Reply to a Telegram message."""
        if not self._app:
            raise RuntimeError("Not connected")

        tg_message = await self._app.bot.send_message(
            chat_id=int(message.channel.id),
            text=content,
            reply_to_message_id=int(message.id),
            **kwargs,
        )

        return self._normalize_message(tg_message)

    async def edit(self, message: Message, new_content: str) -> Message:
        """Edit a Telegram message."""
        if not self._app:
            raise RuntimeError("Not connected")

        tg_message = await self._app.bot.edit_message_text(
            chat_id=int(message.channel.id),
            message_id=int(message.id),
            text=new_content,
        )

        return self._normalize_message(tg_message)

    async def delete(self, message: Message) -> None:
        """Delete a Telegram message."""
        if not self._app:
            raise RuntimeError("Not connected")

        await self._app.bot.delete_message(
            chat_id=int(message.channel.id),
            message_id=int(message.id),
        )

    async def get_channel(self, channel_id: str) -> Channel | None:
        """Get a Telegram chat by ID."""
        if not self._app:
            return None

        try:
            chat = await self._app.bot.get_chat(int(channel_id))
            return Channel(
                id=str(chat.id),
                name=chat.title or chat.username or str(chat.id),
                platform="telegram",
                is_private=chat.type == "private",
                is_group=chat.type in ("group", "supergroup"),
            )
        except Exception:
            return None

    async def get_user(self, user_id: str) -> User | None:
        """Get a Telegram user by ID."""
        if not self._app:
            return None

        try:
            chat = await self._app.bot.get_chat(int(user_id))
            return User(
                id=str(chat.id),
                username=chat.username or str(chat.id),
                display_name=chat.full_name if hasattr(chat, "full_name") else "",
                platform="telegram",
            )
        except Exception:
            return None

    async def listen(self) -> AsyncIterator[Message]:
        """Yield incoming Telegram messages."""
        while self._state not in (ConnectorState.SHUTDOWN, ConnectorState.ERROR):
            try:
                message = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=1.0,
                )
                yield message
            except asyncio.TimeoutError:
                continue
