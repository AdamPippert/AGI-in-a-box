"""
Email Connector

Implements the connector interface for Email using SMTP/IMAP.
"""

from __future__ import annotations

import asyncio
import email
import imaplib
import smtplib
import ssl
from dataclasses import dataclass, field
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import parsedate_to_datetime
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
class EmailConfig(ConnectorConfig):
    """Email-specific configuration."""

    imap_host: str = ""
    imap_port: int = 993
    smtp_host: str = ""
    smtp_port: int = 587
    username: str = ""
    password: str = ""
    use_ssl: bool = True
    use_tls: bool = True
    mailbox: str = "INBOX"
    poll_interval_seconds: float = 30.0
    mark_as_read: bool = True

    def __post_init__(self):
        self.platform = "email"


@register_connector("email")
class EmailConnector(Connector):
    """
    Email connector using SMTP for sending and IMAP for receiving.

    Uses standard library - no extra dependencies required.
    """

    def __init__(self, config: EmailConfig):
        super().__init__(config)
        self._config: EmailConfig = config
        self._imap: imaplib.IMAP4_SSL | imaplib.IMAP4 | None = None
        self._smtp: smtplib.SMTP | smtplib.SMTP_SSL | None = None
        self._message_queue: asyncio.Queue[Message] = asyncio.Queue()
        self._poll_task: asyncio.Task | None = None
        self._seen_ids: set[str] = set()

    def _create_imap_connection(self) -> imaplib.IMAP4_SSL | imaplib.IMAP4:
        """Create IMAP connection."""
        if self._config.use_ssl:
            context = ssl.create_default_context()
            return imaplib.IMAP4_SSL(
                self._config.imap_host,
                self._config.imap_port,
                ssl_context=context,
            )
        return imaplib.IMAP4(self._config.imap_host, self._config.imap_port)

    def _create_smtp_connection(self) -> smtplib.SMTP | smtplib.SMTP_SSL:
        """Create SMTP connection."""
        if self._config.use_ssl and self._config.smtp_port == 465:
            context = ssl.create_default_context()
            smtp = smtplib.SMTP_SSL(
                self._config.smtp_host,
                self._config.smtp_port,
                context=context,
            )
        else:
            smtp = smtplib.SMTP(self._config.smtp_host, self._config.smtp_port)
            if self._config.use_tls:
                smtp.starttls()

        smtp.login(self._config.username, self._config.password)
        return smtp

    def _parse_email(self, msg_data: bytes, msg_id: str) -> Message:
        """Parse raw email into normalized Message."""
        msg = email.message_from_bytes(msg_data)

        from_header = msg.get("From", "")
        from_parts = email.utils.parseaddr(from_header)

        author = User(
            id=from_parts[1],
            username=from_parts[1],
            display_name=from_parts[0] or from_parts[1],
            platform="email",
        )

        to_header = msg.get("To", "")
        to_parts = email.utils.parseaddr(to_header)

        channel = Channel(
            id=to_parts[1],
            name=to_parts[1],
            platform="email",
            is_private=True,
        )

        date_str = msg.get("Date", "")
        try:
            timestamp = parsedate_to_datetime(date_str)
        except (ValueError, TypeError):
            timestamp = datetime.now()

        body = ""
        attachments = []

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                disposition = str(part.get("Content-Disposition", ""))

                if "attachment" in disposition:
                    attachments.append(
                        Attachment(
                            id=part.get_filename() or "attachment",
                            filename=part.get_filename() or "attachment",
                            content_type=content_type,
                            size_bytes=len(part.get_payload(decode=True) or b""),
                            data=part.get_payload(decode=True),
                        )
                    )
                elif content_type == "text/plain" and not body:
                    payload = part.get_payload(decode=True)
                    if payload:
                        body = payload.decode("utf-8", errors="replace")
                elif content_type == "text/html" and not body:
                    payload = part.get_payload(decode=True)
                    if payload:
                        body = payload.decode("utf-8", errors="replace")
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                body = payload.decode("utf-8", errors="replace")

        subject = msg.get("Subject", "")
        if subject:
            body = f"Subject: {subject}\n\n{body}"

        reply_to = None
        in_reply_to = msg.get("In-Reply-To", "")
        if in_reply_to:
            reply_to = in_reply_to.strip("<>")

        return Message(
            id=msg_id,
            content=body,
            author=author,
            channel=channel,
            timestamp=timestamp,
            message_type=MessageType.TEXT,
            reply_to_id=reply_to,
            attachments=attachments,
            raw=msg,
            metadata={
                "subject": subject,
                "message_id": msg.get("Message-ID", ""),
            },
        )

    async def _poll_inbox(self) -> None:
        """Poll IMAP inbox for new messages."""
        while self._state == ConnectorState.CONNECTED:
            try:
                await asyncio.to_thread(self._check_mail)
            except Exception:
                pass

            await asyncio.sleep(self._config.poll_interval_seconds)

    def _check_mail(self) -> None:
        """Check for new mail (runs in thread)."""
        if not self._imap:
            return

        self._imap.select(self._config.mailbox)
        _, data = self._imap.search(None, "UNSEEN")

        for num in data[0].split():
            msg_id = num.decode()
            if msg_id in self._seen_ids:
                continue

            _, msg_data = self._imap.fetch(num, "(RFC822)")
            if msg_data[0] is None:
                continue

            raw_email = msg_data[0][1]
            message = self._parse_email(raw_email, msg_id)

            self._seen_ids.add(msg_id)
            asyncio.get_event_loop().call_soon_threadsafe(
                self._message_queue.put_nowait, message
            )

            if self._config.mark_as_read:
                self._imap.store(num, "+FLAGS", "\\Seen")

    async def connect(self) -> None:
        """Connect to email servers."""
        if not self._config.imap_host or not self._config.smtp_host:
            raise ValueError("IMAP and SMTP hosts required")

        self._imap = await asyncio.to_thread(self._create_imap_connection)
        await asyncio.to_thread(
            self._imap.login, self._config.username, self._config.password
        )

        self._poll_task = asyncio.create_task(self._poll_inbox())

    async def disconnect(self) -> None:
        """Disconnect from email servers."""
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass

        if self._imap:
            try:
                self._imap.logout()
            except Exception:
                pass
            self._imap = None

        if self._smtp:
            try:
                self._smtp.quit()
            except Exception:
                pass
            self._smtp = None

    async def send(self, channel: Channel, content: str, **kwargs: Any) -> Message:
        """Send an email."""
        subject = kwargs.get("subject", "Message from AGI-in-a-Box")

        msg = MIMEMultipart()
        msg["From"] = self._config.username
        msg["To"] = channel.id
        msg["Subject"] = subject
        msg.attach(MIMEText(content, "plain"))

        smtp = await asyncio.to_thread(self._create_smtp_connection)
        try:
            await asyncio.to_thread(smtp.send_message, msg)
        finally:
            smtp.quit()

        return Message(
            id=f"sent_{datetime.now().timestamp()}",
            content=content,
            author=User(
                id=self._config.username,
                username=self._config.username,
                platform="email",
            ),
            channel=channel,
            timestamp=datetime.now(),
            message_type=MessageType.TEXT,
            metadata={"subject": subject},
        )

    async def reply(self, message: Message, content: str, **kwargs: Any) -> Message:
        """Reply to an email."""
        original_subject = message.metadata.get("subject", "")
        subject = f"Re: {original_subject}" if not original_subject.startswith("Re:") else original_subject

        reply_to_channel = Channel(
            id=message.author.id,
            name=message.author.username,
            platform="email",
        )

        return await self.send(
            reply_to_channel,
            content,
            subject=subject,
            in_reply_to=message.metadata.get("message_id"),
            **kwargs,
        )

    async def edit(self, message: Message, new_content: str) -> Message:
        """Email doesn't support editing."""
        raise NotImplementedError("Email messages cannot be edited")

    async def delete(self, message: Message) -> None:
        """Delete an email (move to trash)."""
        if not self._imap:
            raise RuntimeError("Not connected")

        await asyncio.to_thread(
            self._imap.store, message.id.encode(), "+FLAGS", "\\Deleted"
        )
        await asyncio.to_thread(self._imap.expunge)

    async def get_channel(self, channel_id: str) -> Channel | None:
        """Get a channel (email address) by ID."""
        return Channel(
            id=channel_id,
            name=channel_id,
            platform="email",
            is_private=True,
        )

    async def get_user(self, user_id: str) -> User | None:
        """Get a user (email address) by ID."""
        return User(
            id=user_id,
            username=user_id,
            platform="email",
        )

    async def listen(self) -> AsyncIterator[Message]:
        """Yield incoming emails."""
        while self._state not in (ConnectorState.SHUTDOWN, ConnectorState.ERROR):
            try:
                message = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=1.0,
                )
                yield message
            except asyncio.TimeoutError:
                continue
