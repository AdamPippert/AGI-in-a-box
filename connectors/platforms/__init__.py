"""
Platform-specific connector implementations.

Import this module to auto-register all platform connectors.
"""

from .discord import DiscordConnector, DiscordConfig
from .slack import SlackConnector, SlackConfig
from .email import EmailConnector, EmailConfig
from .telegram import TelegramConnector, TelegramConfig

__all__ = [
    "DiscordConnector",
    "DiscordConfig",
    "SlackConnector",
    "SlackConfig",
    "EmailConnector",
    "EmailConfig",
    "TelegramConnector",
    "TelegramConfig",
]
